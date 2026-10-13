"""SAC-Discrete comparator (Christodoulou 2019, arXiv:1910.07207).

SB3's SAC is continuous-only, so this is a minimal self-contained discrete SAC:
categorical actor, twin Q-critics with soft target updates, and auto-tuned
temperature against a target entropy of 0.98 * log|A|. Expectations over the
action distribution are computed exactly (no reparameterization needed in the
discrete case). Hyperparameters mirror the off-policy family in train.py
(lr 3e-4, buffer 100k, warmup 2k, batch 128, one gradient step every 4 env
steps) so the compute budget matches QR-DQN/DQN.

Exposes the SB3-ish surface train.py relies on: learn(), predict(), save().
"""
from __future__ import annotations

import numpy as np
import torch as th
from torch import nn
from torch.nn import functional as F


def _mlp(in_dim: int, out_dim: int, hidden: tuple[int, ...]) -> nn.Sequential:
    layers: list[nn.Module] = []
    last = in_dim
    for h in hidden:
        layers += [nn.Linear(last, h), nn.ReLU()]
        last = h
    layers.append(nn.Linear(last, out_dim))
    return nn.Sequential(*layers)


class _ReplayBuffer:
    def __init__(self, capacity: int, obs_dim: int, rng: np.random.Generator):
        self.capacity, self.rng = capacity, rng
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)  # terminal only, not truncation
        self.pos, self.full = 0, False

    def add(self, obs, action, reward, next_obs, done) -> None:
        i = self.pos
        self.obs[i], self.actions[i], self.rewards[i] = obs, action, reward
        self.next_obs[i], self.dones[i] = next_obs, float(done)
        self.pos = (self.pos + 1) % self.capacity
        self.full = self.full or self.pos == 0

    def sample(self, batch_size: int) -> tuple[th.Tensor, ...]:
        hi = self.capacity if self.full else self.pos
        idx = self.rng.integers(0, hi, size=batch_size)
        t = th.as_tensor
        return (t(self.obs[idx]), t(self.actions[idx]), t(self.rewards[idx]),
                t(self.next_obs[idx]), t(self.dones[idx]))


class SACDiscrete:
    def __init__(self, env, seed: int = 0, learning_rate: float = 3e-4,
                 buffer_size: int = 100_000, learning_starts: int = 2_000,
                 batch_size: int = 128, train_freq: int = 4, gamma: float = 0.99,
                 tau: float = 0.005, net_arch: tuple[int, ...] = (256, 256),
                 target_entropy_scale: float = 0.98):
        self.env = env
        self.gamma, self.tau = gamma, tau
        self.batch_size, self.train_freq = batch_size, train_freq
        self.learning_starts = learning_starts
        th.manual_seed(seed)
        self.rng = np.random.default_rng(seed)
        env.action_space.seed(seed)

        obs_dim = int(np.prod(env.observation_space.shape))
        self.n_actions = int(env.action_space.n)
        self.actor = _mlp(obs_dim, self.n_actions, net_arch)
        self.q1 = _mlp(obs_dim, self.n_actions, net_arch)
        self.q2 = _mlp(obs_dim, self.n_actions, net_arch)
        self.q1_t = _mlp(obs_dim, self.n_actions, net_arch)
        self.q2_t = _mlp(obs_dim, self.n_actions, net_arch)
        self.q1_t.load_state_dict(self.q1.state_dict())
        self.q2_t.load_state_dict(self.q2.state_dict())

        self.actor_opt = th.optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.q_opt = th.optim.Adam(list(self.q1.parameters()) + list(self.q2.parameters()),
                                   lr=learning_rate)
        self.log_alpha = th.zeros(1, requires_grad=True)
        self.alpha_opt = th.optim.Adam([self.log_alpha], lr=learning_rate)
        self.target_entropy = target_entropy_scale * float(np.log(self.n_actions))

        self.buffer = _ReplayBuffer(buffer_size, obs_dim, self.rng)

    def predict(self, obs, deterministic: bool = True):
        obs_t = th.as_tensor(np.asarray(obs, dtype=np.float32)).reshape(1, -1)
        with th.no_grad():
            logits = self.actor(obs_t)
            if deterministic:
                a = logits.argmax(dim=1)
            else:
                a = th.distributions.Categorical(logits=logits).sample()
        return a.numpy(), None

    def learn(self, total_timesteps: int, progress_bar: bool = False) -> "SACDiscrete":
        obs, _ = self.env.reset(seed=int(self.rng.integers(1 << 30)))
        for step in range(total_timesteps):
            if step < self.learning_starts:
                action = int(self.env.action_space.sample())
            else:
                action = int(self.predict(obs, deterministic=False)[0][0])
            next_obs, reward, term, trunc, _ = self.env.step(action)
            self.buffer.add(obs, action, reward, next_obs, term)
            obs = next_obs
            if term or trunc:
                obs, _ = self.env.reset()
            if step >= self.learning_starts and step % self.train_freq == 0:
                self._update()
        return self

    def _update(self) -> None:
        obs, actions, rewards, next_obs, dones = self.buffer.sample(self.batch_size)
        alpha = self.log_alpha.exp().detach()

        with th.no_grad():
            next_logits = self.actor(next_obs)
            next_logp = F.log_softmax(next_logits, dim=1)
            next_probs = next_logp.exp()
            q_next = th.min(self.q1_t(next_obs), self.q2_t(next_obs))
            v_next = (next_probs * (q_next - alpha * next_logp)).sum(dim=1)
            target = rewards + self.gamma * (1.0 - dones) * v_next

        idx = actions.unsqueeze(1)
        q1_a = self.q1(obs).gather(1, idx).squeeze(1)
        q2_a = self.q2(obs).gather(1, idx).squeeze(1)
        q_loss = F.mse_loss(q1_a, target) + F.mse_loss(q2_a, target)
        self.q_opt.zero_grad(); q_loss.backward(); self.q_opt.step()

        logits = self.actor(obs)
        logp = F.log_softmax(logits, dim=1)
        probs = logp.exp()
        with th.no_grad():
            q_min = th.min(self.q1(obs), self.q2(obs))
        actor_loss = (probs * (alpha * logp - q_min)).sum(dim=1).mean()
        self.actor_opt.zero_grad(); actor_loss.backward(); self.actor_opt.step()

        alpha_loss = -(probs.detach() * self.log_alpha *
                       (logp.detach() + self.target_entropy)).sum(dim=1).mean()
        self.alpha_opt.zero_grad(); alpha_loss.backward(); self.alpha_opt.step()

        with th.no_grad():
            for net, net_t in ((self.q1, self.q1_t), (self.q2, self.q2_t)):
                for p, p_t in zip(net.parameters(), net_t.parameters()):
                    p_t.mul_(1.0 - self.tau).add_(self.tau * p)

    def save(self, path: str) -> None:
        if not path.endswith(".zip"):
            path += ".zip"  # torch archive; .zip kept for naming parity with SB3 models
        th.save({"actor": self.actor.state_dict(), "q1": self.q1.state_dict(),
                 "q2": self.q2.state_dict(), "log_alpha": self.log_alpha}, path)

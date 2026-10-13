"""Option B: risk-sensitive Bellman target for CVaR QR-DQN (issue #6).

Option A (cvar_qrdqn.py) selects actions by CVaR at decision time but bootstraps
on the mean-greedy target, so the critic still learns risk-neutral return
distributions. Option B closes that gap: the TD target itself picks next actions
by the CVaR of the lowest-alpha quantiles, so risk sensitivity flows through the
Bellman backup and the learned distribution is risk-averse end to end.

Only the next-action selection inside the target changes (mean over quantiles ->
CVaR of the lowest-alpha quantiles). Action selection at rollout/eval reuses the
Option A CVaR policy, so the agent both acts and bootstraps risk-averse. The
quantile Huber loss, target network, replay, and exploration are untouched.

The train() body is copied from sb3_contrib QRDQN.train() (v2.7.1) with a single
changed block, flagged below. Re-sync if that method changes upstream.
"""
from __future__ import annotations

import torch as th
from sb3_contrib import QRDQN
from sb3_contrib.common.utils import quantile_huber_loss

from cvar_qrdqn import CVaRQRDQNPolicy, CVaRQuantileNetwork


class CVaRBellmanQRDQN(QRDQN):
    """QR-DQN whose TD target is CVaR-greedy (risk-sensitive Bellman backup)."""

    def __init__(self, *args, cvar_alpha: float = 0.25, **kwargs):
        self.cvar_alpha = cvar_alpha
        CVaRQuantileNetwork.cvar_alpha = cvar_alpha  # action selection is CVaR too
        super().__init__(*args, **kwargs)

    def _cvar(self, quantiles: th.Tensor) -> th.Tensor:
        """Mean of the lowest-alpha quantiles along the quantile axis (dim=1).
        quantiles: (batch, n_quantiles, n_actions) -> (batch, 1, n_actions)."""
        n = quantiles.shape[1]
        k = max(1, int(round(self.cvar_alpha * n)))
        return th.sort(quantiles, dim=1).values[:, :k, :].mean(dim=1, keepdim=True)

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)

        losses = []
        for _ in range(gradient_steps):
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)
            discounts = replay_data.discounts if replay_data.discounts is not None else self.gamma

            with th.no_grad():
                next_quantiles = self.quantile_net_target(replay_data.next_observations)
                # --- Option B change: CVaR-greedy next action (was: mean over quantiles) ---
                next_greedy_actions = self._cvar(next_quantiles).argmax(dim=2, keepdim=True)
                # --------------------------------------------------------------------------
                next_greedy_actions = next_greedy_actions.expand(batch_size, self.n_quantiles, 1)
                next_quantiles = next_quantiles.gather(dim=2, index=next_greedy_actions).squeeze(dim=2)
                target_quantiles = replay_data.rewards + (1 - replay_data.dones) * discounts * next_quantiles

            current_quantiles = self.quantile_net(replay_data.observations)
            actions = replay_data.actions[..., None].long().expand(batch_size, self.n_quantiles, 1)
            current_quantiles = th.gather(current_quantiles, dim=2, index=actions).squeeze(dim=2)

            loss = quantile_huber_loss(current_quantiles, target_quantiles, sum_over_quantiles=True)
            losses.append(loss.item())

            self.policy.optimizer.zero_grad()
            loss.backward()
            if self.max_grad_norm is not None:
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()

        self._n_updates += gradient_steps
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/loss", sum(losses) / len(losses))


def make_cvar_bellman_qrdqn(env, cvar_alpha: float = 0.25, **kwargs) -> CVaRBellmanQRDQN:
    """Build a CVaR QR-DQN with a risk-sensitive (CVaR-greedy) Bellman target."""
    CVaRQuantileNetwork.cvar_alpha = cvar_alpha
    kwargs.pop("policy", None)
    return CVaRBellmanQRDQN(policy=CVaRQRDQNPolicy, env=env, cvar_alpha=cvar_alpha, **kwargs)

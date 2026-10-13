"""CVaR risk-averse QR-DQN: the headline method.

Standard QR-DQN learns the return distribution but ACTS on the MEAN of the
quantiles (risk-neutral). Here we keep the distributional critic unchanged but
select actions by the **CVaR** (Conditional Value-at-Risk) of the lower tail:
the mean of the worst alpha-fraction of quantiles. A risk-averse controller
prefers actions whose *bad-case* return is least bad, so it structurally avoids
the high-variance oscillation / overload regime rather than only penalizing it
in the reward.

alpha=1.0 recovers vanilla QR-DQN (mean); smaller alpha = more risk-averse.
This overrides action selection only (Option A: distributional critic trained by
standard QR-DQN target, greedy behavior + eval via CVaR). A fully risk-sensitive
Bellman target (CVaR-greedy bootstrap) is a stronger variant left as future work.
"""
from __future__ import annotations

import torch as th
from sb3_contrib import QRDQN
from sb3_contrib.qrdqn.policies import QRDQNPolicy, QuantileNetwork


class CVaRQuantileNetwork(QuantileNetwork):
    cvar_alpha: float = 0.25  # set by make_cvar_qrdqn before construction

    def _predict(self, observation, deterministic: bool = True) -> th.Tensor:
        quantiles = self(observation)                      # (batch, n_quantiles, n_actions)
        n = quantiles.shape[1]
        k = max(1, int(round(self.cvar_alpha * n)))
        # sort ascending along the quantile axis; the lowest k are the bad-case tail
        lower_tail = th.sort(quantiles, dim=1).values[:, :k, :]
        cvar = lower_tail.mean(dim=1)                       # (batch, n_actions) = CVaR_alpha per action
        return cvar.argmax(dim=1).reshape(-1)


class CVaRQRDQNPolicy(QRDQNPolicy):
    def make_quantile_net(self) -> QuantileNetwork:
        net_args = self._update_features_extractor(self.net_args, features_extractor=None)
        return CVaRQuantileNetwork(**net_args).to(self.device)


def make_cvar_qrdqn(env, cvar_alpha: float = 0.25, **kwargs) -> QRDQN:
    """Build a QR-DQN that selects actions by CVaR_alpha of the return distribution."""
    CVaRQuantileNetwork.cvar_alpha = cvar_alpha
    kwargs.pop("policy", None)
    return QRDQN(policy=CVaRQRDQNPolicy, env=env, **kwargs)

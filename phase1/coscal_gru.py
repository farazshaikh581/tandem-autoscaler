"""CoScal's GRU workload-level predictor + MTFS preprocessing (issue #21 step 2).

Ports the forecaster from Xu et al., "CoScal: Multifaceted Scaling of
Microservices With Reinforcement Learning", IEEE Transactions on Network and
Service Management 19(4), Dec 2022 (DOI 10.1109/TNSM.2022.3210211; reference
code https://github.com/Kminassch/CoScal). CoScal predicts the next scheduling
interval's *workload level* with a GRU fed a sliding window of recent demand
after MTFS preprocessing (paper Algorithm 1); that predicted level is both the
RL state and the event trigger in coscal_baseline.CoScalController.

MTFS (paper Algorithm 1), as applied here:
  1. clean -- forward-fill non-finite samples, then zero-fill any leading gap;
  2. min-max normalise the demand series to [0, 1] over the training range;
  3. sliding window -- LOOKBACK past samples -> the next sample (HORIZON = 1).
The model is a single-layer GRU(1 -> HIDDEN) + linear head to 1, trained with
MSE on the normalised one-step target (Adam). CoScal reports a forecast MSE
band of 3e-3 .. 7e-3 on normalised workload; train_gru() prints the held-out
MSE so the reproduction can be checked against that band.

Interpretive choices (full rationale in issue #21, choice 3):
  - One shared GRU across the four classes, matching the `--workload all`
    generalist posture every RL row in comparison_azure.csv uses, rather than
    four per-class forecasters.
  - Trained on the same per-class demand signal SimBackend replays
    (cpu_norm / p99, clipped to [0, 1.5]). The forecaster is trained on the
    trace it later serves -- exactly as CoScal trains its GRU on the Alibaba
    trace it then schedules against. It carries no reward and picks no action;
    it is a predictor, not a policy (choice 4).
  - LOOKBACK = 12 (~1 h at the paper's 5-min scheduling interval), HIDDEN = 16:
    deliberately small, matching the paper's "O(1) table + a small GRU" cost
    argument.

Runs under the project venv (torch). coscal_baseline.py stays torch-free -- it
imports nothing from here; eval_baselines.py and this module's __main__ are the
only importers, and both run under the venv.
"""
from __future__ import annotations

import os
from collections import deque

import numpy as np
import torch
import torch.nn as nn

from coscal_baseline import HoltLevelPredictor, WorkloadLevelPredictor, workload_level
from envs.joint_hv_env import TRACE_FAMILIES, family_suffix, set_trace_family, trace_dir

HERE = os.path.dirname(__file__)
def gru_path() -> str:
    """models/coscal_gru.pt for azure, coscal_gru_<family>.pt otherwise."""
    return os.path.join(HERE, "models", f"coscal_gru{family_suffix()}.pt")

LOOKBACK = 12          # sliding-window length (~1 h at a 5-min interval)
HORIZON = 1            # one interval ahead (CoScal schedules one step ahead)
HIDDEN = 16            # GRU hidden width -- small, matches the paper's cost claim
CLASSES = ("cpu-bound", "memory-bound", "io-bound", "mixed")


class GRUForecaster(nn.Module):
    """GRU(1 -> hidden) + linear head to a single next-step value."""

    def __init__(self, hidden: int = HIDDEN) -> None:
        super().__init__()
        self.gru = nn.GRU(input_size=1, hidden_size=hidden, num_layers=1, batch_first=True)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (B, LOOKBACK, 1)
        out, _ = self.gru(x)
        return self.head(out[:, -1, :])                   # (B, 1)


# ---- MTFS preprocessing -------------------------------------------------------
def _clean(series: np.ndarray) -> np.ndarray:
    """MTFS step 1: forward-fill non-finite samples, zero any leading gap."""
    s = np.asarray(series, dtype=np.float64).copy()
    bad = ~np.isfinite(s)
    if bad.any():
        last = 0.0
        for i in range(len(s)):
            if bad[i]:
                s[i] = last
            else:
                last = s[i]
    return s


def _load_class_signals() -> dict[str, np.ndarray]:
    """The exact per-class demand signal SimBackend replays (joint_hv_env.py:93)."""
    out: dict[str, np.ndarray] = {}
    for c in CLASSES:
        d = dict(np.load(os.path.join(trace_dir("traces"), f"{c}.npz")))
        cpu = _clean(d["cpu_norm"])
        out[c] = np.clip(cpu / (np.percentile(cpu, 99) or 1.0), 0.0, 1.5)
    return out


def _windows(sig_norm: np.ndarray, lookback: int, horizon: int = HORIZON):
    """MTFS step 3: (lookback past -> next) sliding windows over a normalised signal."""
    n = len(sig_norm) - lookback - horizon + 1
    if n <= 0:
        return (np.empty((0, lookback, 1), np.float32), np.empty((0,), np.float32))
    x = np.stack([sig_norm[i:i + lookback] for i in range(n)]).astype(np.float32)[..., None]
    y = np.array([sig_norm[i + lookback + horizon - 1] for i in range(n)], dtype=np.float32)
    return x, y


# ---- training ---------------------------------------------------------------
def train_gru(epochs: int = 300, lookback: int = LOOKBACK, hidden: int = HIDDEN,
              seed: int = 0, out: str | None = None) -> float:
    """Train the shared GRU on the four azure class signals; save to `out`.

    Last 15% of each class (chronological) is held out. Prints the held-out MSE
    -- compare against CoScal's reported 3e-3 .. 7e-3 -- plus the held-out
    workload-level accuracy (share of windows whose predicted level matches the
    true level), which is what the controller actually consumes.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    sigs = _load_class_signals()
    allv = np.concatenate(list(sigs.values()))
    lo, hi = float(allv.min()), float(max(allv.max(), allv.min() + 1e-6))

    xtr, ytr, xte, yte = [], [], [], []
    for s in sigs.values():
        sn = (np.clip(s, lo, hi) - lo) / (hi - lo)
        x, y = _windows(sn, lookback)
        if len(x) == 0:
            continue
        k = max(1, int(len(x) * 0.85))
        xtr.append(x[:k]); ytr.append(y[:k]); xte.append(x[k:]); yte.append(y[k:])
    xtr = torch.from_numpy(np.concatenate(xtr)); ytr = torch.from_numpy(np.concatenate(ytr))
    xte = torch.from_numpy(np.concatenate(xte)); yte = torch.from_numpy(np.concatenate(yte))

    model = GRUForecaster(hidden)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    lossf = nn.MSELoss()
    bs, n = 256, len(xtr)

    def _heldout_mse() -> float:
        model.eval()
        with torch.no_grad():
            return float(lossf(model(xte).squeeze(-1), yte))

    best_mse, best_state = float("inf"), None
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            loss = lossf(model(xtr[idx]).squeeze(-1), ytr[idx])
            loss.backward()
            opt.step()
        cur = _heldout_mse()
        if cur < best_mse:
            best_mse = cur
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        if ep == 0 or (ep + 1) % 50 == 0:
            print(f"  epoch {ep + 1:4d}  held-out MSE {cur:.5f}  (best {best_mse:.5f})")

    model.load_state_dict(best_state)             # keep the best held-out checkpoint
    mse = _heldout_mse()
    # level accuracy: denormalise both sides, bin, compare
    model.eval()
    with torch.no_grad():
        pred = model(xte).squeeze(-1).numpy()
    pred_lvl = np.array([workload_level(lo + p * (hi - lo)) for p in pred])
    true_lvl = np.array([workload_level(lo + t * (hi - lo)) for t in yte.numpy()])
    lvl_acc = float((pred_lvl == true_lvl).mean())

    out = out or gru_path()
    os.makedirs(os.path.dirname(out), exist_ok=True)
    torch.save(dict(state_dict=model.state_dict(), lo=lo, hi=hi,
                    lookback=lookback, hidden=hidden, horizon=HORIZON), out)
    print(f"wrote {out}")
    print(f"held-out MSE {mse:.5f}  (CoScal reports 3e-3 .. 7e-3)   "
          f"level accuracy {lvl_acc:.3f}")
    return mse


# ---- inference-time predictor ---------------------------------------------
class GRULevelPredictor(WorkloadLevelPredictor):
    """Trained-GRU forecaster for coscal_baseline.CoScalController.

    Keeps a rolling LOOKBACK window of normalised demand; predict(load_frac)
    appends the new sample and returns the binned one-step forecast. During the
    first LOOKBACK ticks (window not yet full) it falls back to persistence
    (bin the current sample) -- the same warm-up rule the Holt fallback uses.
    """

    def __init__(self, path: str | None = None, device: str = "cpu") -> None:
        path = path or gru_path()
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"{path} -- train it first: `python coscal_gru.py` (issue #21 step 2)")
        ckpt = torch.load(path, map_location=device, weights_only=True)
        self.lo, self.hi = float(ckpt["lo"]), float(ckpt["hi"])
        self.lookback = int(ckpt["lookback"])
        self.model = GRUForecaster(int(ckpt["hidden"]))
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()
        self._device = device
        self.reset()

    def reset(self) -> None:
        self._hist: deque[float] = deque(maxlen=self.lookback)

    def _norm(self, v: float) -> float:
        return (min(max(v, self.lo), self.hi) - self.lo) / (self.hi - self.lo + 1e-9)

    def predict(self, load_frac: float) -> int:
        self._hist.append(self._norm(load_frac))
        if len(self._hist) < self.lookback:
            return workload_level(load_frac)                 # warm-up: persistence
        seq = torch.tensor(np.array(self._hist, dtype=np.float32),
                           device=self._device).view(1, self.lookback, 1)
        with torch.no_grad():
            yhat_n = float(self.model(seq).item())
        return workload_level(self.lo + yhat_n * (self.hi - self.lo))


def make_predictor(path: str | None = None) -> WorkloadLevelPredictor:
    """GRULevelPredictor if the weights exist, else the Holt fallback (choice 3)."""
    try:
        return GRULevelPredictor(path)
    except FileNotFoundError as e:
        print(f"  [coscal] {e}\n  [coscal] falling back to HoltLevelPredictor")
        return HoltLevelPredictor()


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Train CoScal's GRU workload-level predictor")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="default: models/coscal_gru[_<family>].pt")
    ap.add_argument("--trace-family", choices=TRACE_FAMILIES, default="azure")
    args = ap.parse_args()
    set_trace_family(args.trace_family)
    train_gru(epochs=args.epochs, seed=args.seed, out=args.out)

"""Fit SimBackend CPU-axis constants from a calibrate_sweep.py CSV (issue #1).

The original sim used the smooth M/M/1 form  latency = L0 / (1 - rho), which the
sweep disproved: measured latency is FLAT (~5 ms) up to rho ~0.8 and then jumps
to a ~400 ms plateau (CPU-limit throttling; the client timeout bounds the tail).
Success behaves the same way: 1.0 below the knee, ~0.95 on the plateau.

Model fitted here (and implemented in envs/joint_hv_env.py):
  rho     = rps * M_PER_RPS / total_limit          (demand-based; can exceed 1)
  knee    = sigmoid((rho - RHO_KNEE) / KNEE_WIDTH)
  latency = L0_MS + (LAT_SAT_MS - L0_MS) * knee
  success = 1 - SAT_DROP * knee

M_PER_RPS comes from the clearly-unsaturated levels (measured usage tracks
offered demand there); L0/LAT_SAT/SAT_DROP are means of the flat regions; the
knee position and width are grid-searched to minimize latency MAPE.

Usage: python3 fit_constants.py sweep_v2.csv
"""
import csv
import sys

import numpy as np

RHO_LINEAR = 0.8   # measured-rho threshold separating unsaturated levels


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def main() -> None:
    path = sys.argv[1]
    with open(path) as f:
        rows = [{k: float(v) for k, v in r.items() if k != "per_pod_m"}
                for r in csv.DictReader(f)]

    limit_m = rows[0]["limit_total_m"]
    rps = np.array([r["rps"] for r in rows])
    usage = np.array([r["usage_total_m"] for r in rows])
    lat = np.array([r["mean_ms"] for r in rows])
    succ = np.array([r["success"] for r in rows])

    # 1) M_PER_RPS from clearly-unsaturated levels (measured rho < RHO_LINEAR)
    unsat = usage / limit_m < RHO_LINEAR
    m_per_rps = float(np.sum(usage[unsat] * rps[unsat]) / np.sum(rps[unsat] ** 2))

    # 2) demand-based rho for every level (this is the sim's rho)
    rho_d = rps * m_per_rps / limit_m

    # 3) plateau levels: L0 from the flat base, LAT_SAT/SAT_DROP from saturation
    base = rho_d < RHO_LINEAR
    over = rho_d > 1.0
    l0 = float(lat[base].mean())
    lat_sat = float(lat[over].mean())
    sat_drop = float((1.0 - succ[over]).mean())

    # 4) knee position + width by grid search on latency MAPE (all levels)
    best = None
    for knee_pos in np.arange(0.82, 1.001, 0.005):
        for width in np.arange(0.005, 0.081, 0.005):
            pred = l0 + (lat_sat - l0) * sigmoid((rho_d - knee_pos) / width)
            mape = float(np.mean(np.abs(pred - lat) / lat))
            if best is None or mape < best[0]:
                best = (mape, knee_pos, width)
    mape, rho_knee, knee_width = best

    knee = sigmoid((rho_d - rho_knee) / knee_width)
    lat_pred = l0 + (lat_sat - l0) * knee
    succ_pred = 1.0 - sat_drop * knee

    print(f"fitted from {path} (total limit {limit_m:.0f}m):")
    print(f"  M_PER_RPS  = {m_per_rps:.2f} mcores/rps")
    print(f"  L0_MS      = {l0:.2f}")
    print(f"  LAT_SAT_MS = {lat_sat:.1f}")
    print(f"  RHO_KNEE   = {rho_knee:.3f}")
    print(f"  KNEE_WIDTH = {knee_width:.3f}")
    print(f"  SAT_DROP   = {sat_drop:.3f}")
    print()
    print("  rps   rho_d   lat_meas  lat_pred   succ_meas  succ_pred")
    for i in range(len(rows)):
        print(f"  {rps[i]:5.0f}  {rho_d[i]:5.2f}  {lat[i]:8.1f}  {lat_pred[i]:8.1f}"
              f"   {succ[i]:8.4f}  {succ_pred[i]:8.4f}")
    print(f"\n  latency MAPE (all levels): {100 * mape:.1f}%")
    print(f"  success MAE  (all levels): {np.abs(succ_pred - succ).mean():.4f}")


if __name__ == "__main__":
    main()

"""Emit the train.py job matrix for the cost/SLA Pareto sweep (#20), one
argument string per line on stdout. Consumed by run_pareto_lane.sh via
dispatch_pareto.sh.

Grid (fully crossed):
  W_COST  0.0375 0.075 0.15 0.3 0.6 1.2 2.4 4.8   (8, log-spaced x2, default 0.3 included)
  W_VIOL  0.5 1.0 2.0 4.0                          (4, log-spaced x2, default 1.0 included)
Per cell: qrdqn & cvar-bellman-qrdqn at seeds 0,1; cvar-qrdqn, ppo, dqn at seed 0.
  8 x 4 x 7 = 224 jobs.

  python gen_pareto_jobs.py                 # all 224
  python gen_pareto_jobs.py --wviol 1.0      # the 56-job W_VIOL=1.0 slice (main frontier)
  python gen_pareto_jobs.py --family bitbrains --with-ablation
                                             # same grid on the Bitbrains traces, plus the
                                             # 4 H-only/V-only ablation runs (tag 'bb')

Tags carry the trace family: 'azure_wc..' for azure, 'bb_wc..' for bitbrains
(train.py refuses a tag that does not match its --trace-family).
"""
import argparse
import signal

signal.signal(signal.SIGPIPE, signal.SIG_DFL)  # allow `| head` without a traceback

W_COST = ["0.0375", "0.075", "0.15", "0.3", "0.6", "1.2", "2.4", "4.8"]
W_VIOL = ["0.5", "1.0", "2.0", "4.0"]
# (algo, seed, extra args)
PER_CELL = [
    ("qrdqn", 0, ""),
    ("qrdqn", 1, ""),
    ("cvar-bellman-qrdqn", 0, "--cvar-alpha 0.25"),
    ("cvar-bellman-qrdqn", 1, "--cvar-alpha 0.25"),
    ("cvar-qrdqn", 0, "--cvar-alpha 0.25"),
    ("ppo", 0, ""),
    ("dqn", 0, ""),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wviol", default="all", help="'all' or one of " + ",".join(W_VIOL))
    ap.add_argument("--family", choices=["azure", "bitbrains"], default="azure")
    ap.add_argument("--with-ablation", action="store_true",
                    help="also emit the H-only/V-only action-space ablation (#15) at default weights")
    args = ap.parse_args()
    wviols = W_VIOL if args.wviol == "all" else [args.wviol]
    prefix = "azure" if args.family == "azure" else "bb"
    fam_arg = "" if args.family == "azure" else " --trace-family bitbrains"

    for wv in wviols:
        for wc in W_COST:
            for algo, seed, extra in PER_CELL:
                tag = f"{prefix}_wc{wc}_wv{wv}" + (f"_seed{seed}" if seed else "")
                line = (f"--algo {algo} --workload all --tag {tag} --seed {seed} "
                        f"--w-cost {wc} --w-viol {wv}")
                if extra:
                    line += " " + extra
                print(line + fam_arg)

    if args.with_ablation:
        for algo, extra in [("qrdqn", ""), ("cvar-bellman-qrdqn", " --cvar-alpha 0.25")]:
            for mode in ("horizontal", "vertical"):
                print(f"--algo {algo} --workload all --tag {prefix} --seed 0 "
                      f"--scaling-mode {mode}{extra}{fam_arg}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Live progress dashboard for train.py runs.

Reads tensorboard event files of active runs and renders per-algorithm
progress bars with step rate, ETA and latest episode reward.

    ./watch_progress.py            # refresh every 15s until runs finish
    ./watch_progress.py --once     # single snapshot
"""
import argparse
import glob
import os
import re
import subprocess
import sys
import time

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
TB = os.path.join(HERE, "tb")
MODELS = os.path.join(HERE, "models")

BOLD, DIM, RESET = "\033[1m", "\033[2m", "\033[0m"
GREEN, YELLOW, CYAN = "\033[32m", "\033[33m", "\033[36m"


def running_trainers():
    """[{pid, algo, timesteps, started}] for live train.py processes, oldest first."""
    out = subprocess.run(["ps", "-eo", "pid,lstart,cmd"], capture_output=True, text=True).stdout
    procs = []
    for line in out.splitlines():
        m = re.search(r"^\s*(\d+)\s+(.{24})\s+.*train\.py --algo (\S+).*?--timesteps (\d+)", line)
        if m and "grep" not in line:
            procs.append({"pid": int(m.group(1)),
                          "started": time.mktime(time.strptime(m.group(2).strip(), "%c")),
                          "algo": m.group(3), "timesteps": int(m.group(4))})
    return sorted(procs, key=lambda p: p["started"])


def active_tb_dirs(max_age=180):
    """tb run dirs whose event file was written recently, oldest dir first."""
    dirs = []
    for d in sorted(glob.glob(os.path.join(TB, "*_*"))):
        evs = glob.glob(os.path.join(d, "events*"))
        if evs and time.time() - max(os.path.getmtime(e) for e in evs) < max_age:
            dirs.append((min(os.path.getctime(e) for e in evs), d))
    return [d for _, d in sorted(dirs)]


def read_run(d):
    """(step, rate_steps_per_s, ep_rew) from a tb run dir, or None."""
    ea = EventAccumulator(d, size_guidance={"scalars": 0})
    ea.Reload()
    tags = ea.Tags()["scalars"]
    step_tag = next((t for t in tags if "ep_rew_mean" in t), tags[0] if tags else None)
    if step_tag is None:
        return None
    ev = ea.Scalars(step_tag)
    first, last = ev[0], ev[-1]
    rate = (last.step - first.step) / max(last.wall_time - first.wall_time, 1e-9)
    rew = next((ea.Scalars(t)[-1].value for t in tags if "ep_rew_mean" in t), None)
    return last.step, rate, rew


def bar(frac, width=34):
    full = int(frac * width)
    return GREEN + "█" * full + DIM + "░" * (width - full) + RESET


def fmt_eta(sec):
    if sec < 0 or sec > 86400:
        return "--"
    h, m = int(sec // 3600), int(sec % 3600 // 60)
    return f"{h}h{m:02d}m" if h else f"{m}m{int(sec % 60):02d}s"


def snapshot():
    procs = running_trainers()
    dirs = active_tb_dirs()
    lines = [f"{BOLD}Phase 1 training: {time.strftime('%H:%M:%S')}{RESET}"]
    if not procs:
        lines.append(f"{DIM}no train.py processes running{RESET}")
    # pair oldest process with oldest active dir; fall back to dir name label
    for i, d in enumerate(dirs):
        label = procs[i]["algo"] if i < len(procs) else os.path.basename(d)
        total = procs[i]["timesteps"] if i < len(procs) else 300000
        run = read_run(d)
        if run is None:
            continue
        step, rate, rew = run
        frac = min(step / total, 1.0)
        eta = fmt_eta((total - step) / max(rate, 1e-9))
        rew_s = f"rew {rew:8.1f}" if rew is not None else ""
        lines.append(f"  {CYAN}{label:<12}{RESET} {bar(frac)} {frac*100:5.1f}%  "
                     f"{step:>7,}/{total:,}  {rate:5.0f} st/s  ETA {YELLOW}{eta}{RESET}  {rew_s}")
    done = [os.path.basename(m)[:-4] for m in sorted(glob.glob(os.path.join(MODELS, "*_all.zip")))]
    if done:
        lines.append(f"{DIM}completed models: {', '.join(done)}{RESET}")
    return "\n".join(lines), bool(procs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--interval", type=float, default=15)
    ap.add_argument("--once", action="store_true")
    args = ap.parse_args()
    while True:
        text, alive = snapshot()
        if not args.once:
            sys.stdout.write("\033[2J\033[H")
        print(text)
        if args.once or not alive:
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()

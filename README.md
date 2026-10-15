# TANDEM: joint horizontal and vertical autoscaling with in-place pod resize

Code, evaluation results and the scripts that rebuild the workload traces, for the paper

> F. Shaikh, G. Reali, M. Femminella. *TANDEM: Risk-Sensitive Joint Horizontal and
> Vertical Autoscaling of Containerized Network Services with In-Place Pod Resizing.*

TANDEM is a reinforcement learning controller that sets the replica count and the CPU
and memory request of each pod of a Kubernetes Deployment in one decision, applying
vertical changes with in-place pod resize. It extends QR-DQN with a CVaR-greedy
training target. This repository contains the real-cluster measurements of Section 3,
the calibrated simulator and the controllers of Sections 4 and 5, all evaluation
results of Section 6, and the scripts that draw the paper's figures.

## Setup

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

Simulation needs only Python. The real-cluster experiments need a Kubernetes 1.35 or
later cluster with `metrics-server` (we used k3s) and `kubectl` access.

**Workload traces.** The traces are not redistributed here; they are rebuilt, array for
array, from the two public datasets. Download and unpack

- the Bitbrains GWA-T-12 fastStorage trace
  (`https://atlarge-research.com/gwa-traces/gwa_t_12_fastStorage.zip`) into
  `phase1/data/`, so that the machines are in `phase1/data/fastStorage/2013-8/`;
- the Azure Functions Trace 2019
  (`https://github.com/Azure/AzurePublicDataset`, release `dataset-functions-2019`) into
  `phase1/data/azure_functions_2019/`;

then run

```bash
cd phase1 && python3 build_traces.py && python3 ../phase2/derive_profiles.py --rebuild
```

Trained models are not stored in git. Download `models.tar.gz` from the latest release
and unpack it into `phase1/` (it creates `phase1/models/`). It contains every model whose
results appear in the paper, except the 224 models of the reward-weight sweep, whose
results are in `phase1/eval/`.

## Reproducing the paper

Commands run from `phase1/` unless they show another path; the figure commands run from the repository root. Each training run takes
3 x 10^5 steps; the QR-DQN family takes about three hours on one CPU core.

**Figures.** The figures are drawn from the committed results and need no training
(the episode figure also needs the traces and the models):

```bash
python3 paper/figures/plot_knee.py        # calibration figure (Section 4.2)
python3 paper/figures/plot_overview.py    # both full-page result figures (Section 6)
python3 paper/figures/plot_episode.py     # mixed-class episode (needs the models)
```

**Training and evaluation in simulation (Section 6.1).** Azure is the default trace
family; Bitbrains runs add `--trace-family bitbrains` and a `bb` tag:

```bash
python3 train.py --algo cvar-bellman-qrdqn --workload all --tag azure --seed 0     # TANDEM
python3 train.py --algo qrdqn --workload all --tag azure_seed1 --seed 1            # QR-DQN, seed 1
python3 train.py --algo qrdqn --workload all --tag bb_seed3 --seed 3 --trace-family bitbrains
python3 eval_baselines.py --tag azure                                              # threshold, MPC, CoScal, ...
python3 aggregate_eval.py --suffix azure && python3 stat_significance.py --tag azure
```

The other learners use `--algo cvar-qrdqn | dqn | ppo | trpo | sac-discrete | pandey-ppo | pandey-ddpg`.

**Action-space ablation (6.2):** add `--scaling-mode horizontal` or `--scaling-mode vertical`.

**Reward-weight sweep and fixed configurations (6.4):** `gen_pareto_jobs.py` lists the
224 training jobs, `run_pareto_lane.sh` runs a list of them, `run_pareto_baselines.sh`
sweeps MPC and the threshold controllers, `static_frontier.py` evaluates every fixed
configuration, and `pareto_sweep.py` aggregates the results.

**Held-out machines (6.5):** `eval_heldout.py`; models trained on the pool of 34
machines use `train.py --domain-rand`.

**Real cluster (6.6):** deploy `phase0/namespace.yaml`, `phase0/web-service.yaml` and
`phase0/web-service-io.yaml`, then run `eval_real.py` (learned controllers),
`eval_real_keda.py` (KEDA, see `keda/README.md`) and `compare_sim_real.py`; the scripts
`run_real_bb.sh`, `run_keda_real.sh` and `run_coupling_bb.sh` list the windows and
settings used.

**Co-located functions (6.7):** `python3 ../phase2/eval_multifn.py`.

**Calibration (4.2):** `calibration/calibrate_sweep.py` measures a sweep on the cluster,
`calibration/fit_constants.py` fits the model, and `calibration/knee_sensitivity.py`
re-evaluates the trained controllers under the refitted model. `calibration/CALIBRATION.md`
records the measurements.

**Section 3:** see `phase0/README.md`.

## Data

The workload traces are derived from two public datasets and are rebuilt by
`phase1/build_traces.py` rather than redistributed. The memory and IO series come from
the Bitbrains GWA-T-12 fastStorage trace; the load of the Azure family comes from the
Azure Functions 2019 trace, which is licensed under CC BY 4.0. `phase1/CLASSIFICATION.md` describes how machines were assigned to workload
classes, and `phase1/azure_trace_assignment.csv` which Azure function drives which trace.
If you use the traces, please also cite:

- S. Shen, V. van Beek, A. Iosup. Statistical characterization of business-critical
  workloads hosted in cloud datacenters. CCGrid 2015.
- M. Shahrad et al. Serverless in the wild: characterizing and optimizing the serverless
  workload at a large cloud provider. USENIX ATC 2020.

## License

The code is released under the MIT License (`LICENSE`). The datasets the traces are built
from remain subject to their own terms.

## Citation

See `CITATION.cff`.

# Scheduler-agnostic cluster execution

This repository has no checked-in Slurm, PBS, LSF, Kubernetes, or institutional scheduler convention. The experiment tooling therefore deliberately does not assume one.

Run from `Final Project/bomberman_rl/`. Long runs require a clean committed checkout so `provenance.json` contains the exact git commit; a local smoke run may add `--allow-dirty` and is marked as such. `prepare` freezes the external config, resolved runtime hyperparameters, commit and job list. A worker verifies that its own checkout is still exactly that clean commit before it starts.

```bash
conda run -n ml_homework python scripts/run_experiment.py prepare \
  --config experiments/r01_a00_smoke.json --run-id r01_cluster_001
```

The command writes the full normalized config snapshot, git provenance, `jobs.json`, and one scheduler-ready JSON parameter file per job under `runs/r01_cluster_001/job_parameters/`. Job paths are relative to `runs/r01_cluster_001/`, so the entire run directory can move from a Mac to Hetzner or a cluster filesystem. On every worker, clone this repository and check out the recorded commit before submitting one training parameter file per worker:

```bash
conda run -n ml_homework python scripts/run_experiment.py job \
  --job-file runs/r01_cluster_001/job_parameters/train_seed101.json
```

After every training job has completed and its `agent/latest_model.npz` has been synchronized into the same run directory, submit the dependent greedy evaluation jobs:

```bash
conda run -n ml_homework python scripts/run_experiment.py job \
  --job-file runs/r01_cluster_001/job_parameters/eval_train101_seed201.json
```

Synchronize the entire `runs/<run_id>/` directory, preserving paths and modification times; do not flatten job directories or copy a model to `agent_code/research_agent/artifacts/`. On the aggregation host, run the only command allowed to update the promoted model:

```bash
conda run -n ml_homework python scripts/aggregate_results.py \
  --run-dir runs/r01_cluster_001 --promote
```

Promotion is deterministic: maximize mean official score, then minimize score standard deviation, then minimize mean suicides, then maximize mean coins. The chosen checkpoint uses the smaller training seed as its final tie-break; competing configuration runs use lexical `run_id`. The aggregator alone writes `runs/promoted/<primary-scenario>/active_model.npz` and `best_summary.json`; training jobs never write an active model. This keeps incomparable scenarios such as `coin-heaven` and `classic` out of one leaderboard.

If a job finished with a non-zero exit code, retry it explicitly. The failed attempt is preserved below `runs/<run_id>/failed_attempts/<job_id>/attemptNN/`; a successful job is never overwritten:

```bash
conda run -n ml_homework python scripts/run_experiment.py job \
  --job-file runs/r01_cluster_001/job_parameters/train_seed101.json --retry
```

For an exported final agent, copy the selected checkpoint to `agent_code/<final_agent>/model.npz`. The normal official framework then loads that packaged file without requiring `BOMBERMAN_MODEL_PATH`; experiment evaluation always supplies an explicit checkpoint path instead.

For a local sequential run (training, evaluation, aggregation), use:

```bash
conda run -n ml_homework python scripts/run_experiment.py run \
  --config experiments/r01_a00_smoke.json --allow-dirty
```

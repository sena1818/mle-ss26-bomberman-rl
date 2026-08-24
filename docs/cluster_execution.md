# Scheduler-agnostic cluster execution

This repository has no checked-in Slurm, PBS, LSF, Kubernetes, or institutional scheduler convention. The experiment tooling therefore deliberately does not assume one.

Run from `Final Project/bomberman_rl/`. Long runs require a clean committed checkout so `provenance.json` contains the exact git commit; a local smoke run may add `--allow-dirty` and is marked as such.

```bash
conda run -n ml_homework python scripts/run_experiment.py prepare \
  --config experiments/r01_a00_smoke.json --run-id r01_cluster_001
```

The command writes the full normalized config snapshot, git provenance, `jobs.json`, and one scheduler-ready JSON parameter file per job under `runs/r01_cluster_001/job_parameters/`. Submit one training parameter file per worker with the scheduler's ordinary single-command mechanism:

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

Promotion is deterministic: maximize mean official score, then minimize score standard deviation, then minimize mean suicides, then maximize mean coins. The chosen checkpoint uses the smaller training seed as its final tie-break; competing configuration runs use lexical `run_id`. The aggregator alone writes `runs/promoted/active_model.npz` and `best_summary.json`; training jobs never write an active model.

For a local sequential run (training, evaluation, aggregation), use:

```bash
conda run -n ml_homework python scripts/run_experiment.py run \
  --config experiments/r01_a00_smoke.json --allow-dirty
```

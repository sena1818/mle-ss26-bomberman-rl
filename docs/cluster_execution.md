# Scheduler-agnostic cluster execution

This repository has no checked-in Slurm, PBS, LSF, Kubernetes, or institutional scheduler convention. The experiment tooling therefore deliberately does not assume one.

Run from `Final Project/bomberman_rl/`. Long runs require a clean committed checkout so `provenance.json` contains the exact git commit; a local smoke run may add `--allow-dirty` and is marked as such. `prepare` freezes the external config, resolved runtime hyperparameters, commit and job list. A worker verifies that its own checkout is still exactly that clean commit before it starts.

The runner's private job copy is deliberately **not** a Python environment. On a fresh Hetzner checkout, create one shared environment once at the repository root before preparing jobs. The R01 E01 run was verified with Python 3.12.3, NumPy 2.5.2 and tqdm 4.70.0:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install "numpy==2.5.2" "tqdm==4.70.0"
```

On Hetzner, replace the `conda run -n ml_homework python` prefix in the examples below with `.venv/bin/python` (or use another already verified environment). Do not put `.venv` below a job artifact: the runtime allowlist excludes it by design.

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

## 单机并行：`run --jobs N`

`run --jobs N` 在**一台**机器上并发执行同一阶段的 job（训练阶段全部完成后才开始评估阶段）。
Hetzner 是单台服务器而非调度集群，因此这是那里的默认执行方式；上面按 job 逐个提交的流程
保留给真正的调度器。

```bash
.venv/bin/python scripts/run_experiment.py run \
  --config experiments/r01_a03_dose_response_classic.json \
  --run-id r01_a03_dose_response_<tag> --jobs 8
```

阶段内的 job 相互独立（各自拥有私有 artifact 目录与私有框架副本，只通过文件通信），
阶段之间保序。并发是**线程**级的：每个 job 的实际工作都在一个 `subprocess.run` 里，
它会释放 GIL；进程内的部分只有几毫秒的文件复制。

选 `N`：不要超过物理核数。一次 365-job 的 arm 在 8 并发下约 10–15 分钟，
`A05` 这类平均 5 步就结束的 arm 约 1 分钟。

**磁盘**：每个 job 会复制一份私有框架副本，成功后删除，但峰值占用约等于 `N × 单副本`。
一个跑完的 arm 约 2.5 GiB（`A05` 约 150 MiB）。用 `scripts/prune_runs.py` 瘦身。

### 可复现性（2026-08-26 核实）

同一 commit 下，服务器（Linux / Python 3.12.3 / NumPy 2.5.2）与 macOS ARM / NumPy 2.4.6
的结果**逐位一致**：把 `A05` 臂在 commit `85034d9` 上本地重跑，与 8-25 的服务器版每个指标
都相同（`steps 15.57 ± 20.5`、`bomb_rate 0.1877 ± 0.0131`、`suicides 0.9756 ± 0.0527`）。
串行与 `--jobs 8` 的结果同样一致。**因此跨平台、跨并发度的定量比较是合法的**，
run 目录可以在 Mac 与 Hetzner 之间自由搬运。

## Evaluating checkpoints instead of only the final model

By default an experiment evaluates only each training seed's final
`latest_model.npz`. That is the historical behaviour and it stays the default so
older runs remain comparable. A config may opt into a checkpoint sweep:

```json
"checkpoint_evaluation": {
  "mode": "all",
  "validation_seeds": [2001, 2002, 2003],
  "holdout_seeds": [3001, 3002, 3003]
}
```

- `mode: "all"` creates one evaluation job per saved checkpoint round, turning a
  single final number into a learning curve. Checkpoints are addressed by round,
  not by file name, because the file name also encodes an update count that does
  not exist until the training job has run; the worker resolves the file and
  fails loudly if a round matches zero or more than one checkpoint.
- `validation_seeds` choose the checkpoint. `holdout_seeds` never do — they are
  evaluated separately and are what a final claim should quote. The two sets may
  not overlap; `prepare` rejects a config where they do.
- Each named suite in `evaluation_suites` may carry its own
  `checkpoint_evaluation` block. Without one it stays on `latest`, so a transfer
  diagnostic does not silently inflate into a full sweep.

## Comparing several arms of one study

Runs that differ in exactly one declared dimension can be tabulated together:

```bash
conda run -n ml_homework python scripts/compare_runs.py \
  --run-dir runs/<a02_run> --run-dir runs/<a03_run> --run-dir runs/<a05_run> \
  --split holdout --checkpoint-round 500 --out runs/dose_response_table.json
```

The script reads only `evaluation_summary.json` and `experiment_config.snapshot.json`,
never writes into a run, and refuses to line up runs that differ in more than one
dimension unless `--allow-multiple-differences` is passed deliberately.

For a behavioural dose-response sweep, aggregate **without** `--promote` and
compare a fixed checkpoint round as above.  Promotion deliberately optimizes
official score; it is not a valid selector for an experiment whose primary
question is whether bombing behaviour changes with the death penalty.  The
summary still retains validation-selected checkpoint results for a later final
performance claim, while its `*_checkpoint_curve` records preserve the full
learning curve.

## Disk footprint and retention

A job's private framework copy exists only so the official framework's fixed log
paths cannot collide between concurrent jobs. It is not a result. Two rules keep
it from dominating the disk:

1. `copy_runtime` is an **allowlist**: every top-level `*.py`, plus `assets/` and
   `agent_code/`. It used to be a deny-list, which meant each job copied whatever
   new thing appeared in the repository root — including a `.venv`, at 154 MiB
   per job and 95 GiB across one three-arm study. The trained model those jobs
   produce is 3.5 KiB.
2. A job **deletes its own runtime on success**. Both framework logs are copied
   into the job directory first, aggregation reads only `official_stats.json`
   and the agent JSONL, and `provenance.json` plus `command.json` record which
   commit and command produced the result. A failed job keeps its runtime;
   `--keep-runtime` forces it for debugging.

For run directories created before those rules, reclaim space explicitly. The
script never touches `evaluation_summary.json`, `official_stats.json` or any
`.npz` model, and only prunes jobs whose `completion.json` records exit code 0:

```bash
conda run -n ml_homework python scripts/prune_runs.py --run-dir runs/<run_id> --drop-runtime
```

It is a dry run until `--apply` is passed. `--compress-logs` additionally gzips
the framework logs and the agent JSONL; the aggregator reads `agent.jsonl.gz`
transparently, so a compressed run can still be re-summarized.

To keep a result without keeping the run, write a minimal archive. It holds the
config snapshot, provenance, job list, evaluation summary, every trained model
including periodic checkpoints, and each job's official statistics:

```bash
conda run -n ml_homework python scripts/prune_runs.py --run-dir runs/<run_id> --slim-copy archive/ --apply
```

Measured on existing runs, a slim copy is 600x to 7500x smaller than its source.
This is the form to keep on a laptop or commit to long-term storage.

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

This command writes an `evaluation_summary.json` but **does not promote** a
model. That is intentional: a tiny smoke can produce a noisy one-seed score
that must never replace a real experiment's scenario pointer. Promotion always
requires a deliberate extra `--promote`, and normal long-run practice remains
to run `aggregate_results.py --promote` only after all declared evaluation jobs
have completed.

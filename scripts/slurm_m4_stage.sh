#!/usr/bin/env bash
#SBATCH --job-name=m4
#SBATCH --nodes=1                 # see NOTE 1: this workload does not span nodes
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16        # -> --jobs 16; do not exceed the node's cores
#SBATCH --time=24:00:00           # CONFIRM against the partition's limit
#SBATCH --partition=cpu           # bwUniCluster 3.0; CONFIRM: sinfo -o "%P %l %c %m"
                                  # 3.0 replaced 2.0's single/multiple with cpu/dev_cpu
#SBATCH --mem=32G                 # ~0.5 GB per concurrent training job, plus slack
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#
# One SLURM job = one stage of the M4 line.  Chain them with dependencies so a
# stage that fails its gate stops the rest:
#
#   p=$(sbatch --parsable --job-name=m4pilot  scripts/slurm_m4_stage.sh pilot  20260901)
#   a=$(sbatch --parsable --dependency=afterok:$p --job-name=m4anchor scripts/slurm_m4_stage.sh anchor 20260901)
#   l=$(sbatch --parsable --dependency=afterok:$a --job-name=m4lr1    scripts/slurm_m4_stage.sh lr1e4  20260901)
#   sbatch        --dependency=afterok:$a --job-name=m4lr5 scripts/slurm_m4_stage.sh lr5e4 20260901
#
# lr1e4 and lr5e4 both depend only on the anchor and are independent of each
# other, so they run concurrently if the queue allows.  STOP after them and
# make the step-size decision before submitting stage 3 (see docs/06 section 8).
#
# NOTE 1  ``run --jobs N`` is process-level parallelism inside ONE node: it
#         spawns N subprocesses on the machine it runs on.  It does not
#         distribute over nodes.  Asking for --nodes=2 leaves the second idle.
#         To use more of the cluster, submit the independent STAGES in
#         parallel, not one stage across nodes.
#
# NOTE 2  Thread oversubscription is the classic way to make this slower than a
#         laptop.  With N concurrent jobs each defaulting to "use every core",
#         the node thrashes.  All three limits below are required, not tidy.
#
# NOTE 3  This workload is small-file heavy and bwUniCluster's shared filesystem
#         is Lustre, which is not.  Each training job appends to its own JSONL
#         once per round -- 10,000 rounds x 5 seeds -- and an arm ends up with
#         a few thousand files.  If training is much slower than the benchmark
#         predicted, this is the first thing to suspect, and the fix is to run
#         on the node-local scratch and copy the run directory back at the end.
#         Check what you have with: df -h "$TMPDIR"

set -u -o pipefail

STAGE="${1:?usage: sbatch scripts/slurm_m4_stage.sh <stage> <date-tag>}"
DATE="${2:?usage: sbatch scripts/slurm_m4_stage.sh <stage> <date-tag>}"

# --- site-specific: CONFIRM ALL THREE before the first submission ------------
module purge
module load devel/python/3.11 2>/dev/null || module load python 2>/dev/null || \
    { echo "adjust the module name: module avail python" >&2; exit 1; }
REPO="${REPO:-$HOME/bomberman_rl}"          # a clean CLONE at a committed revision
VENV="${VENV:-$REPO/.venv}"                 # created once on a LOGIN node (compute nodes have no network)
export LOG_DIR="${LOG_DIR:-$HOME/m4_logs}"  # NOT inside $REPO: a dirty tree makes prepare refuse
# ----------------------------------------------------------------------------

# NOTE 2, all three: torch, numpy's BLAS, and OpenMP each default to grabbing
# every core.  One thread each, with the parallelism coming from job count.
export BOMBERMAN_TORCH_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

cd "$REPO" || exit 1
JOBS="${SLURM_CPUS_PER_TASK:-8}"

echo "stage=$STAGE date=$DATE jobs=$JOBS node=$(hostname) repo=$REPO"
git -C "$REPO" rev-parse --short HEAD
git -C "$REPO" status --porcelain | head -5

exec "$VENV/bin/python" - "$STAGE" "$DATE" "$JOBS" <<'PY'
import subprocess, sys
stage, date, jobs = sys.argv[1], sys.argv[2], sys.argv[3]
# The stage table lives in run_m4_line.sh; this dispatches one of them so that
# each SLURM job is one stage and the scheduler owns the ordering.
config = {
    "pilot":      "m4_r07_a06_e09_t02_pilot",
    "anchor":     "m4_r07_a06_e09_t02_anchor",
    "lr1e4":      "m4_r07_a06_e09_t02_lr1e4",
    "lr5e4":      "m4_r07_a06_e09_t02_lr5e4",
    "opponents":  "m4_r07_a06_e09_t02opp_opponents",
    "no_shaping": "m4_r07_a03_e09_t02_no_shaping",
    "bc":         "m4_r07_a06_e10_t02_bc",
    "dueling":    "m4_r08_a06_e09_t02_dueling",
}[stage]
run_id = f"m4_{stage}_{date}"
gate = ["--training-only"] if stage not in {"pilot", "anchor"} else []
steps = [
    ["scripts/run_experiment.py", "run", "--config", f"experiments/{config}.json",
     "--run-id", run_id, "--jobs", jobs],
    ["scripts/aggregate_results.py", "--run-dir", f"runs/{run_id}"],
    ["scripts/check_pilot.py", "--run-dir", f"runs/{run_id}", *gate],
    ["scripts/prune_runs.py", "--run-dir", f"runs/{run_id}",
     "--drop-runtime", "--compress-logs", "--apply"],
]
for step in steps:
    print("+", " ".join(step), flush=True)
    result = subprocess.run([sys.executable, *step])
    if result.returncode and step[0] != "scripts/prune_runs.py":
        sys.exit(result.returncode)   # a failed gate must fail the SLURM job,
                                      # so --dependency=afterok stops the chain
PY

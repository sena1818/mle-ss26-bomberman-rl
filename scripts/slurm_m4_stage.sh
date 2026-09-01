#!/usr/bin/env bash
#SBATCH --job-name=m4
#SBATCH --nodes=1                 # see NOTE 1: this workload does not span nodes
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16        # -> --jobs 16; do not exceed the node's cores
#SBATCH --time=24:00:00           # cpu allows up to 3-00:00:00
#SBATCH --partition=cpu           # bwUniCluster 3.0, confirmed: cpu = 3 day limit,
                                  # 192 cores/node; dev_cpu = 30 min, good for a
                                  # launch-path test before queueing a real stage
#SBATCH --mem=32G                 # ~0.5 GB per concurrent training job, plus slack
# NO --output/--error defaults on purpose.  SLURM writes them relative to the
# SUBMIT directory, which is this repository, and it creates them the moment
# the job starts -- before anything runs.  git status --porcelain counts
# untracked files, so prepare then sees a dirty checkout and refuses.  A
# default here would be a trap, so the submission must pass both:
#
#   --output="$LOG_DIR/%x_%j.out" --error="$LOG_DIR/%x_%j.err"
#
# The preflight below fails with the reason if anything dirties the tree.
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
# Must be the SAME interpreter the venv was built with on the login node, or
# the venv's symlinked python and the loaded module's shared libraries diverge.
module load devel/python/3.12.3-gnu-14.2 || {
    echo "module devel/python/3.12.3-gnu-14.2 unavailable; module avail python" >&2; exit 1; }
# Default to the directory sbatch was invoked from, which is the repository if
# you submit from inside it.  $HOME/bomberman_rl was the old default and is
# wrong for the documented setup, where the clone lives in a workspace: the job
# failed at `cd "$REPO"` with nothing having run.  Override with REPO=... if you
# submit from somewhere else.
REPO="${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
VENV="${VENV:-$REPO/.venv}"                 # created once on a LOGIN node (compute nodes have no network)
export LOG_DIR="${LOG_DIR:-$HOME/m4_logs}"  # NOT inside $REPO: a dirty tree makes prepare refuse
# ----------------------------------------------------------------------------

# NOTE 2, all three: torch, numpy's BLAS, and OpenMP each default to grabbing
# every core.  One thread each, with the parallelism coming from job count.
export BOMBERMAN_TORCH_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

# Fail here, with the reason, rather than three lines later on a missing file.
for required in "scripts/run_experiment.py" "experiments" ".git"; do
    [ -e "$REPO/$required" ] || {
        echo "REPO=$REPO does not look like this repository (missing $required)." >&2
        echo "  Submit from inside the clone, or pass it explicitly:" >&2
        echo "    sbatch --export=ALL,REPO=\"\$PWD\",VENV=\"\$PWD/.venv\",LOG_DIR=\"\$HOME/m4_logs\" \\" >&2
        echo "           scripts/slurm_m4_stage.sh <stage> <date>" >&2
        exit 1
    }
done
[ -x "$VENV/bin/python" ] || {
    echo "No interpreter at $VENV/bin/python. Build it on a LOGIN node:" >&2
    echo "    module load devel/python/3.12.3-gnu-14.2" >&2
    echo "    python -m venv .venv && .venv/bin/python -m pip install numpy tqdm torch" >&2
    exit 1
}

cd "$REPO" || exit 1

# The highest-probability failure, named rather than left to a generic refusal
# three layers down: a long run requires a clean committed checkout, and the
# easiest way to break that is to let SLURM write this job's own log into it.
DIRT=$(git status --porcelain)
if [ -n "$DIRT" ]; then
    echo "REFUSING: $REPO is not a clean checkout, and prepare requires one." >&2
    echo "$DIRT" | sed 's/^/    /' >&2
    echo "" >&2
    case "$DIRT" in
        *.out*|*.err*|*slurm-*)
            echo "  Those look like this job's own SLURM logs.  They land in the SUBMIT" >&2
            echo "  directory, which is this repository.  Resubmit with both of:" >&2
            echo "      --output=\"\$LOG_DIR/%x_%j.out\" --error=\"\$LOG_DIR/%x_%j.err\"" >&2
            echo "  and delete the stray files before retrying." >&2
            ;;
        *)
            echo "  Commit them, remove them, or exclude them locally, then resubmit." >&2
            ;;
    esac
    exit 1
fi

JOBS="${SLURM_CPUS_PER_TASK:-8}"

# The step-size decision is a gate, not a convention.  An increment submitted
# before it exists is measured against a base nobody chose -- which is the
# mistake docs/05 section 0.20 recorded and this ordering exists to avoid.  A
# human can always sbatch a stage directly, so the check lives here too, not
# only in run_m4_line.sh.
# The decision was made once, on one anchor, and later arms are submitted under
# their own date tags -- so the anchor to verify against is not always this
# stage's date.  ANCHOR_DATE defaults to DATE, which is what the first batch
# did; a later arm passes the date of the anchor its step size came from.
ANCHOR_DATE="${ANCHOR_DATE:-$DATE}"
case "$STAGE" in
  opponents|no_shaping|bc|dueling|oppbc)
    "$VENV/bin/python" scripts/decide_learning_rate.py \
        --anchor "runs/m4_anchor_${ANCHOR_DATE}" --verify || {
        echo "REFUSING $STAGE. Produce the decision with:" >&2
        echo "    $VENV/bin/python scripts/decide_learning_rate.py \\" >&2
        echo "        --anchor runs/m4_anchor_${ANCHOR_DATE} \\" >&2
        echo "        --candidate runs/m4_lr1e4_${ANCHOR_DATE} \\" >&2
        echo "        --candidate runs/m4_lr5e4_${ANCHOR_DATE} --apply" >&2
        echo "  (or pass ANCHOR_DATE=<the anchor's date tag> if the decision" >&2
        echo "   was made on an earlier batch, which is the usual case.)" >&2
        exit 1
    }
    ;;
esac

echo "stage=$STAGE date=$DATE jobs=$JOBS node=$(hostname) repo=$REPO"
git -C "$REPO" rev-parse --short HEAD
git -C "$REPO" status --porcelain | head -5

exec "$VENV/bin/python" - "$STAGE" "$DATE" "$JOBS" <<'PY'
import subprocess, sys
stage, date, jobs = sys.argv[1], sys.argv[2], sys.argv[3]
# The stage table lives in run_m4_line.sh; this dispatches one of them so that
# each SLURM job is one stage and the scheduler owns the ordering.
config = {
    "smoke":      "m4_r07_a06_e00_smoke",   # 20 rounds; proves the launch path only
    "pilot":      "m4_r07_a06_e09_t02_pilot",
    "anchor":     "m4_r07_a06_e09_t02_anchor",
    "lr1e4":      "m4_r07_a06_e09_t02_lr1e4",
    "lr5e4":      "m4_r07_a06_e09_t02_lr5e4",
    "l07":        "m4_r07_a06_e09_t02_l07_cosine",
    "opponents":  "m4_r07_a06_e09_t02opp_opponents",
    "no_shaping": "m4_r07_a03_e09_t02_no_shaping",
    "bc":         "m4_r07_a06_e10_t02_bc",
    "oppbc":      "m4_r07_a06_e10_t02opp_oppbc",   # docs/05 section 0.35
    "dueling":    "m4_r08_a06_e09_t02_dueling",
}[stage]
run_id = f"m4_{stage}_{date}"
# The smoke is 20 rounds and cannot pass a learning gate by construction; it
# exists to prove module, venv, REPO and provenance all line up.
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

#!/usr/bin/env bash
#SBATCH --job-name=m4t
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=02:00:00
#SBATCH --partition=cpu
#SBATCH --mem=32G
# No --output/--error defaults, deliberately.  SLURM writes them relative to
# the submit directory, which is this repository, and creates them before
# anything runs -- so a default here leaves a dirty checkout behind.  Pass both
# on the command line instead:
#
#   sbatch --output="$LOG_DIR/%x_%j.out" --error="$LOG_DIR/%x_%j.err" \
#          --export=ALL,REPO="$PWD",VENV="$PWD/.venv",LOG_DIR="$HOME/m4_logs" \
#          scripts/slurm_m4_transfer.sh
#
# Nothing is trained here: this replays finished checkpoints against opponents
# they never met.  Two hours is generous for about 540 evaluation jobs.
set -u -o pipefail

module purge
module load devel/python/3.12.3-gnu-14.2 || {
    echo "module devel/python/3.12.3-gnu-14.2 unavailable; module avail python" >&2; exit 1; }

REPO="${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
VENV="${VENV:-$REPO/.venv}"

# One thread each: torch, the BLAS behind numpy and OpenMP all default to
# taking every core, and the parallelism here comes from the job count.
export BOMBERMAN_TORCH_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

for required in "scripts/m4_transfer.sh" "frozen_opponents" "runs"; do
    [ -e "$REPO/$required" ] || { echo "REPO=$REPO is missing $required" >&2; exit 1; }
done
[ -x "$VENV/bin/python" ] || { echo "no interpreter at $VENV/bin/python" >&2; exit 1; }

cd "$REPO" || exit 1
echo "node=$(hostname) repo=$REPO jobs=${SLURM_CPUS_PER_TASK:-16}"
git -C "$REPO" rev-parse --short HEAD

PY="$VENV/bin/python" JOBS="${SLURM_CPUS_PER_TASK:-16}" exec bash scripts/m4_transfer.sh

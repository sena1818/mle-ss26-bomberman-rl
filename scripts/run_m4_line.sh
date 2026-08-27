#!/usr/bin/env bash
# Run the whole M4 line on one machine, in order, with the gates enforced.
#
# The order is not a convenience: the pilot validates the mechanics before any
# long arm is paid for, and the anchor must show it can learn before the four
# increments measured against it are worth running.  Each stage therefore stops
# the script if its gate fails, rather than letting twenty CPU-hours produce
# numbers that cannot be interpreted.
#
#   ./scripts/run_m4_line.sh --jobs 8 --date 20260901
#   ./scripts/run_m4_line.sh --jobs 8 --date 20260901 --dry-run
#   ./scripts/run_m4_line.sh --jobs 8 --date 20260901 --from anchor
#
# Finished stages are skipped on a re-run, so it is safe to restart after an
# interruption.  Logs go outside the repository: a dirty working tree makes the
# next prepare refuse to start.

set -u -o pipefail

PYTHON="${PYTHON:-python}"
JOBS=8
DATE="$(date +%Y%m%d)"
LOG_DIR="${LOG_DIR:-$HOME/m4_logs}"
DRY_RUN=0
START_AT=""

while [ $# -gt 0 ]; do
  case "$1" in
    --jobs) JOBS="$2"; shift 2 ;;
    --date) DATE="$2"; shift 2 ;;
    --log-dir) LOG_DIR="$2"; shift 2 ;;
    --from) START_AT="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) sed -n '2,20p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

# stage-name : config : gate
#   gate "pilot"    -- full check_pilot, including the learning-curve checks
#   gate "training" -- mechanics only; the arm's science is read from the summary
STAGES=(
  "pilot:m4_r07_a06_e09_t02_pilot:pilot"
  "anchor:m4_r07_a06_e09_t02_anchor:pilot"
  "no_shaping:m4_r07_a03_e09_t02_no_shaping:training"
  "opponents:m4_r07_a06_e09_t02opp_opponents:training"
  "bc:m4_r07_a06_e10_t02_bc:training"
  "dueling:m4_r08_a06_e09_t02_dueling:training"
)

say() { printf '\n=== %s ===\n' "$*"; }

run_stage() {
  local name="$1" config="$2" gate="$3"
  local run_id="m4_${name}_${DATE}"
  local run_dir="runs/${run_id}"

  if [ -f "${run_dir}/evaluation_summary.json" ]; then
    say "${name}: already finished (${run_dir}), skipping"
    return 0
  fi

  say "${name}: ${config} -> ${run_id} (jobs ${JOBS})"
  if [ "$DRY_RUN" = "1" ]; then
    echo "would run:       $PYTHON scripts/run_experiment.py run --config experiments/${config}.json --run-id ${run_id} --jobs ${JOBS}"
    echo "would aggregate: $PYTHON scripts/aggregate_results.py --run-dir ${run_dir}"
    echo "would gate:      $PYTHON scripts/check_pilot.py --run-dir ${run_dir}$( [ "$gate" = training ] && echo ' --training-only')"
    return 0
  fi

  mkdir -p "$LOG_DIR"
  if ! $PYTHON scripts/run_experiment.py run \
        --config "experiments/${config}.json" --run-id "${run_id}" --jobs "${JOBS}" \
        > "${LOG_DIR}/${run_id}.log" 2>&1; then
    echo "FAILED: ${name} did not finish. See ${LOG_DIR}/${run_id}.log" >&2
    return 1
  fi

  $PYTHON scripts/aggregate_results.py --run-dir "${run_dir}" > "${LOG_DIR}/${run_id}.aggregate.log" 2>&1

  local gate_args=("--run-dir" "${run_dir}")
  [ "$gate" = "training" ] && gate_args+=("--training-only")
  if ! $PYTHON scripts/check_pilot.py "${gate_args[@]}" | tee "${LOG_DIR}/${run_id}.checks.log"; then
    echo "GATE FAILED: ${name}. Later stages are not worth running until this is understood." >&2
    return 1
  fi

  $PYTHON scripts/prune_runs.py --run-dir "${run_dir}" --drop-runtime --compress-logs --apply \
      >> "${LOG_DIR}/${run_id}.log" 2>&1 || true
  return 0
}

say "M4 line, date tag ${DATE}, ${JOBS} parallel jobs, logs in ${LOG_DIR}"
if [ "$DRY_RUN" != "1" ]; then
  $PYTHON -c "import torch" 2>/dev/null || {
    echo "PyTorch is not importable. M4 is the only line that needs it: python -m pip install torch" >&2
    exit 1
  }
  say "throughput on this machine (do not extrapolate from another host)"
  $PYTHON scripts/benchmark_cnn.py --rounds 10000 --steps-per-round 300 \
      --output "diagnostics/m4_throughput_$(hostname -s)_${DATE}.json" || exit 1
fi

started=0
for entry in "${STAGES[@]}"; do
  IFS=':' read -r name config gate <<< "$entry"
  if [ -n "$START_AT" ] && [ "$started" = "0" ]; then
    [ "$name" = "$START_AT" ] && started=1 || { echo "skipping ${name} (--from ${START_AT})"; continue; }
  fi
  run_stage "$name" "$config" "$gate" || exit 1
done

say "done. Quote reportable_result from each runs/<id>/evaluation_summary.json,"
echo "never selected_checkpoint. Report coins_share for every arm that has opponents."

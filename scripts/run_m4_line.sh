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
LR_SETTLED=0

while [ $# -gt 0 ]; do
  case "$1" in
    --jobs) JOBS="$2"; shift 2 ;;
    --date) DATE="$2"; shift 2 ;;
    --log-dir) LOG_DIR="$2"; shift 2 ;;
    --from) START_AT="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    --lr-settled) LR_SETTLED=1; shift ;;
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
  # Step size comes before the increments, not after them: docs/05 section 0.20
  # published "more capacity is harmful" and then found it was an artefact of a
  # step size held fixed while the network widened.  An increment measured
  # against a mis-tuned base can invert its own sign.
  "lr1e4:m4_r07_a06_e09_t02_lr1e4:training"
  "lr5e4:m4_r07_a06_e09_t02_lr5e4:training"
  "STOP_FOR_LR_DECISION::"
  "opponents:m4_r07_a06_e09_t02opp_opponents:training"
  "no_shaping:m4_r07_a03_e09_t02_no_shaping:training"
  "bc:m4_r07_a06_e10_t02_bc:training"
  "dueling:m4_r08_a06_e09_t02_dueling:training"
)

say() { printf '\n=== %s ===\n' "$*"; }

run_stage() {
  local name="$1" config="$2" gate="$3"
  local run_id="m4_${name}_${DATE}"
  local run_dir="runs/${run_id}"

  if [ -f "${run_dir}/evaluation_summary.json" ]; then
    # Re-gate rather than skip.  Skipping on the mere existence of a summary
    # would wave through a run that finished and then FAILED its gate, which is
    # exactly the run you least want to build on.
    say "${name}: already finished (${run_dir}), re-checking its gate"
    local finished_args=("--run-dir" "${run_dir}")
    [ "$gate" = "training" ] && finished_args+=("--training-only")
    if [ "$DRY_RUN" = "1" ]; then return 0; fi
    if ! $PYTHON scripts/check_pilot.py "${finished_args[@]}"; then
      echo "GATE FAILED on the existing ${name} run. Later stages are not worth running." >&2
      return 1
    fi
    return 0
  fi

  if [ -d "${run_dir}" ]; then
    # run_experiment.py refuses an existing run directory, so a run interrupted
    # mid-training cannot simply be re-entered by this script.  Say so with the
    # command rather than failing thirty seconds later with a FileExistsError.
    echo "PARTIAL: ${run_dir} exists without a summary -- ${name} was interrupted." >&2
    echo "  Resume it, or archive it and start clean, before re-running this script:" >&2
    echo "    $PYTHON scripts/run_experiment.py run --config experiments/${config}.json --run-id ${run_id} --jobs ${JOBS} --retry" >&2
    return 1
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

  if ! $PYTHON scripts/aggregate_results.py --run-dir "${run_dir}" \
        > "${LOG_DIR}/${run_id}.aggregate.log" 2>&1; then
    echo "FAILED: could not aggregate ${name}. See ${LOG_DIR}/${run_id}.aggregate.log" >&2
    return 1
  fi

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

if [ "$LR_SETTLED" = "1" ] && [ "$DRY_RUN" != "1" ]; then
  # --lr-settled asserts a decision was made and applied.  Check that it was:
  # the four downstream arms must all carry the same step size, or the "same
  # base" the increments are measured against does not exist.
  $PYTHON - <<'CHECK' || exit 1
import json, sys
from pathlib import Path
arms = ["m4_r07_a06_e09_t02opp_opponents", "m4_r07_a03_e09_t02_no_shaping",
        "m4_r07_a06_e10_t02_bc", "m4_r08_a06_e09_t02_dueling"]
rates = {a: json.loads(Path(f"experiments/{a}.json").read_text())["agent"].get("learning_rate")
         for a in arms}
if len(set(rates.values())) != 1:
    print("--lr-settled, but the downstream arms disagree on the step size:", file=sys.stderr)
    for arm, rate in rates.items():
        print(f"    {rate!r:>10}  {arm}", file=sys.stderr)
    print("  They are increments on one base; that base has to be one number.", file=sys.stderr)
    sys.exit(1)
settled = next(iter(rates.values()))
print(f"--lr-settled: the four downstream arms all run at "
      f"{'the route default (2.5e-4)' if settled is None else settled}")
CHECK
fi

say "M4 line, date tag ${DATE}, ${JOBS} parallel jobs, logs in ${LOG_DIR}"
if [ "$DRY_RUN" != "1" ]; then
  $PYTHON -c "import torch" 2>/dev/null || {
    echo "PyTorch is not importable. M4 is the only line that needs it: python -m pip install torch" >&2
    exit 1
  }
  say "throughput on this machine (do not extrapolate from another host)"
  # Written to the log directory, NOT to diagnostics/: that path is tracked and
  # is not in .gitignore, so writing there dirties the checkout and the very
  # next prepare refuses to start.  The script would have blocked itself before
  # the pilot on any clean clone.  Copy it into diagnostics/ and commit it
  # deliberately if the number is worth keeping.
  mkdir -p "$LOG_DIR"
  $PYTHON scripts/benchmark_cnn.py --rounds 10000 --steps-per-round 300 \
      --output "${LOG_DIR}/m4_throughput_$(hostname -s)_${DATE}.json" || exit 1
fi

started=0
for entry in "${STAGES[@]}"; do
  IFS=':' read -r name config gate <<< "$entry"

  if [ "$name" = "STOP_FOR_LR_DECISION" ]; then
    [ "$LR_SETTLED" = "1" ] && continue
    if [ "$DRY_RUN" = "1" ]; then
      say "STOP: the step-size decision goes here (dry run continues past it)"
      continue
    fi
    say "STOP -- the step size has to be decided before the increments run"
    cat >&2 <<'NOTE'
The three step-size arms are finished.  Everything after this point is an
increment measured AGAINST a base, so the base has to be settled first: docs/05
section 0.20 published "more capacity is harmful" and later found it was an
artefact of a step size held fixed, which is what an increment measured on a
mis-tuned base looks like.  Running stages 3-6 now would repeat that mistake
with the order rearranged but nothing else changed.

  1. Compare the three validation curves:
       python scripts/compare_runs.py --run-dir runs/m4_anchor_<date>                                       --run-dir runs/m4_lr1e4_<date>                                       --run-dir runs/m4_lr5e4_<date>
     Pre-registered rule: if neither new level is distinguishable from the
     anchor, KEEP 2.5e-4.  Distinguishable means the same 4.7 resolution G-A
     uses, on the pooled validation curve, not on a single checkpoint.

  2. If a new level wins, set agent.learning_rate to it in all four downstream
     configs and COMMIT them.  Editing the file rather than passing a flag is
     deliberate: prepare refuses a dirty checkout, and the committed config is
     the provenance record for the arm.

       experiments/m4_r07_a06_e09_t02opp_opponents.json
       experiments/m4_r07_a03_e09_t02_no_shaping.json
       experiments/m4_r07_a06_e10_t02_bc.json
       experiments/m4_r08_a06_e09_t02_dueling.json

  3. Re-run this script with --lr-settled to continue from stage 3.
NOTE
    exit 2
  fi

  if [ -n "$START_AT" ] && [ "$started" = "0" ]; then
    [ "$name" = "$START_AT" ] && started=1 || { echo "skipping ${name} (--from ${START_AT})"; continue; }
  fi
  run_stage "$name" "$config" "$gate" || exit 1
done

say "done. Quote reportable_result from each runs/<id>/evaluation_summary.json,"
echo "never selected_checkpoint. Report coins_share for every arm that has opponents."

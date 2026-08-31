#!/usr/bin/env bash
# M4 opponents arm: transfer to opponents it never trained against.
#
# Why this before any new training arm.  The tournament is other groups'
# agents (docs/05 section 0.34).  On rule_based, M4 opponents (3.8464) and
# M3's eps=0 (3.8335) are a tie, and Rainbow's +2.16 is 95% kills that do not
# survive a change of opponent -- 4.91x against rule_based, 1.11-1.87x against
# anything else.  rule_based no longer separates the candidates, and the axis
# that does has never been measured for M4.  No training happens here; every
# checkpoint already exists.
#
# Sample size is BOARD SEEDS, not repeats.  frozen_agent and coin_collector are
# deterministic, so a repeat is bit-identical and carries no information
# (section 0.34, point seven).  Twelve boards, and paired_transfer.py also
# prints the six-board answer that section 7.45.4 says is usually enough.
#
# Run from the repository root.  scaffold reads runs/<id>/job_parameters, which
# --slim-copy does not keep, so this needs the FULL source run -- on
# bwUniCluster that is $(ws_find m4)/bomberman_rl/runs/m4_opponents_20260828b.
set -euo pipefail

PY="${PY:-python}"
SOURCE="${SOURCE:-m4_opponents_20260828b}"
ROUND="${ROUND:-10000}"
JOBS="${JOBS:-8}"
BOARDS="${BOARDS:-3001 3002 3003 3004 3005 3006 3007 3008 3009 3010 3011 3012}"

# No defaults on purpose.  Which checkpoint was frozen is exactly the thing
# that has to match what section 7.45 measured; a plausible-looking default
# would read as a weaker or stronger opponent rather than as a mistake.
: "${R029_MODEL:?set R029_MODEL to the frozen R02_9 checkpoint that section 7.45 used}"
: "${RBOW_MODEL:?set RBOW_MODEL to the frozen Rainbow checkpoint (submission model: seed 1005)}"

for m in "$R029_MODEL" "$RBOW_MODEL"; do
    [ -f "$m" ] || { echo "missing frozen checkpoint: $m" >&2; exit 1; }
done
[ -d "runs/$SOURCE/job_parameters" ] || {
    echo "runs/$SOURCE/job_parameters is missing." >&2
    echo "  scaffold templates from it, and --slim-copy does not keep it, so an" >&2
    echo "  archived copy will not do.  Point SOURCE at the full run." >&2
    exit 1; }

scaffold () {                       # $1 = tag, rest = scaffold arguments
    local tag="$1"; shift
    if [ -d "runs/rep_m4t_$tag" ]; then echo "runs/rep_m4t_$tag exists, not rebuilt"; return; fi
    "$PY" scripts/repeat_measure.py scaffold \
        --source-run "$SOURCE" --run-id "rep_m4t_$tag" \
        --repeats 1 --suite classic_versus_opponents \
        --seed-role holdout --checkpoint-round "$ROUND" \
        --eval-seeds $BOARDS "$@"
}

# Rainbow is route R02_11 (categorical head, noisy layers), NOT R02_9.  Loading
# it on the wrong route builds a different network, which would read as a
# weaker opponent rather than as a mistake.
scaffold r029 --opponents frozen_agent frozen_agent frozen_agent \
              --frozen-route R02_9  --frozen-model "$R029_MODEL"
scaffold rbow --opponents frozen_agent frozen_agent frozen_agent \
              --frozen-route R02_11 --frozen-model "$RBOW_MODEL"
scaffold coin --opponents coin_collector_agent coin_collector_agent coin_collector_agent

# scaffold writes job_parameters/*.json.  There is no `run --config` path for a
# scaffolded run: each job is executed on its own, parallelised here.
for tag in r029 rbow coin; do
    echo "=== running rep_m4t_$tag"
    ls "runs/rep_m4t_$tag/job_parameters"/*.json \
      | xargs -P "$JOBS" -I{} "$PY" scripts/run_experiment.py job --job-file {}
done

echo
for tag in r029 rbow coin; do
    "$PY" scripts/repeat_measure.py report --run-dir "runs/rep_m4t_$tag"
done
cat <<'NOTE'

Familiar-opponent control, already measured: 3 x rule_based = 3.8464 +- 0.0827
M3 on these same three, pooled (docs/05 section 0.34):
    Rainbow 2.9541    eps0 2.8522    R02_9 2.6663
NOTE

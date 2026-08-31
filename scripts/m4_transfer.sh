#!/usr/bin/env bash
# Do the M4 arms hold up against opponents they never trained against?
#
# Why this before any new training arm.  The tournament is other groups'
# agents (docs/05 section 0.34).  On rule_based, M4 opponents (3.8464) and
# M3's eps=0 (3.8335) are a tie, and Rainbow's +2.16 there is 95% kills that
# do not survive a change of opponent -- 4.91x against rule_based, 1.11-1.87x
# against anything else.  rule_based no longer separates the candidates, and
# the axis that does has never been measured for M4.  Nothing is trained here.
#
# Three source arms, because two different stories predict the same win and
# they imply opposite next steps:
#   opponents  representation AND opponent exposure
#   anchor     the CNN representation alone -- it never saw an opponent
#   bc         whether the cloned start changes transfer at all
# If anchor transfers too, the credit is the representation's.  If only
# opponents does, the credit is the training distribution's.
#
# Sample size is BOARD SEEDS, not repeats: frozen_agent and coin_collector are
# deterministic, so a repeat is bit-identical and carries no information
# (section 0.34, point seven).  Seeds 4001-4012 are not arbitrary -- they are
# the ones section 7.45 measured the M3 arms on, and only the same boards can
# be paired against those numbers.
#
# Run from the repository root.  scaffold templates from
# runs/<source>/job_parameters, which --slim-copy does not keep, so this needs
# the FULL runs -- on bwUniCluster, $(ws_find m4)/bomberman_rl/runs/.
set -euo pipefail

PY="${PY:-python}"
ROUND="${ROUND:-10000}"
JOBS="${JOBS:-16}"
DATE="${DATE:-20260828b}"
ARMS="${ARMS:-opponents anchor bc}"
BOARDS="${BOARDS:-4001 4002 4003 4004 4005 4006 4007 4008 4009 4010 4011 4012}"

# Committed, not pulled out of runs/: frozen_opponents/README.md explains why.
# The sha256 of each is recorded into every scaffolded run's provenance, so a
# swapped opponent fails the comparison instead of quietly changing it.
#   R02_9  08576fe67b1d98aab7875371ad007aba5696fab86f17cc40c403656765f215d8
#   R02_11 013b2bd47375a2cc2bc2588eb9813f31311e814edae54557591904216b8a684c
R029_MODEL="${R029_MODEL:-frozen_opponents/R02_9_seed1001_round05000.npz}"
RBOW_MODEL="${RBOW_MODEL:-frozen_opponents/R02_11_rainbow_seed1005_round10000.npz}"

for m in "$R029_MODEL" "$RBOW_MODEL"; do
    [ -f "$m" ] || { echo "missing frozen checkpoint: $m" >&2; exit 1; }
done
for arm in $ARMS; do
    [ -d "runs/m4_${arm}_${DATE}/job_parameters" ] || {
        echo "runs/m4_${arm}_${DATE}/job_parameters is missing." >&2
        echo "  scaffold templates from it and --slim-copy does not keep it, so an" >&2
        echo "  archived copy will not do.  Run this where the full runs live." >&2
        exit 1; }
done

# Rainbow is route R02_11 -- categorical head, noisy layers -- not R02_9.
# Building it on the wrong route makes a different network, which would read as
# a weaker opponent rather than as a mistake.
opponent_args () {
    case "$1" in
      r029) echo "--opponents frozen_agent frozen_agent frozen_agent --frozen-route R02_9  --frozen-model $R029_MODEL" ;;
      rbow) echo "--opponents frozen_agent frozen_agent frozen_agent --frozen-route R02_11 --frozen-model $RBOW_MODEL" ;;
      coin) echo "--opponents coin_collector_agent coin_collector_agent coin_collector_agent" ;;
    esac
}

for arm in $ARMS; do
  for foe in rbow coin r029; do
    run="rep_m4t_${arm}_${foe}"
    if [ -d "runs/$run" ]; then echo "runs/$run exists, not rebuilt"; continue; fi
    "$PY" scripts/repeat_measure.py scaffold \
        --source-run "m4_${arm}_${DATE}" --run-id "$run" \
        --repeats 1 --suite classic_versus_opponents \
        --seed-role holdout --checkpoint-round "$ROUND" \
        --eval-seeds $BOARDS $(opponent_args "$foe")
  done
done

# scaffold writes job_parameters/*.json; there is no `run --config` path for a
# scaffolded run.  Each job runs on its own, parallelised here.
for arm in $ARMS; do
  for foe in rbow coin r029; do
    run="rep_m4t_${arm}_${foe}"
    echo "=== running $run"
    ls "runs/$run/job_parameters"/*.json \
      | xargs -P "$JOBS" -I{} "$PY" scripts/run_experiment.py job --job-file {}
  done
done

echo
for arm in $ARMS; do
  for foe in rbow coin r029; do
    "$PY" scripts/repeat_measure.py report --run-dir "runs/rep_m4t_${arm}_${foe}"
  done
done
cat <<'NOTE'

Familiar opponent, already measured: M4 opponents vs 3 x rule_based = 3.8464 +- 0.0827
M3 on these same boards and opponents, pooled (docs/05 section 0.34):
    Rainbow 2.9541    eps0 2.8522    R02_9 2.6663
NOTE

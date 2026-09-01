#!/bin/bash
# Start one or more experiment runs on the Hetzner host, detached from the
# shell that started them, one after another.
#
#   scripts/hetzner_launch.sh JOBS NAME CONFIG:RUN_ID [CONFIG:RUN_ID ...]
#
# JOBS is the --jobs value every run gets; NAME names the chain's log under
# /root/.  Runs execute sequentially so a chain of small arms can share a fixed
# core budget with a long arm started separately.  Logs go to /root, not into
# the checkout: a log inside the repository dirties the worktree and the next
# prepare refuses it (docs/M3_SESSION_PROMPT.md).
set -euo pipefail
if [ "$#" -lt 3 ]; then
  echo "usage: $0 JOBS NAME CONFIG:RUN_ID [CONFIG:RUN_ID ...]" >&2
  exit 2
fi
JOBS="$1"; NAME="$2"; shift 2
HERE="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="${BOMBERMAN_PYTHON:-/root/bomberman-r01/.venv/bin/python}"
LOG="/root/${NAME}.log"
CHAIN="/root/${NAME}.chain.sh"
{
  echo "#!/bin/bash"
  echo "cd '$HERE'"
  for pair in "$@"; do
    config="${pair%%:*}"; run_id="${pair##*:}"
    echo "echo \"=== \$(date -u +%FT%TZ) start $run_id\""
    echo "'$PYTHON' scripts/run_experiment.py run --config '$config' --run-id '$run_id' --jobs '$JOBS'"
    echo "echo \"=== \$(date -u +%FT%TZ) done $run_id exit \$?\""
  done
  echo "echo CHAIN_DONE"
} > "$CHAIN"
chmod +x "$CHAIN"
nohup "$CHAIN" > "$LOG" 2>&1 < /dev/null &
echo "started $NAME (pid $!), log $LOG"

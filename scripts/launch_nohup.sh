#!/usr/bin/env bash
# Launch one background (nohup) run per config, all in parallel; survives logout.
#   bash scripts/launch_nohup.sh TOTAL_WORKERS config/a.yaml config/b.yaml ...
# TOTAL_WORKERS is split evenly across the configs. Logs: logs/<config>_<time>.log
# Run scripts/setup_server.sh first (it downloads the data once, so parallel runs do not race).
set -euo pipefail
cd "$(dirname "$0")/.."
TOTAL="$1"; shift
[ "$#" -ge 1 ] || { echo "give at least one config"; exit 1; }
[ -x .venv/bin/python ] || { echo "run scripts/setup_server.sh first"; exit 1; }
PER=$(( TOTAL / $# )); [ "$PER" -ge 1 ] || PER=1
mkdir -p logs
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
for c in "$@"; do
  name=$(basename "$c" .yaml)
  log="logs/${name}_${STAMP}.log"
  nohup bash -c ".venv/bin/python -m dte.run --config '$c' --workers $PER && \
    RUN_DIR=\$(grep -o 'results/[^ ]*' '$log' | head -1) && \
    .venv/bin/python -m dte.aggregate \"\$RUN_DIR\" --bootstrap 2000 --seed 1 > \"\$RUN_DIR/summary.txt\" && \
    echo FINISHED \$RUN_DIR" > "$log" 2>&1 &
  echo "$!  $c  -> $log  ($PER workers)" | tee -a logs/pids_${STAMP}.txt
done
echo "monitor: tail -f logs/*_${STAMP}.log    stop all: kill \$(awk '{print \$1}' logs/pids_${STAMP}.txt)"

#!/usr/bin/env bash
# One-command pipeline for the compute server.
#   bash scripts/run_server.sh config/experiment.yaml 32
# Stops at the first error (set -e); every step's output is in results/<run>/.
set -euo pipefail
CONFIG="$1"
WORKERS="$2"
cd "$(dirname "$0")/.."
if [ ! -d .venv ]; then
  python3 -m venv .venv
  .venv/bin/pip install -q -r requirements.txt
fi
.venv/bin/python -m pytest -q tests          # validation tests must pass before any experiment
.venv/bin/python -m dte.fetch --config "$CONFIG"
.venv/bin/python -m dte.run --config "$CONFIG" --workers "$WORKERS" --chunk 50
RUN_DIR=$(ls -dt results/*/ | head -1)
.venv/bin/python -m dte.aggregate "$RUN_DIR" --bootstrap 2000 --seed 1 > "${RUN_DIR}summary.txt"
echo "finished: ${RUN_DIR}"

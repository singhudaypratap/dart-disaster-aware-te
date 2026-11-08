#!/usr/bin/env bash
# One-time (and after every git pull) setup: virtualenv, validation tests, data download.
#   bash scripts/setup_server.sh config/experiment.yaml [more configs...]
set -euo pipefail
cd "$(dirname "$0")/.."
[ -d .venv ] || python3 -m venv .venv
.venv/bin/pip install -q -r requirements.txt
.venv/bin/python -m pytest -q tests
for c in "$@"; do .venv/bin/python -m dte.fetch --config "$c"; done
echo "setup OK"

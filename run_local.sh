#!/usr/bin/env bash
set -euo pipefail
if [ ! -x .venv/bin/python ]; then python3 -m venv .venv; fi
source .venv/bin/activate
python -m pip install --upgrade pip
if [ ! -f .venv/dependencies_installed.flag ]; then
  pip install -r requirements.txt
  touch .venv/dependencies_installed.flag
fi
python -m uvicorn server:app --host 0.0.0.0 --port 8000

#!/usr/bin/env bash
set -euo pipefail

script_directory="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export ENTITY_V02_API_BASE="${ENTITY_V02_API_BASE:-http://eval-v3-api-owen-2.pegasus-eval.svc.cluster.local:8090}"
export ENTITY_V02_SYNC_SECONDS="${ENTITY_V02_SYNC_SECONDS:-60}"

exec "${script_directory}/.venv/bin/streamlit" run "${script_directory}/app.py" \
  --server.address 127.0.0.1 \
  --server.port 8501 \
  --server.headless true \
  --browser.gatherUsageStats false

#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${SCRIPT_DIR}/logs"
mkdir -p "${LOG_DIR}"

if ! pgrep -f "port-forward svc/eval-v3-api-lia 18091:8090" >/dev/null; then
  nohup kubectl --context training -n pegasus-eval \
    port-forward svc/eval-v3-api-lia 18091:8090 \
    >"${LOG_DIR}/port_forward.log" 2>&1 &
  echo "$!" >"${LOG_DIR}/port_forward.pid"
fi

sleep 3

set -a
source "${HOME}/lia-ooo-bot/.env"
set +a

python3 "${SCRIPT_DIR}/run_eval.py" \
  --host "http://localhost:18091" \
  --poll \
  --poll-seconds 120 \
  --timeout-seconds 21600 \
  --slack-channel "#fun-lia-trashcan"

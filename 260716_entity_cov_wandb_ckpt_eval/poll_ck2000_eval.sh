#!/usr/bin/env bash
set -euo pipefail

EVAL_RUN_ID="${EVAL_RUN_ID:-3b6b3e0f-b778-50a9-859e-9bb319900ca2}"
POLL_SECONDS="${POLL_SECONDS:-180}"
SLACK_CHANNEL="${SLACK_CHANNEL:-#fun-lia-trashcan}"
EVAL_RUNS_URL="${EVAL_RUNS_URL:-http://xplatform-training.twelve.labs/sme-studio/api/eval/runs}"

if [[ -f "${HOME}/.env" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "${HOME}/.env"
  set +a
fi

post_slack() {
  local text="$1"
  if [[ -z "${SLACK_BOT_TOKEN:-}" || -z "${SLACK_CHANNEL}" ]]; then
    return 0
  fi
  python3 - "$SLACK_CHANNEL" "$text" <<'PY'
import json
import os
import sys
import urllib.request

channel, text = sys.argv[1], sys.argv[2]
request = urllib.request.Request(
    "https://slack.com/api/chat.postMessage",
    data=json.dumps({"channel": channel, "text": text}).encode("utf-8"),
    headers={
        "Authorization": f"Bearer {os.environ['SLACK_BOT_TOKEN']}",
        "Content-Type": "application/json; charset=utf-8",
    },
    method="POST",
)
with urllib.request.urlopen(request, timeout=30) as response:
    payload = json.loads(response.read().decode("utf-8"))
if not payload.get("ok"):
    raise SystemExit(f"Slack post failed: {payload}")
PY
}

extract_status() {
  python3 -c '
import json
import sys

payload = json.load(sys.stdin).get("evalRun", {})
print(json.dumps({
    "status": payload.get("status", ""),
    "completed": payload.get("completed", ""),
    "failed": payload.get("failed", ""),
    "totalTasks": payload.get("totalTasks", ""),
    "batchId": payload.get("batchId", ""),
}, sort_keys=True))
'
}

post_slack "[cc-generated] Started polling ck2000 entity coverage eval: ${EVAL_RUN_ID}"

last_summary=""
while true; do
  status_payload="$(curl -sS --max-time 60 "${EVAL_RUNS_URL}/${EVAL_RUN_ID}")"
  status_json="$(printf '%s' "${status_payload}" | extract_status)"
  status="$(python3 -c 'import json,sys; print(json.load(sys.stdin)["status"])' <<< "${status_json}")"
  completed="$(python3 -c 'import json,sys; print(json.load(sys.stdin)["completed"])' <<< "${status_json}")"
  failed="$(python3 -c 'import json,sys; print(json.load(sys.stdin)["failed"])' <<< "${status_json}")"
  total_tasks="$(python3 -c 'import json,sys; print(json.load(sys.stdin)["totalTasks"])' <<< "${status_json}")"
  batch_id="$(python3 -c 'import json,sys; print(json.load(sys.stdin)["batchId"])' <<< "${status_json}")"
  now="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  summary="ck2000 eval: status=${status}, completed=${completed}/${total_tasks}, failed=${failed}, batch=${batch_id}"
  echo "{\"checked_at\":\"${now}\",\"eval_run_id\":\"${EVAL_RUN_ID}\",\"status\":\"${status}\",\"completed\":\"${completed}\",\"failed\":\"${failed}\",\"totalTasks\":\"${total_tasks}\",\"batchId\":\"${batch_id}\"}"
  if [[ "${summary}" != "${last_summary}" ]]; then
    post_slack "[cc-generated] ${summary}"
    last_summary="${summary}"
  fi
  if [[ "${status}" =~ ^(completed|failed|cancelled|interrupted)$ ]]; then
    post_slack "[cc-generated] ck2000 entity coverage eval reached terminal status: ${summary}"
    exit 0
  fi
  sleep "${POLL_SECONDS}"
done

#!/usr/bin/env bash
set -uo pipefail

EVAL_RUN_ID="${EVAL_RUN_ID:-6a75028e-b51d-5b44-a07c-21d2d3b0ff43}"
EVAL_API_BASE="${EVAL_API_BASE:-http://eval-v3-api-owen-2.pegasus-eval.svc.cluster.local:8090}"
POLL_SECONDS="${POLL_SECONDS:-120}"
HEARTBEAT_SECONDS="${HEARTBEAT_SECONDS:-1200}"
SLACK_CHANNEL="${SLACK_CHANNEL:-#fun-lia-trashcan}"

set -a
# shellcheck disable=SC1090
source "${HOME}/lia-ooo-bot/.env"
set +a
: "${SLACK_BOT_TOKEN:?SLACK_BOT_TOKEN is required}"

post_slack() {
  local message="$1"
  curl -sS -X POST https://slack.com/api/chat.postMessage \
    -H "Authorization: Bearer ${SLACK_BOT_TOKEN}" \
    -H "Content-Type: application/json; charset=utf-8" \
    --data "$(jq -n --arg channel "${SLACK_CHANNEL}" --arg text "${message}" \
      '{channel: $channel, text: $text, unfurl_links: false}')" \
    | jq -c '{ok, error, channel, ts}'
}

last_summary=""
last_notification_epoch=0
post_slack "[cc-generated] Started polling A-1790 entity_cov_v02 64K retry: ${EVAL_RUN_ID}"

while true; do
  response="$(curl -sS --max-time 30 "${EVAL_API_BASE}/eval/runs/${EVAL_RUN_ID}" 2>/dev/null || true)"
  summary="$(jq -rc '.evalRun | "status=\(.status // "unknown") completed=\(.completed // 0)/\(.totalTasks // "?") failed=\(.failed // 0) batch=\(.batchId // "unknown")"' <<<"${response}" 2>/dev/null || printf 'status=unreachable')"
  status="$(jq -r '.evalRun.status // "unreachable"' <<<"${response}" 2>/dev/null || printf 'unreachable')"
  current_epoch="$(date +%s)"

  printf '[%s] %s\n' "$(date -Iseconds)" "${summary}"
  if [[ "${summary}" != "${last_summary}" || $((current_epoch - last_notification_epoch)) -ge "${HEARTBEAT_SECONDS}" ]]; then
    post_slack "[cc-generated] A-1790 entity_cov_v02 64K retry: ${summary}"
    last_summary="${summary}"
    last_notification_epoch="${current_epoch}"
  fi

  if [[ "${status}" =~ ^(completed|failed|cancelled|interrupted)$ ]]; then
    post_slack "[cc-generated] A-1790 entity_cov_v02 64K retry reached terminal status: ${summary}"
    exit 0
  fi
  sleep "${POLL_SECONDS}"
done

#!/usr/bin/env bash
# [cc-generated] Poll two harness logs and post Pegasus job states to Slack.
set -euo pipefail

: "${SFT_LOG_PATH:?required}"
: "${KIAN_LOG_PATH:?required}"

ORCHESTRATOR_URL="${ORCHESTRATOR_URL:-http://xplatform-training.twelve.labs/orchestrator}"
POLL_SEC="${POLL_SEC:-180}"
SLACK_CHANNEL="${SLACK_CHANNEL:-fun-lia-trashcan}"
CPU_HOST="${CPU_HOST:-cpu}"
CPU_ENV_FILE="${CPU_ENV_FILE:-/home/jeongyeon-nam/lia-ooo-bot/.env}"

summarize_log() {
  local label=$1
  local log_path=$2
  local jobs_file
  jobs_file=$(mktemp)
  trap 'rm -f "$jobs_file"' RETURN

  rg -o 'job [0-9a-f]{8}-[0-9a-f-]{27}' "$log_path" | awk '{print $2}' | sort -u >"$jobs_file"
  local total
  total=$(wc -l <"$jobs_file" | tr -d ' ')
  if [[ "$total" == "0" ]]; then
    printf '%s total=0 pending=0 processing=0 completed=0 failed=0 error=0 cancelled=0 unknown=0' "$label"
    return
  fi

  export ORCHESTRATOR_URL
  local statuses
  statuses=$(xargs -P 30 -n 1 sh -c '
    curl -fsS --max-time 20 "$ORCHESTRATOR_URL/jobs/$1" | jq -r ".status // \"UNKNOWN\"" || echo UNKNOWN
  ' _ <"$jobs_file")

  count_status() {
    local status_name=$1
    local count
    count=$(printf '%s\n' "$statuses" | grep -cx "$status_name" || true)
    printf '%s' "${count:-0}"
  }

  printf '%s total=%s pending=%s processing=%s completed=%s failed=%s error=%s cancelled=%s unknown=%s' \
    "$label" "$total" \
    "$(count_status JOB_STATUS_PENDING)" \
    "$(count_status PROCESSING)" \
    "$(count_status COMPLETED)" \
    "$(count_status FAILED)" \
    "$(count_status ERROR)" \
    "$(count_status JOB_STATUS_CANCELLED)" \
    "$(count_status UNKNOWN)"
}

post_to_slack() {
  local message=$1
  local payload
  payload=$(python3 -c 'import json, sys; print(json.dumps({"channel": sys.argv[1], "text": sys.argv[2]}))' "$SLACK_CHANNEL" "$message")
  printf '%s' "$payload" | ssh -T "$CPU_HOST" "set -a; . '$CPU_ENV_FILE'; set +a; curl -sS -X POST https://slack.com/api/chat.postMessage -H \"Authorization: Bearer \$SLACK_BOT_TOKEN\" -H 'Content-Type: application/json; charset=utf-8' --data @- >/dev/null"
}

while true; do
  sft_status=$(summarize_log "sft-2node" "$SFT_LOG_PATH")
  kian_status=$(summarize_log "kian-1node" "$KIAN_LOG_PATH")
  message="[cc-generated] Pegasus jobs (3m): ${sft_status} | ${kian_status}"
  printf '%s\n' "$message"
  post_to_slack "$message"
  sleep "$POLL_SEC"
done

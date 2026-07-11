#!/bin/bash
# Generic long-running job status poller -> Slack.
#
# Every $POLL_SEC, runs $STATUS_CMD and captures its stdout as current status.
# Sends slack messages on: init, status change, periodic heartbeat, terminal.
#
# Required env:
#   JOB_ID           unique id (used in messages + log filename)
#   STATUS_CMD       shell command (string) that echoes current status; may
#                    reference $JOB_ID
#   SUCCESS_PATTERN  extended regex; when status matches -> exit 0 with :white_check_mark:
#   FAILURE_PATTERN  extended regex; when status matches -> exit 1 with :x:
#
# Optional env:
#   LABEL         human description (default: $JOB_ID)
#   ARTIFACT_URL  mentioned in success message (default: none)
#   CHANNEL       slack channel (default: fun-lia-trashcan)
#   ENV_FILE      .env containing SLACK_BOT_TOKEN (default: ~/pegasus/.env)
#   TOKEN_FILE    slack token path (default: ~/tmp/.slack_bot_token)
#   POLL_SEC      poll interval (default: 120)
#   HEARTBEAT_SEC heartbeat interval (default: 1200 = 20min)
#
# Usage (background on cpu node):
#   ssh cpu "JOB_ID=... STATUS_CMD=... SUCCESS_PATTERN=... FAILURE_PATTERN=... \
#     nohup ~/poll_generic.sh > ~/poll.out 2>&1 &"

set -uo pipefail

: "${JOB_ID:?required}"
: "${STATUS_CMD:?required}"
: "${SUCCESS_PATTERN:?required}"
: "${FAILURE_PATTERN:?required}"

LABEL="${LABEL:-$JOB_ID}"
ARTIFACT_URL="${ARTIFACT_URL:-}"
CHANNEL="${CHANNEL:-fun-lia-trashcan}"
ENV_FILE="${ENV_FILE:-$HOME/pegasus/.env}"
TOKEN_FILE="${TOKEN_FILE:-$HOME/tmp/.slack_bot_token}"
POLL_SEC="${POLL_SEC:-120}"
HEARTBEAT_SEC="${HEARTBEAT_SEC:-1200}"
LOG="$HOME/poll_${JOB_ID}.log"

export JOB_ID
if [[ -z ${SLACK_BOT_TOKEN:-} && -f $ENV_FILE ]]; then
  set -a
  source "$ENV_FILE"
  set +a
fi
if [[ -n ${SLACK_BOT_TOKEN:-} ]]; then
  SLACK_TOKEN="$SLACK_BOT_TOKEN"
else
  SLACK_TOKEN="$(cat "$TOKEN_FILE")"
fi

slack() {
  local msg="$1"
  local payload
  payload=$(python3 -c "import json,sys; print(json.dumps({'channel': sys.argv[1], 'text': sys.argv[2]}))" "$CHANNEL" "$msg")
  curl -sS -X POST https://slack.com/api/chat.postMessage \
    -H "Authorization: Bearer ${SLACK_TOKEN}" \
    -H "Content-Type: application/json; charset=utf-8" \
    --data "$payload" >>"$LOG" 2>&1
  echo "" >>"$LOG"
}

check_status() {
  local out
  out="$(eval "$STATUS_CMD" 2>/dev/null)" || out=""
  [ -z "$out" ] && out="Unknown"
  printf '%s' "$out"
}

prev_status=""
last_heartbeat=$(date +%s)

slack "[cc-generated] :rocket: polling \`${JOB_ID}\` — ${LABEL}"

while true; do
  status="$(check_status)"
  ts=$(date -Iseconds)
  echo "$ts $status" >>"$LOG"

  if [ "$status" != "$prev_status" ] && [ -n "$prev_status" ]; then
    slack "[cc-generated] :arrows_counterclockwise: \`${JOB_ID}\` status: \`${prev_status}\` → \`${status}\`"
  fi
  prev_status="$status"

  now=$(date +%s)
  if [ $((now - last_heartbeat)) -ge "$HEARTBEAT_SEC" ]; then
    slack "[cc-generated] :heartbeat: \`${JOB_ID}\` still \`${status}\`"
    last_heartbeat=$now
  fi

  if [[ "$status" =~ $SUCCESS_PATTERN ]]; then
    if [ -n "$ARTIFACT_URL" ]; then
      slack "[cc-generated] :white_check_mark: \`${JOB_ID}\` succeeded. Artifact: \`${ARTIFACT_URL}\`"
    else
      slack "[cc-generated] :white_check_mark: \`${JOB_ID}\` succeeded."
    fi
    exit 0
  fi
  if [[ "$status" =~ $FAILURE_PATTERN ]]; then
    slack "[cc-generated] :x: \`${JOB_ID}\` failed. Last status: \`${status}\`"
    exit 1
  fi

  sleep "$POLL_SEC"
done

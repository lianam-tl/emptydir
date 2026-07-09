#!/bin/bash
# Generic tlab job status poller — posts to Slack #fun-lia-trashcan.
# Terminates on Succeeded/Complete/Failed.
#
# Usage:
#   JOB=<tlab-job-name> S3_OUT=<optional-artifact-url> LABEL=<optional-desc> \
#     nohup ~/poll_tlab_job.sh > ~/poll.out 2>&1 &
#
# Env:
#   JOB           (required) tlab job / pytorchjob name in namespace `research`
#   S3_OUT        (optional) artifact path to mention on success message
#   LABEL         (optional) human description (default: JOB name)
#   CHANNEL       (optional) slack channel (default: fun-lia-trashcan)
#   TOKEN_FILE    (optional) default ~/tmp/.slack_bot_token
#   POLL_SEC      (optional) default 120
#   HEARTBEAT_SEC (optional) default 1200 (20 min)

set -uo pipefail

JOB_NAME="${JOB:?set JOB=<tlab-job-name>}"
CHANNEL="${CHANNEL:-fun-lia-trashcan}"
S3_OUT="${S3_OUT:-}"
LABEL="${LABEL:-$JOB_NAME}"
TOKEN_FILE="${TOKEN_FILE:-$HOME/tmp/.slack_bot_token}"
LOG="$HOME/poll_${JOB_NAME}.log"
POLL_SEC="${POLL_SEC:-120}"
HEARTBEAT_SEC="${HEARTBEAT_SEC:-1200}"

SLACK_TOKEN="$(cat "$TOKEN_FILE")"

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

kstatus() {
  kubectl get pytorchjob "$JOB_NAME" -n research -o jsonpath='{.status.conditions[?(@.status=="True")].type}' 2>/dev/null || echo "Unknown"
}

prev_status=""
last_heartbeat=0

slack "[cc-generated] :rocket: polling tlab job \`${JOB_NAME}\` — ${LABEL}"

while true; do
  status="$(kstatus)"
  [ -z "$status" ] && status="Unknown"
  ts=$(date -Iseconds)
  echo "$ts $status" >>"$LOG"

  if [ "$status" != "$prev_status" ] && [ -n "$prev_status" ]; then
    slack "[cc-generated] :arrows_counterclockwise: \`${JOB_NAME}\` status: \`${prev_status}\` → \`${status}\`"
  fi
  prev_status="$status"

  now=$(date +%s)
  if [ $((now - last_heartbeat)) -ge "$HEARTBEAT_SEC" ]; then
    slack "[cc-generated] :heartbeat: \`${JOB_NAME}\` still \`${status}\`"
    last_heartbeat=$now
  fi

  case "$status" in
    *Succeeded*|*Complete*)
      if [ -n "$S3_OUT" ]; then
        slack "[cc-generated] :white_check_mark: \`${JOB_NAME}\` finished. Artifact at \`${S3_OUT}\`"
      else
        slack "[cc-generated] :white_check_mark: \`${JOB_NAME}\` finished."
      fi
      exit 0
      ;;
    *Failed*)
      slack "[cc-generated] :x: \`${JOB_NAME}\` failed. Check \`tlab logs ${JOB_NAME}\` or s3://.../logs/${JOB_NAME}/"
      exit 1
      ;;
  esac

  sleep "$POLL_SEC"
done

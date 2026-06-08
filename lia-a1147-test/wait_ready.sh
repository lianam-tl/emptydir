#!/usr/bin/env bash
# [cc-generated] Wait until smoke-test pod is ready; slack progress every 2 min.
set -uo pipefail

set -a; source /Users/long8v/pegasus/.env; set +a

POD=lia-a1147-vllm-video-test
NS=pegasus-platform
CHANNEL='#fun-lia-trashcan'

slack() {
  curl -s -X POST 'https://slack.com/api/chat.postMessage' \
    -H "Authorization: Bearer $SLACK_BOT_TOKEN" \
    -H "Content-Type: application/json; charset=utf-8" \
    --data-binary "$(/usr/bin/jq -nc --arg c "$CHANNEL" --arg t "$1" '{channel:$c, text:$t}')" >/dev/null
}

START=$(date +%s)
slack ":hourglass: [A-1147] pod ${POD} applied, waiting for ready (model=Qwen3.5-0.8B)..."

LAST_HB=$START
while true; do
  PHASE=$(kubectl get pod -n "$NS" "$POD" -o jsonpath='{.status.phase}' 2>/dev/null)
  READY=$(kubectl get pod -n "$NS" "$POD" -o jsonpath='{.status.containerStatuses[0].ready}' 2>/dev/null)
  NOW=$(date +%s)
  ELAPSED_M=$(( (NOW - START) / 60 ))

  if [ "$READY" = "true" ]; then
    slack ":white_check_mark: [A-1147] pod ready after ${ELAPSED_M}m. proceed with port-forward + smoke test."
    echo "READY"
    exit 0
  fi
  if [ "$PHASE" = "Failed" ] || [ "$PHASE" = "Succeeded" ] || [ "$PHASE" = "Unknown" ]; then
    TAIL=$(kubectl logs -n "$NS" "$POD" -c worker --tail=40 2>&1 | sed 's/[`]//g')
    slack ":x: [A-1147] pod left Running phase=${PHASE} after ${ELAPSED_M}m. tail:\`\`\`${TAIL}\`\`\`"
    echo "FAILED phase=$PHASE"
    exit 1
  fi

  # heartbeat every 2 min
  if [ $(( NOW - LAST_HB )) -ge 120 ]; then
    LAST_LOG=$(kubectl logs -n "$NS" "$POD" -c worker --tail=1 2>/dev/null | sed 's/[`]//g' | cut -c1-160)
    slack ":hourglass_flowing_sand: [A-1147] pod still loading — t=${ELAPSED_M}m, phase=${PHASE}, ready=${READY}. tail=\`${LAST_LOG}\`"
    LAST_HB=$NOW
  fi
  sleep 15
done

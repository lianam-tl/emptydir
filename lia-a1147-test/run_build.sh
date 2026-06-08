#!/usr/bin/env bash
# [cc-generated] Wrapper: build vllm-video image for A-1147, slack progress to #fun-lia-trashcan.
set -uo pipefail

LOG=/tmp/a1147/build.log
BRANCH=lia/A-1147-vllm-direct-n-rollouts
CHANNEL='#fun-lia-trashcan'

set -a
source /Users/long8v/pegasus/.env
set +a

slack() {
  local text="$1"
  curl -s -X POST 'https://slack.com/api/chat.postMessage' \
    -H "Authorization: Bearer $SLACK_BOT_TOKEN" \
    -H "Content-Type: application/json; charset=utf-8" \
    --data-binary "$(/usr/bin/jq -nc \
      --arg t "$text" --arg c "$CHANNEL" \
      '{channel:$c, text:$t}')" >/dev/null
}

START_TS=$(date +%s)
slack ":rocket: [A-1147] vllm-video build started — branch=\`$BRANCH\` host=$(hostname -s) log=\`$LOG\`"

# Background heartbeat every 5 min
(
  while sleep 300; do
    LINES=$(wc -l < "$LOG" 2>/dev/null || echo 0)
    ELAPSED=$(( ($(date +%s) - START_TS) / 60 ))
    LAST=$(tail -1 "$LOG" 2>/dev/null | sed 's/[`]//g' | cut -c1-160)
    slack ":hourglass_flowing_sand: [A-1147] build still running — t=${ELAPSED}m, $LINES log lines. tail=\`$LAST\`"
  done
) &
HEARTBEAT_PID=$!

# Actual build — point REPO_DIR at the already-checked-out worktree
# (main ~/xplatform can't `git checkout` a branch that's checked out in a worktree)
REPO_DIR=/Users/long8v/xplatform-A-1147-n-rollouts \
  bash /Users/long8v/emptydir/build_vllm_video_local.sh "$BRANCH" >> "$LOG" 2>&1
RC=$?

kill "$HEARTBEAT_PID" 2>/dev/null || true

ELAPSED_TOTAL=$(( ($(date +%s) - START_TS) / 60 ))

if [ "$RC" -eq 0 ] && grep -q "^Pushed:" "$LOG"; then
  IMG=$(grep "^Pushed:" "$LOG" | tail -1 | awk '{print $2}')
  echo "$IMG" > /tmp/a1147/image_uri
  slack ":white_check_mark: [A-1147] build done in ${ELAPSED_TOTAL}m. image=\`$IMG\`"
else
  TAIL=$(tail -40 "$LOG" | sed 's/[`]//g')
  slack ":x: [A-1147] build FAILED (rc=$RC) in ${ELAPSED_TOTAL}m. tail:\`\`\`$TAIL\`\`\`"
fi

exit $RC

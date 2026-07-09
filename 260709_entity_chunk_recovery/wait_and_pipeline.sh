#!/bin/bash
# Wait for the 2 backfill batches (chunk20m/chunk45m entity-h0-sme-1300) to
# reach "batch completed" (all requests either completed or failed), then run
# raw-output recovery + gpt-5.2 scoring for those 2 combos and Slack the metrics.
#
# Usage: nohup ./wait_and_pipeline.sh > /path/to/log 2>&1 &

set -u
BR_URL="http://xplatform-training.twelve.labs/batch-request/batch-runs"
BATCH_20M="batch-21affe62-cea7-4e61-8acd-01132abc79a5"   # chunk20m/entity-h0-sme-1300, 13 tasks
BATCH_45M="batch-6b20b325-345e-419e-b175-c72d3576b420"   #  chunk45m/entity-h0-sme-1300,  7 tasks
TOKEN=$(cat ~/tmp/.slack_bot_token)
CHANNEL="#fun-lia-trashcan"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON="$HOME/.venv/bin/python"

slack() {
  local msg="$1"
  curl -sS -X POST https://slack.com/api/chat.postMessage \
    -H "Authorization: Bearer $TOKEN" \
    -H "Content-Type: application/json; charset=utf-8" \
    --data "$(jq -Rn --arg t "$msg" --arg c "$CHANNEL" '{channel:$c, text:$t}')" \
    > /dev/null 2>&1
}

# Return "done" if all requests are finished (completed OR failed), else "wait".
batch_done() {
  local bid="$1"
  local payload
  payload=$(curl -sS "$BR_URL/$bid/requests?limit=1000") || { echo "err"; return; }
  local counts
  counts=$(echo "$payload" | jq -r '
    .requests
    | reduce .[] as $r ({total: 0, done: 0, running: 0, completed: 0, failed: 0};
        .total += 1
        | if ($r.record.status // "") == "completed" then .completed += 1 | .done += 1
          elif ($r.record.status // "") == "failed"    then .failed    += 1 | .done += 1
          else .running += 1
          end)
    | "\(.total) \(.done) \(.completed) \(.failed) \(.running)"
  ')
  echo "$counts"
}

slack "[entity-cov 1300-backfill] 시작 — 2 runs 대기 중 (chunk20m batch=${BATCH_20M:0:15}…, chunk45m batch=${BATCH_45M:0:15}…). 각 batch=completed 시 recover+score 자동 실행."

while true; do
  read t20 d20 c20 f20 r20 <<< "$(batch_done "$BATCH_20M")"
  read t45 d45 c45 f45 r45 <<< "$(batch_done "$BATCH_45M")"
  echo "$(date -Iseconds)  chunk20m: total=$t20 done=$d20 (c=$c20 f=$f20 r=$r20)  |  chunk45m: total=$t45 done=$d45 (c=$c45 f=$f45 r=$r45)"
  if [ "$t20" = "13" ] && [ "$d20" = "13" ] && [ "$t45" = "7" ] && [ "$d45" = "7" ]; then
    echo "$(date -Iseconds)  both batches finished; running recovery + scoring"
    break
  fi
  sleep 60
done

slack "[entity-cov 1300-backfill] batches 완료 (chunk20m: $c20/13 completed $f20 failed, chunk45m: $c45/7 completed $f45 failed) — recover + score 시작."

# Run recovery for just the 2 new combos.
cd "$SCRIPT_DIR"
$PYTHON recover_raw_outputs.py --only chunk20m/entity-h0-sme-1300 2>&1 | tee -a wait_pipeline.log
$PYTHON recover_raw_outputs.py --only chunk45m/entity-h0-sme-1300 2>&1 | tee -a wait_pipeline.log

# Run scoring for just the 2 new combos (parallel doesn't help — only 2 combos, script default parallel=3 fine)
$PYTHON score_recovered.py --only chunk20m/entity-h0-sme-1300 --parallel 1 2>&1 | tee -a wait_pipeline.log
$PYTHON score_recovered.py --only chunk45m/entity-h0-sme-1300 --parallel 1 2>&1 | tee -a wait_pipeline.log

# Slack final metrics (grab from tail of tee'd log).
FINAL_TAIL=$(tail -40 wait_pipeline.log | tr '\n' '|' | tail -c 1600)
slack "[entity-cov 1300-backfill] DONE — tail: $FINAL_TAIL"

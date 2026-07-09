#!/bin/bash
# Wait for the 3 pegasus-15 batches (chunk10m/20m/45m) to reach batch-complete,
# then run raw-output recovery + gpt-5.2 scoring for those 3 combos and Slack
# the metrics.

set -u
BR_URL="http://xplatform-training.twelve.labs/batch-request/batch-runs"
BATCH_10M="batch-081550ac-e194-4e11-ad04-ca07109428da"   # chunk10m/pegasus-15, 20 tasks
BATCH_20M="batch-2677c8bf-e279-4ff7-a197-d5abeae6bd72"   # chunk20m/pegasus-15, 13 tasks
BATCH_45M="batch-0d9b0c49-05f9-40dc-83bf-344fe1b63e03"   # chunk45m/pegasus-15,  7 tasks
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

batch_done() {
  local bid="$1"
  local payload
  payload=$(curl -sS "$BR_URL/$bid/requests?limit=1000") || { echo "err"; return; }
  echo "$payload" | jq -r '
    .requests
    | reduce .[] as $r ({total: 0, done: 0, completed: 0, failed: 0, running: 0};
        .total += 1
        | if ($r.record.status // "") == "completed" then .completed += 1 | .done += 1
          elif ($r.record.status // "") == "failed"    then .failed    += 1 | .done += 1
          else .running += 1
          end)
    | "\(.total) \(.done) \(.completed) \(.failed) \(.running)"
  '
}

slack "[pegasus-15 grid] 시작 — 3 runs 대기 중 (chunk10m/20m/45m). 각 batch=completed 시 recover+score 자동 실행."

while true; do
  read t10 d10 c10 f10 r10 <<< "$(batch_done "$BATCH_10M")"
  read t20 d20 c20 f20 r20 <<< "$(batch_done "$BATCH_20M")"
  read t45 d45 c45 f45 r45 <<< "$(batch_done "$BATCH_45M")"
  echo "$(date -Iseconds)  chunk10m: $d10/$t10 (c=$c10 f=$f10)  |  chunk20m: $d20/$t20 (c=$c20 f=$f20)  |  chunk45m: $d45/$t45 (c=$c45 f=$f45)"
  if [ "$t10" = "20" ] && [ "$d10" = "20" ] && [ "$t20" = "13" ] && [ "$d20" = "13" ] && [ "$t45" = "7" ] && [ "$d45" = "7" ]; then
    echo "$(date -Iseconds)  all 3 batches finished; running recovery + scoring"
    break
  fi
  sleep 60
done

slack "[pegasus-15 grid] batches 완료 (10m: $c10/20 20m: $c20/13 45m: $c45/7) — recover + score 시작."

cd "$SCRIPT_DIR"
for chunk in chunk10m chunk20m chunk45m; do
  $PYTHON recover_raw_outputs.py --only $chunk/pegasus-15 2>&1 | tee -a wait_pipeline_pegasus15.log
done
for chunk in chunk10m chunk20m chunk45m; do
  $PYTHON score_recovered.py --only $chunk/pegasus-15 --parallel 1 2>&1 | tee -a wait_pipeline_pegasus15.log
done

FINAL_TAIL=$(tail -60 wait_pipeline_pegasus15.log | tr '\n' '|' | tail -c 1800)
slack "[pegasus-15 grid] DONE — tail: $FINAL_TAIL"

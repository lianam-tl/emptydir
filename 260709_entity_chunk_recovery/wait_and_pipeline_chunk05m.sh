#!/bin/bash
# Wait for the 4 chunk_05m batches (pegasus-15, pegasus-15-sft, pegasus-15-rl,
# entity-h0-sme-2200) to reach batch-complete, then run raw-output recovery +
# gpt-5.2 scoring for those 4 combos and Slack the metrics.

set -u
BR_URL="http://xplatform-training.twelve.labs/batch-request/batch-runs"
BATCH_PEGASUS_15="batch-14bcdded-96db-4980-a12d-77eb5665c09a"   # pegasus-15
BATCH_PEGASUS_15_SFT="batch-832d3937-a2e5-49c3-968e-34176db3039a"   # pegasus-15-sft
BATCH_PEGASUS_15_RL="batch-60988c72-e5e4-4b8a-baee-6301e1e3220b"   # pegasus-15-rl
BATCH_ENTITY_H0_SME_2200="batch-fce615b6-c352-4750-bb8e-a0d3ae6f7900"   # entity-h0-sme-2200
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
    | reduce .[] as $r ({total:0, done:0, completed:0, failed:0, running:0};
        .total += 1
        | if ($r.record.status // "") == "completed" then .completed += 1 | .done += 1
          elif ($r.record.status // "") == "failed"    then .failed    += 1 | .done += 1
          else .running += 1
          end)
    | "\(.total) \(.done) \(.completed) \(.failed)"
  '
}

slack "[chunk_05m grid] 시작 — 4 runs 대기 중 (pegasus-15, pegasus-15-sft, pegasus-15-rl, entity-h0-sme-2200). 각 batch=completed 시 recover+score 자동 실행."

while true; do
  read tp dp cp fp <<< "$(batch_done "$BATCH_PEGASUS_15")"
  read tps dps cps fps <<< "$(batch_done "$BATCH_PEGASUS_15_SFT")"
  read tpr dpr cpr fpr <<< "$(batch_done "$BATCH_PEGASUS_15_RL")"
  read tes des ces fes <<< "$(batch_done "$BATCH_ENTITY_H0_SME_2200")"
  echo "$(date -Iseconds)  pegasus-15: $dp/$tp  pegasus-15-sft: $dps/$tps  pegasus-15-rl: $dpr/$tpr  entity-h0-sme-2200: $des/$tes"
  if [ "$tp" = "34" ] && [ "$dp" = "34" ] && \
     [ "$tps" = "34" ] && [ "$dps" = "34" ] && \
     [ "$tpr" = "34" ] && [ "$dpr" = "34" ] && \
     [ "$tes" = "34" ] && [ "$des" = "34" ]; then
    echo "$(date -Iseconds)  all 4 batches finished; running recovery + scoring"
    break
  fi
  sleep 60
done

slack "[chunk_05m grid] 4 batches 완료 — recover + score 시작."

cd "$SCRIPT_DIR"
for model in pegasus-15 pegasus-15-sft pegasus-15-rl entity-h0-sme-2200; do
  $PYTHON recover_raw_outputs.py --only chunk05m/$model 2>&1 | tee -a wait_pipeline_chunk05m.log
done
for model in pegasus-15 pegasus-15-sft pegasus-15-rl entity-h0-sme-2200; do
  $PYTHON score_recovered.py --only chunk05m/$model --parallel 1 2>&1 | tee -a wait_pipeline_chunk05m.log
done

FINAL_TAIL=$(tail -80 wait_pipeline_chunk05m.log | tr '\n' '|' | tail -c 2000)
slack "[chunk_05m grid] DONE — tail: $FINAL_TAIL"

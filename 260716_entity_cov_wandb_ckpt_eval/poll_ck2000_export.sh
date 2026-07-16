#!/usr/bin/env bash
set -euo pipefail

JOB_NAME="${JOB_NAME:-export-ck2000-safetensors-ti9cpt}"
POLL_SECONDS="${POLL_SECONDS:-180}"
SLACK_CHANNEL="${SLACK_CHANNEL:-#fun-lia-trashcan}"
CKPT_PATH="${CKPT_PATH:-s3://tl-data-training-pegasus-us-west-2/checkpoints/jeongyeon-nam/consol_260416_clean_filter_less_aug_highres_h0_mn_sme_2x_lr2e-6_qwen3_5_27b-base/checkpoint-2000}"
PAYLOAD_PATH="${PAYLOAD_PATH:-payloads/lia-entity-cov-chunk10m-h0-ck2000-20260716-165505.json}"
RESULTS_PATH="${RESULTS_PATH:-payloads/submission_results.jsonl}"
EVAL_RUNS_URL="${EVAL_RUNS_URL:-http://xplatform-training.twelve.labs/sme-studio/api/eval/runs}"

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

last_status=""
post_slack "[cc-generated] Started polling ck2000 safetensors export: ${JOB_NAME}"

submit_eval_and_poll() {
  local response_path="payloads/ck2000_submit_response.json"
  local eval_run_id
  local status
  local completed
  local failed
  local total_tasks
  local batch_id

  post_slack "[cc-generated] ck2000 safetensors are ready; submitting entity coverage eval."
  curl -sS -X POST "${EVAL_RUNS_URL}" \
    -H 'Content-Type: application/json' \
    --data-binary "@${PAYLOAD_PATH}" > "${response_path}"

  eval_run_id="$(python3 - "${response_path}" <<'PY'
import json
import sys
with open(sys.argv[1], encoding="utf-8") as file:
    payload = json.load(file)
print(payload.get("evalRun", {}).get("id", ""))
PY
)"
  if [[ -z "${eval_run_id}" ]]; then
    post_slack "[cc-generated] ck2000 eval submit failed: $(cat "${response_path}")"
    cat "${response_path}"
    exit 1
  fi

  python3 - "${PAYLOAD_PATH}" "${response_path}" "${RESULTS_PATH}" <<'PY'
import json
import sys
from pathlib import Path

payload_path = Path(sys.argv[1])
response_path = Path(sys.argv[2])
results_path = Path(sys.argv[3])
payload = json.loads(payload_path.read_text(encoding="utf-8"))
response = json.loads(response_path.read_text(encoding="utf-8"))
record = {
    "config": payload["config"],
    "dataset": payload["dataset"],
    "eval_run_id": response["evalRun"]["id"],
    "model_path": payload["modelPath"],
    "name": payload["name"],
    "payload_path": str(payload_path),
    "response": response,
    "source": "h0_ck2000",
    "step": 2000,
}
with results_path.open("a", encoding="utf-8") as file:
    file.write(json.dumps(record, sort_keys=True) + "\n")
PY

  post_slack "[cc-generated] ck2000 entity coverage eval submitted: ${eval_run_id}"
  while true; do
    status_payload="$(curl -sS "${EVAL_RUNS_URL}/${eval_run_id}")"
    status="$(python3 -c 'import json,sys; p=json.load(sys.stdin).get("evalRun",{}); print(p.get("status",""))' <<< "${status_payload}")"
    completed="$(python3 -c 'import json,sys; p=json.load(sys.stdin).get("evalRun",{}); print(p.get("completed",""))' <<< "${status_payload}")"
    failed="$(python3 -c 'import json,sys; p=json.load(sys.stdin).get("evalRun",{}); print(p.get("failed",""))' <<< "${status_payload}")"
    total_tasks="$(python3 -c 'import json,sys; p=json.load(sys.stdin).get("evalRun",{}); print(p.get("totalTasks",""))' <<< "${status_payload}")"
    batch_id="$(python3 -c 'import json,sys; p=json.load(sys.stdin).get("evalRun",{}); print(p.get("batchId",""))' <<< "${status_payload}")"
    echo "{\"eval_run_id\":\"${eval_run_id}\",\"status\":\"${status}\",\"completed\":\"${completed}\",\"failed\":\"${failed}\",\"totalTasks\":\"${total_tasks}\",\"batchId\":\"${batch_id}\"}"
    post_slack "[cc-generated] ck2000 eval: status=${status}, ${completed}/${total_tasks}, failed=${failed}, batch=${batch_id}"
    if [[ "${status}" =~ ^(completed|failed|cancelled|interrupted)$ ]]; then
      return 0
    fi
    sleep "${POLL_SECONDS}"
  done
}

while true; do
  status="$(tlab jobs 2>/dev/null | awk -v job="${JOB_NAME}" '$1 == job {print $6}')"
  status="${status:-UNKNOWN}"
  safetensors_count="$(aws s3 ls --profile training "${CKPT_PATH}/" 2>/dev/null | awk '/\\.safetensors$/ {count += 1} END {print count + 0}')"
  index_count="$(aws s3 ls --profile training "${CKPT_PATH}/" 2>/dev/null | awk '/model\\.safetensors\\.index\\.json$/ {count += 1} END {print count + 0}')"
  now="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "{\"checked_at\":\"${now}\",\"job\":\"${JOB_NAME}\",\"status\":\"${status}\",\"safetensors_count\":${safetensors_count},\"index_count\":${index_count}}"

  summary="ck2000 export ${JOB_NAME}: status=${status}, safetensors=${safetensors_count}, index=${index_count}"
  if [[ "${summary}" != "${last_status}" ]]; then
    post_slack "[cc-generated] ${summary}"
    last_status="${summary}"
  fi

  if [[ "${safetensors_count}" -gt 0 && "${index_count}" -gt 0 ]]; then
    post_slack "[cc-generated] ck2000 safetensors export appears ready: ${CKPT_PATH}"
    submit_eval_and_poll
    exit 0
  fi
  if [[ "${status}" =~ ^(Succeeded|Completed|Failed|Error|Cancelled|Stopped)$ ]]; then
    post_slack "[cc-generated] ck2000 export reached terminal status without confirmed safetensors: ${summary}"
    exit 1
  fi
  sleep "${POLL_SECONDS}"
done

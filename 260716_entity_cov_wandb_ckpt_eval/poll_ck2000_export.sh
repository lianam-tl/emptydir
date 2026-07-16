#!/usr/bin/env bash
set -euo pipefail

JOB_NAME="${JOB_NAME:-export-ck2000-safetensors-ti9cpt}"
POLL_SECONDS="${POLL_SECONDS:-180}"
SLACK_CHANNEL="${SLACK_CHANNEL:-#fun-lia-trashcan}"
CKPT_PATH="${CKPT_PATH:-s3://tl-data-training-pegasus-us-west-2/checkpoints/jeongyeon-nam/consol_260416_clean_filter_less_aug_highres_h0_mn_sme_2x_lr2e-6_qwen3_5_27b-base/checkpoint-2000}"

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
    exit 0
  fi
  if [[ "${status}" =~ ^(Succeeded|Completed|Failed|Error|Cancelled|Stopped)$ ]]; then
    post_slack "[cc-generated] ck2000 export reached terminal status without confirmed safetensors: ${summary}"
    exit 1
  fi
  sleep "${POLL_SECONDS}"
done

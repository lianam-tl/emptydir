#!/usr/bin/env bash
# [cc-generated] Report indexing progress for two local e2e harness stacks.
#
# Each stack exposes one job-status endpoint per knowledge store. This script
# sums those endpoints and emits a single stable status line for poll_generic.
set -euo pipefail

: "${SFT_BASE_URL:?required}"
: "${SFT_KNOWLEDGE_STORE_IDS:?required comma-separated IDs}"
: "${KIAN_BASE_URL:?required}"
: "${KIAN_KNOWLEDGE_STORE_IDS:?required comma-separated IDs}"

summarize_run() {
  local label=$1
  local base_url=$2
  local knowledge_store_ids=$3
  local response
  local total=0 success=0 failed=0 running=0 dispatched=0 created=0 cancelled=0 unavailable=0
  local knowledge_store_id

  IFS=',' read -r -a ids <<<"$knowledge_store_ids"
  for knowledge_store_id in "${ids[@]}"; do
    response=$(curl -fsS --max-time 15 "$base_url/_internal/knowledge-stores/$knowledge_store_id/job-status") || {
      unavailable=$((unavailable + 1))
      continue
    }
    total=$((total + $(jq -r '.jobs_total // 0' <<<"$response")))
    success=$((success + $(jq -r '.jobs_success // 0' <<<"$response")))
    failed=$((failed + $(jq -r '.jobs_failed // 0' <<<"$response")))
    running=$((running + $(jq -r '.jobs_running // 0' <<<"$response")))
    dispatched=$((dispatched + $(jq -r '.jobs_dispatched // 0' <<<"$response")))
    created=$((created + $(jq -r '.jobs_created // 0' <<<"$response")))
    cancelled=$((cancelled + $(jq -r '.jobs_cancelled // 0' <<<"$response")))
  done

  printf '%s processed=%d/%d running=%d dispatched=%d created=%d failed=%d cancelled=%d unavailable=%d' \
    "$label" "$success" "$total" "$running" "$dispatched" "$created" "$failed" "$cancelled" "$unavailable"
}

sft_status=$(summarize_run "sft-2node" "$SFT_BASE_URL" "$SFT_KNOWLEDGE_STORE_IDS")
kian_status=$(summarize_run "kian-1node" "$KIAN_BASE_URL" "$KIAN_KNOWLEDGE_STORE_IDS")
printf '%s | %s' "$sft_status" "$kian_status"

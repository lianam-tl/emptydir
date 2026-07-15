#!/usr/bin/env bash
# Summarize shared-queue pressure and one model worker's readiness for a live e2e run.
set -euo pipefail

worker_name=${WORKER_NAME:-kian-socerl-s60-4node}
queue_stats=$(curl -fsS http://xplatform-training.twelve.labs/orchestrator/jobs/stats)
unhealthy_pods=$(kubectl --context training get pods -n pegasus-platform -o json \
  | jq --arg worker_name "model-${worker_name}-" '[.items[] | select(.metadata.name | startswith($worker_name)) | select(.status.phase != "Running" or ([.status.containerStatuses[]?.ready] | all | not))] | length')

printf 'state=running queue_PENDING=%s queue_PROCESSING=%s unhealthy_%s_pods=%s\n' \
  "$(jq -r '(.byStatus.JOB_STATUS_PENDING // .byStatus.PENDING // 0)' <<<"$queue_stats")" \
  "$(jq -r '.byStatus.PROCESSING' <<<"$queue_stats")" \
  "$worker_name" \
  "$unhealthy_pods"

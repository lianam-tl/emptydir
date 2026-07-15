#!/usr/bin/env bash
# Summarize shared-queue pressure and Kian worker readiness for a live e2e run.
set -euo pipefail

queue_stats=$(curl -fsS http://xplatform-training.twelve.labs/orchestrator/jobs/stats)
unhealthy_pods=$(kubectl --context training get pods -n pegasus-platform -o json \
  | jq '[.items[] | select(.metadata.name | startswith("model-kian-socerl-s60-4node-")) | select(.status.phase != "Running" or ([.status.containerStatuses[]?.ready] | all | not))] | length')

printf 'state=running queue_PENDING=%s queue_PROCESSING=%s unhealthy_kian_pods=%s\n' \
  "$(jq -r '(.byStatus.JOB_STATUS_PENDING // .byStatus.PENDING // 0)' <<<"$queue_stats")" \
  "$(jq -r '.byStatus.PROCESSING' <<<"$queue_stats")" \
  "$unhealthy_pods"

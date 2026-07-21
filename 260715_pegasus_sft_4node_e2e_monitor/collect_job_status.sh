#!/usr/bin/env bash
# [cc-generated] Summarize orchestrator statuses for a recorded e2e job set.
set -euo pipefail

job_ids_file=${1:?usage: collect_job_status.sh JOB_IDS_FILE}
orchestrator_url=${ORCHESTRATOR_URL:-http://xplatform-training.twelve.labs/orchestrator}
parallel_requests=${PARALLEL_REQUESTS:-20}
export ORCHESTRATOR_URL="$orchestrator_url"

if [[ ! -s "$job_ids_file" ]]; then
  echo "state=unknown total=0 PENDING=0 PROCESSING=0 COMPLETED=0 FAILED=0 CANCELLED=0 UNKNOWN=0"
  exit 0
fi

statuses=$(xargs -r -P "$parallel_requests" -n 1 sh -c '
  curl -fsS --max-time 20 "$ORCHESTRATOR_URL/jobs/$1" \
    | jq -r ".status // \"UNKNOWN\"" \
    || echo UNKNOWN
' _ <"$job_ids_file")

count_status() {
  local status_name=$1
  local count
  count=$(printf '%s\n' "$statuses" | grep -cx "$status_name" || true)
  printf '%s' "${count:-0}"
}

total=$(wc -l <"$job_ids_file" | tr -d ' ')
pending=$(count_status JOB_STATUS_PENDING)
processing=$(count_status PROCESSING)
completed=$(count_status COMPLETED)
failed=$(count_status FAILED)
cancelled=$(count_status JOB_STATUS_CANCELLED)
unknown=$(count_status UNKNOWN)

if ((failed > 0 || cancelled > 0)); then
  state=failed
elif ((pending == 0 && processing == 0 && unknown == 0)); then
  state=completed
else
  state=running
fi

printf 'state=%s total=%s PENDING=%s PROCESSING=%s COMPLETED=%s FAILED=%s CANCELLED=%s UNKNOWN=%s\n' \
  "$state" "$total" "$pending" "$processing" "$completed" "$failed" "$cancelled" "$unknown"

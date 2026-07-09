#!/usr/bin/env bash
# [cc-generated] Poll entity-coverage chunk_10m/chunk_20m/chunk_45m eval-service runs.
set -uo pipefail

WORK_DIR="$HOME/lia_eval_poll_chunk10_20"
RUNS_JSON="$WORK_DIR/runs.json"
STATE_JSON="$WORK_DIR/state.json"
LOG_FILE="$WORK_DIR/poll.log"
EVAL_HEALTH_URL="http://127.0.0.1:18090/readyz"
EVAL_BASE_URL="http://127.0.0.1:18090/api/eval"
BATCH_BASE_URL="http://xplatform-training.twelve.labs/batch-request"
CHANNEL="#fun-lia-trashcan"
POLL_SECONDS=300
MAX_LOOPS=576

mkdir -p "$WORK_DIR"
[ -f "$STATE_JSON" ] || printf '{}\n' > "$STATE_JSON"

load_env_file() {
  local env_file="$1"
  [ -f "$env_file" ] || return 0
  set -a
  # shellcheck disable=SC1090
  . "$env_file"
  set +a
}
load_env_file "$HOME/lia-ooo-bot/.env"
load_env_file "$HOME/pegasus/.env"
load_env_file "$HOME/.env"

SLACK_TOKEN="${SLACK_BOT_TOKEN:-${SLACK_USER_TOKEN:-}}"

log() {
  printf '[%s] %s\n' "$(date -Is)" "$*" | tee -a "$LOG_FILE" >/dev/null
}

post_slack() {
  local text="$1"
  if [ -z "$SLACK_TOKEN" ]; then
    log "SLACK token missing; would send: $text"
    return 0
  fi
  curl -sS -X POST https://slack.com/api/chat.postMessage \
    -H "Authorization: Bearer ${SLACK_TOKEN}" \
    -H "Content-Type: application/json; charset=utf-8" \
    --data "$(jq -n --arg channel "$CHANNEL" --arg text "$text" '{channel:$channel,text:$text,unfurl_links:false,unfurl_media:false}')" \
    | jq -c '{ok,error,channel,ts}' >> "$LOG_FILE" 2>&1 || true
}

ensure_eval_port_forward() {
  if curl -sS -m 3 "$EVAL_HEALTH_URL" >/dev/null 2>&1; then
    return 0
  fi
  pkill -f "kubectl --context training -n pegasus-platform port-forward svc/eval-service-lia 18090:8090" >/dev/null 2>&1 || true
  nohup kubectl --context training -n pegasus-platform port-forward svc/eval-service-lia 18090:8090 \
    > "$WORK_DIR/eval-service-port-forward.log" 2>&1 &
  sleep 3
}

compact_eval_status() {
  jq -c '{status:.evalRun.status,total:.evalRun.totalTasks,completed:.evalRun.completed,failed:.evalRun.failed,batchId:.evalRun.batchId,completedAt:.evalRun.completedAt}'
}

compact_batch_status() {
  jq -c '{status,submitted:.submitted_count,completed:.completed_count,failed:.failed_count,cancelled:.cancelled_count,started_at,completed_at}'
}

summary_line() {
  local label="$1" run_id="$2" batch_id="$3" eval_compact="$4" batch_compact="$5"
  local eval_status total completed failed batch_status submitted batch_completed batch_failed cancelled
  eval_status=$(jq -r '.status // "unknown"' <<<"$eval_compact")
  total=$(jq -r '.total // "?"' <<<"$eval_compact")
  completed=$(jq -r '.completed // 0' <<<"$eval_compact")
  failed=$(jq -r '.failed // 0' <<<"$eval_compact")
  batch_status=$(jq -r '.status // "unknown"' <<<"$batch_compact")
  submitted=$(jq -r '.submitted // 0' <<<"$batch_compact")
  batch_completed=$(jq -r '.completed // 0' <<<"$batch_compact")
  batch_failed=$(jq -r '.failed // 0' <<<"$batch_compact")
  cancelled=$(jq -r '.cancelled // 0' <<<"$batch_compact")
  printf '[entity-eval chunk10/20/45] %s: eval=%s %s/%s failed=%s | batch=%s submitted=%s completed=%s failed=%s cancelled=%s | run=%s batch=%s' \
    "$label" "$eval_status" "$completed" "$total" "$failed" "$batch_status" "$submitted" "$batch_completed" "$batch_failed" "$cancelled" "$run_id" "$batch_id"
}

metrics_suffix() {
  local run_id="$1"
  local metrics
  metrics=$(curl -sS -m 20 "$EVAL_BASE_URL/runs/$run_id/evaluation" 2>/dev/null || true)
  if jq -e '.evaluation' >/dev/null 2>&1 <<<"$metrics"; then
    printf '\nmetrics: `%s`' "$(jq -c '.evaluation' <<<"$metrics")"
  fi
}

post_slack "[entity-eval chunk10/20/45] started polling nine TP=2 runs for Pegasus1.5-2604, ff-sft, entity-h0-added. Poll interval=${POLL_SECONDS}s."

loop=0
while [ "$loop" -lt "$MAX_LOOPS" ]; do
  loop=$((loop + 1))
  ensure_eval_port_forward
  all_terminal=1

  while IFS=$'\t' read -r label run_id batch_id; do
    eval_json=$(curl -sS -m 20 "$EVAL_BASE_URL/runs/$run_id" 2>/dev/null || printf '{}')
    batch_json=$(curl -sS -m 20 "$BATCH_BASE_URL/batch-runs/$batch_id" 2>/dev/null || printf '{}')

    eval_compact=$(compact_eval_status <<<"$eval_json" 2>/dev/null || printf '{"status":"unknown"}')
    batch_compact=$(compact_batch_status <<<"$batch_json" 2>/dev/null || printf '{"status":"unknown"}')
    signature=$(jq -cn --argjson eval "$eval_compact" --argjson batch "$batch_compact" '{eval:$eval,batch:$batch}')
    previous=$(jq -c --arg id "$run_id" '.[$id] // empty' "$STATE_JSON")

    eval_status=$(jq -r '.status // "unknown"' <<<"$eval_compact")
    if [[ ! "$eval_status" =~ ^(completed|failed|cancelled)$ ]]; then
      all_terminal=0
    fi

    if [ "$signature" != "$previous" ]; then
      line=$(summary_line "$label" "$run_id" "$batch_id" "$eval_compact" "$batch_compact")
      if [[ "$eval_status" =~ ^(completed|failed|cancelled)$ ]]; then
        line="${line}$(metrics_suffix "$run_id")"
      fi
      post_slack "$line"
      tmp_state=$(mktemp)
      jq --arg id "$run_id" --argjson sig "$signature" '.[$id]=$sig' "$STATE_JSON" > "$tmp_state" && mv "$tmp_state" "$STATE_JSON"
    fi
  done < <(jq -r '.[] | [.label,.run_id,.batch_id] | @tsv' "$RUNS_JSON")

  if [ "$all_terminal" -eq 1 ]; then
    post_slack "[entity-eval chunk10/20/45] polling finished; all nine tracked runs are terminal."
    exit 0
  fi
  sleep "$POLL_SECONDS"
done

post_slack "[entity-eval chunk10/20/45] polling stopped after max loops; at least one run is still non-terminal."

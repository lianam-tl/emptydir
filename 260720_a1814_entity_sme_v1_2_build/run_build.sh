#!/usr/bin/env bash
set -uo pipefail

RUN_ROOT="${RUN_ROOT:-/fsx/jeongyeon-nam/a1814-entity-sme-v1-2-build}"
WORKTREE="${WORKTREE:-/fsx/jeongyeon-nam/pegasus-worktrees/lia-a-1814-scale-entity-sme-v1-2}"
PYTHON="${PYTHON:-/home/jeongyeon-nam/pegasus/.venv/bin/python}"

mkdir -p "$RUN_ROOT" "$RUN_ROOT/cache" "$RUN_ROOT/hf-cache" "$RUN_ROOT/output" "$RUN_ROOT/tmp"
rm -f "$RUN_ROOT/build.exitcode"

set -a
source "$WORKTREE/.env"
set +a

export AWS_PROFILE=training
export HF_DATASETS_CACHE="$RUN_ROOT/hf-cache/datasets"
export HF_HOME="$RUN_ROOT/hf-cache"
export PYTHONUNBUFFERED=1
export TMPDIR="$RUN_ROOT/tmp"
export XDG_CACHE_HOME="$RUN_ROOT/cache"

ulimit -n 4096
cd "$WORKTREE"

printf '\n[%s] Starting A-1814 full build\n' "$(date --iso-8601=seconds)" >> "$RUN_ROOT/build.log"
printf '%s\n' \
  "$PYTHON -u data/h0_from_dc/build_duration_diverse_tdf.py --workers 8 --resume --output-dir $RUN_ROOT/output --report-html $RUN_ROOT/a1814_full_report.html" \
  > "$RUN_ROOT/build.command"

status=0
"$PYTHON" -u data/h0_from_dc/build_duration_diverse_tdf.py \
  --workers 8 \
  --resume \
  --output-dir "$RUN_ROOT/output" \
  --report-html "$RUN_ROOT/a1814_full_report.html" \
  >> "$RUN_ROOT/build.log" 2>&1 || status=$?

printf '%s\n' "$status" > "$RUN_ROOT/build.exitcode"
printf '[%s] Build exited with status %s\n' "$(date --iso-8601=seconds)" "$status" >> "$RUN_ROOT/build.log"
exit "$status"

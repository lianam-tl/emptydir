#!/bin/bash
# [cc-generated] Driver for ckpt_watcher.py — polls every 14min.
set -u
DIR="$HOME/eval-polling"
PY="$HOME/lia-ooo-bot/.venv/bin/python"
LOG="$DIR/ckpt_watcher.log"
echo "===== ckpt_watcher loop start $(date) pid=$$ =====" >>"$LOG"
while true; do
  "$PY" "$DIR/ckpt_watcher.py" >>"$LOG" 2>&1
  sleep 840
done

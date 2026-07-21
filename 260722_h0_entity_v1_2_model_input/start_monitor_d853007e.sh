#!/bin/bash
set -euo pipefail

MONITOR_DIR="$HOME/260721_a1689_h0_mn_sme_preprocess"

set -a
source "$HOME/lia-ooo-bot/.env"
set +a

export JOB_ID="d853007e"
export STATUS_CMD="python3 $MONITOR_DIR/get_xplatform_status.py \$JOB_ID"
export SUCCESS_PATTERN="^(succeeded|completed)$"
export FAILURE_PATTERN="^(failed|errored|error)$"
export LABEL="h0_entity_v1_2 model-input merge (36 sources)"
export ARTIFACT_URL="s3://tl-data-training-pegasus-us-west-2/annotation/preprocessed_datasets/model_input/h0_entity_v1_2/mixture_stats.json"
export CHANNEL="fun-lia-trashcan"
export POLL_SEC="120"
export HEARTBEAT_SEC="1200"

exec "$MONITOR_DIR/poll_generic.sh"

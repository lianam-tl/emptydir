#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$HOME/260721_a1689_h0_mn_sme_preprocess"

set -a
source "$HOME/lia-ooo-bot/.env"
set +a

export JOB_ID="6c8c8dcc"
export STATUS_CMD="python3 $SCRIPT_DIR/get_xplatform_status.py \$JOB_ID"
export SUCCESS_PATTERN="^(succeeded|completed)$"
export FAILURE_PATTERN="^(failed|errored|error)$"
export LABEL="A-1689 H0 movies/news SME base preprocess (6,000s)"
export ARTIFACT_URL="s3://tl-data-training-pegasus-us-west-2/annotation/preprocessed_datasets/base/tl_h0_movies_and_news_sme/default_sft_sme_h0_highres/sft_sme/"
export CHANNEL="fun-lia-trashcan"
export POLL_SEC="120"
export HEARTBEAT_SEC="1200"

exec "$SCRIPT_DIR/poll_generic.sh"

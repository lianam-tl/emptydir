#!/bin/bash
# Pull base → build → push the vllm-video worker image locally.
# Usage:
#   SHA=$(git -C /path/to/xplatform rev-parse --short=7 HEAD) \
#   XPLATFORM=/path/to/xplatform \
#   bash build.sh
#
# Optional: if SLACK_BOT_TOKEN is set in env (e.g. sourced from ~/pegasus/.env)
# each milestone pings #fun-lia-trashcan.
set -e
exec >> /tmp/cttkw_build.log 2>&1
echo "=== $(date -u +%FT%TZ) STARTING ==="

: "${SHA:?SHA required (e.g. \`SHA=\$(git -C \$XPLATFORM rev-parse --short=7 HEAD)\`)}"
: "${XPLATFORM:?XPLATFORM root path required}"

BASE_IMG=${BASE_IMG:-219219941196.dkr.ecr.us-west-2.amazonaws.com/x-platform-model-workers:vllm-v0.19.0-20260411}
OUT_TAG=${OUT_TAG:-476114115052.dkr.ecr.us-west-2.amazonaws.com/tl-data-training-pegasus-vllm-video:lia-vllm-chat-template-kwargs-${SHA}}

notify() {
  [ -z "$SLACK_BOT_TOKEN" ] && return 0
  curl -s -X POST https://slack.com/api/chat.postMessage \
    -H "Authorization: Bearer $SLACK_BOT_TOKEN" \
    -H "Content-Type: application/json" \
    -d "{\"channel\":\"#fun-lia-trashcan\",\"text\":\"$1\"}" > /dev/null
}

echo "=== $(date -u +%FT%TZ) PULL base image ==="
docker pull "$BASE_IMG"
notify "[cttkw e2e] base image pulled. starting docker build (sha=$SHA) ..."

echo "=== $(date -u +%FT%TZ) BUILD ==="
cd "$XPLATFORM"
docker build \
  --platform linux/amd64 \
  -f services/pipeline/model-workers/shared/deploy/Dockerfile \
  --build-arg BASE_IMAGE="$BASE_IMG" \
  -t "$OUT_TAG" \
  services/
notify "[cttkw e2e] build done. pushing $OUT_TAG ..."

echo "=== $(date -u +%FT%TZ) PUSH ==="
docker push "$OUT_TAG"
notify "[cttkw e2e] image pushed: lia-vllm-chat-template-kwargs-$SHA. ready to deploy."

echo "=== $(date -u +%FT%TZ) DONE ==="

#!/usr/bin/env bash
# [cc-generated] Build & push xplatform workflow-engine image to training ECR from macOS.
#
# Usage: bash ~/build_workflow_engine_local.sh [BRANCH]
#   default BRANCH = lia/vllm-chat-template-kwargs
set -euo pipefail

BRANCH="${1:-lia/vllm-chat-template-kwargs}"
REPO_DIR="${REPO_DIR:-$HOME/xplatform}"

AWS_REGION="us-west-2"
AWS_PROFILE="training"
ECR_ACCOUNT_ID="476114115052"
ECR_REGISTRY="${ECR_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"
ECR_REPO="tl-data-training-pegasus-workflow-engine"

echo "════════════════════════════════════════════════════════"
echo "[$(date -Iseconds)] local workflow-engine build (branch=$BRANCH)"
echo "════════════════════════════════════════════════════════"

cd "$REPO_DIR"
git fetch origin "$BRANCH"
git checkout "$BRANCH"
git pull --ff-only origin "$BRANCH" 2>/dev/null || true

SHORT_SHA="$(git rev-parse --short=7 HEAD)"
BRANCH_SLUG="${BRANCH//\//-}"
TAG="${BRANCH_SLUG}-${SHORT_SHA}"
FULL_IMAGE="${ECR_REGISTRY}/${ECR_REPO}:${TAG}"

echo "[plan] tag=$TAG image=$FULL_IMAGE"

echo "[ecr] login"
aws ecr get-login-password --profile "$AWS_PROFILE" --region "$AWS_REGION" \
  | docker login --username AWS --password-stdin "$ECR_REGISTRY"

BUILDER="vllm-video-builder"
docker buildx use "$BUILDER" 2>/dev/null || docker buildx create --name "$BUILDER" --driver docker-container --use
docker buildx inspect --bootstrap >/dev/null

cd "$REPO_DIR/services"
START_TS=$(date +%s)
docker buildx build \
  --platform linux/amd64 \
  --file job-managing-plane/workflow-engine/deploy/Dockerfile \
  --tag "$FULL_IMAGE" \
  --provenance=false --sbom=false \
  --push \
  .
ELAPSED=$(( $(date +%s) - START_TS ))

echo "════════════════════════════════════════════════════════"
echo "[done $(date -Iseconds)] elapsed=${ELAPSED}s"
echo "Pushed: $FULL_IMAGE"
echo "════════════════════════════════════════════════════════"

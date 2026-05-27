#!/usr/bin/env bash
# [cc-generated] Build & push xplatform vllm-video worker to training ECR from local Mac (M-series).
#
# Usage: bash ~/build_vllm_video_local.sh [BRANCH]
#   default BRANCH = lia/vllm-chat-template-kwargs
set -euo pipefail

BRANCH="${1:-lia/vllm-chat-template-kwargs}"
REPO_DIR="${REPO_DIR:-$HOME/xplatform}"

AWS_REGION="us-west-2"
AWS_PROFILE="training"
ECR_ACCOUNT_ID="476114115052"
ECR_REGISTRY="${ECR_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"
ECR_REPO="tl-data-training-pegasus-vllm-video"
BASE_IMAGE="${ECR_REGISTRY}/tl-data-training-x-platform-model-workers:vllm-v0.19.0-20260411"

COLIMA_PROFILE="default"
COLIMA_CPU="${COLIMA_CPU:-4}"
COLIMA_MEM="${COLIMA_MEM:-8}"
COLIMA_DISK="${COLIMA_DISK:-80}"

echo "════════════════════════════════════════════════════════"
echo "[$(date -Iseconds)] local vllm-video build (branch=$BRANCH)"
echo "════════════════════════════════════════════════════════"

# ─── 1. Install colima + docker if missing ───────────────────
need_brew=()
command -v colima >/dev/null || need_brew+=(colima)
command -v docker >/dev/null || need_brew+=(docker)
docker buildx version >/dev/null 2>&1 || need_brew+=(docker-buildx)
if [ ${#need_brew[@]} -gt 0 ]; then
  echo "[install] brew install ${need_brew[*]}"
  brew install "${need_brew[@]}"
fi

# ─── 2. Start colima (vz + rosetta + x86_64) ─────────────────
if ! colima status "$COLIMA_PROFILE" >/dev/null 2>&1; then
  echo "[colima] starting profile=$COLIMA_PROFILE (vz, rosetta, x86_64, cpu=$COLIMA_CPU mem=${COLIMA_MEM}g disk=${COLIMA_DISK}g)"
  colima start --profile "$COLIMA_PROFILE" \
    --vm-type vz --vz-rosetta --arch x86_64 \
    --cpu "$COLIMA_CPU" --memory "$COLIMA_MEM" --disk "$COLIMA_DISK"
else
  echo "[colima] already running"
fi
docker context use colima >/dev/null 2>&1 || true
echo "[docker] server $(docker version --format '{{.Server.Version}}')"

# ─── 3. Ensure sparse-checkout has runtime-common ────────────
cd "$REPO_DIR"
if git sparse-checkout list 2>/dev/null | grep -q .; then
  if ! git sparse-checkout list 2>/dev/null | grep -qx 'services/shared-lib/python/runtime-common'; then
    echo "[sparse] adding services/shared-lib/python/runtime-common"
    git sparse-checkout add services/shared-lib/python/runtime-common
  fi
fi
git fetch origin "$BRANCH"
git checkout "$BRANCH"
git pull --ff-only origin "$BRANCH" 2>/dev/null || true

SHORT_SHA="$(git rev-parse --short=7 HEAD)"
BRANCH_SLUG="${BRANCH//\//-}"
TAG="${BRANCH_SLUG}-${SHORT_SHA}"
FULL_IMAGE="${ECR_REGISTRY}/${ECR_REPO}:${TAG}"

echo "[plan]"
echo "  branch:     $BRANCH"
echo "  short sha:  $SHORT_SHA"
echo "  tag:        $TAG"
echo "  full image: $FULL_IMAGE"
echo "  base image: $BASE_IMAGE"

# ─── 4. ECR login ────────────────────────────────────────────
echo "[ecr] login"
aws ecr get-login-password --profile "$AWS_PROFILE" --region "$AWS_REGION" \
  | docker login --username AWS --password-stdin "$ECR_REGISTRY"

# ─── 5. Buildx builder ──────────────────────────────────────
BUILDER="vllm-video-builder"
if ! docker buildx inspect "$BUILDER" >/dev/null 2>&1; then
  echo "[buildx] creating $BUILDER"
  docker buildx create --name "$BUILDER" --driver docker-container --use
else
  docker buildx use "$BUILDER"
fi
docker buildx inspect --bootstrap >/dev/null

# ─── 6. Build & push ────────────────────────────────────────
cd "$REPO_DIR/services"
echo "[build] starting (context=services/)"
START_TS=$(date +%s)
docker buildx build \
  --platform linux/amd64 \
  --build-arg "BASE_IMAGE=$BASE_IMAGE" \
  --file pipeline/model-workers/shared/deploy/Dockerfile \
  --tag "$FULL_IMAGE" \
  --provenance=false --sbom=false \
  --push \
  .
ELAPSED=$(( $(date +%s) - START_TS ))

echo ""
echo "════════════════════════════════════════════════════════"
echo "[done $(date -Iseconds)] elapsed=${ELAPSED}s"
echo "Pushed: $FULL_IMAGE"
echo "════════════════════════════════════════════════════════"

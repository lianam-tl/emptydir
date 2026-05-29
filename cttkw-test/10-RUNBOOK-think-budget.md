# thinking_token_budget PoC runbook

Verify that `--reasoning-parser qwen3` + per-request `thinking_token_budget` truncates the `<think>...</think>` block on the lia-test-27b worker.

## Built image

```
476114115052.dkr.ecr.us-west-2.amazonaws.com/tl-data-training-pegasus-vllm-video:lia-vllm-chat-template-kwargs-c003840
```

Built locally with `build_vllm_video_local.sh lia/vllm-chat-template-kwargs` on 2026-05-29.

## Steps

### 1) Deploy worker

Port-forward infracontroller, then POST the updated worker spec:

```bash
kubectl -n pegasus-platform port-forward svc/infracontroller 18092:8080 &
curl -X POST http://localhost:18092/model-deployments \
  -H 'Content-Type: application/json' \
  -d @09-infracontroller-llm-worker-thinkbudget.json
```

Changes vs `01-infracontroller-llm-worker.json`:
- image → new ECR tag (above)
- `EXTRA_ARGS` → adds `--reasoning-parser qwen3`

### 2) Find the kserve service for lia-test-27b

```bash
kubectl -n kserve-models get svc | grep lia-test-27b
# pick the *-predictor-default service (or whichever exposes port 80 → 8000)
```

### 3) Port-forward the worker

```bash
kubectl -n kserve-models port-forward svc/lia-test-27b-predictor-default 18093:80 &
# (substitute the actual service name from step 2)
```

### 4) Run the sweep

Two options.

**Smoke (synthetic random video, fastest):**

```bash
uv run --with numpy --with requests python 10-test-think-budget.py \
  --host http://localhost:18093 \
  --synthetic-shape 16,224,224
```

**Real video (matches PR-snippet Run 3 setup):**

Requires a pre-decoded UINT8 `[T,H,W,3]` npy. JOURNEY.md Run 3 used shape `[556, 224, 448, 3]` from `tl_gemini_sports_soccer_h16/00276801_9min_01687_02244.mp4`.

```bash
uv run --with numpy --with requests python 10-test-think-budget.py \
  --host http://localhost:18093 \
  --video-npy /path/to/soccer.npy \
  --video-fps 2.5 --start 0.0 --end 222.4
```

## Expected output

Three rows (budget=None, budget=10, budget=100). The compare block at the end:

```
=========== COMPARE ===========
budget=None        len=#### </think>=False completion=4096 finish=length    # baseline: no truncation, hits max_tokens
budget=10          len=  ## </think>=True  completion=  ~30 finish=stop      # PASS: forced </think> after 10 reasoning tokens
budget=100         len=  ## </think>=True  completion= ~150 finish=stop      # PASS: forced </think> after 100 reasoning tokens
```

If budget=10 still shows `</think>=False` and `completion=4096`, the reasoning parser isn't catching the prompt-prepended `<think>` (current PR pre-pends `<think>\n` rather than letting the model emit it). That would be the failure mode to investigate next.

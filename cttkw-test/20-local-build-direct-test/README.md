# Local build + direct chat-completions sweep playbook

Lightweight e2e for **vllm-video worker** only — skips wf-engine / orchestrator / eval-service entirely. Useful when you just changed worker code (e.g. `server.py`) and want a quick regression check without rebuilding the whole stack (wf-engine image takes ~30 min).

Used on 2026-06-02 to validate PR #80 fixup batch (`8428c34`): allowlist (Wade #1), normalize (sj #4), and chat-completions plumbing on the current branch tip.

## Why direct vLLM only

If your changes are limited to:
- `services/pipeline/model-workers/shared/model_worker/vllm_video/server.py`
- worker-side request validation, sampling params, output extraction

then a direct `/v1/chat/completions` smoke against the worker covers everything **without** rebuilding wf-engine / eval-service / orchestrator. The Python task code (`vllm_common.py`, etc.) talks to the worker over the same wire format — so you're testing the same surface that production hits.

Note: this won't cover:
- V2 binary `/v2/models/.../infer` path (raw V2 needs a different snippet)
- merge-side aggregation (`merge_chunk_segments.py`, `sme_merge.py`) — only covered by chunked pipelines
- caller-side plumbing (`call_vllm`, `call_vllm_text` etc.) — cover with unit tests

## 1. Build locally (colima linux/amd64)

```bash
# Prereq: colima running x86_64
colima status

# ECR login (training + dev)
aws ecr get-login-password --profile training --region us-west-2 \
  | docker login --username AWS --password-stdin 476114115052.dkr.ecr.us-west-2.amazonaws.com
aws ecr get-login-password --profile training --region us-west-2 \
  | docker login --username AWS --password-stdin 219219941196.dkr.ecr.us-west-2.amazonaws.com

# From xplatform root
SHA=$(git rev-parse --short=7 HEAD)
BASE_IMG=219219941196.dkr.ecr.us-west-2.amazonaws.com/x-platform-model-workers:vllm-v0.19.0-20260411
OUT_TAG=476114115052.dkr.ecr.us-west-2.amazonaws.com/tl-data-training-pegasus-vllm-video:lia-vllm-chat-template-kwargs-${SHA}

docker pull "$BASE_IMG"  # cached → no-op
docker build \
  --platform linux/amd64 \
  -f services/pipeline/model-workers/shared/deploy/Dockerfile \
  --build-arg BASE_IMAGE="$BASE_IMG" \
  -t "$OUT_TAG" \
  services/
docker push "$OUT_TAG"
```

Wall time on M1 Pro + cached base: **~90s end-to-end** (no real layer rebuild — Dockerfile only copies code + pip-installs a tiny tail of deps). Use `time docker build ...` to confirm.

The wrapper script `build.sh` runs all three steps and slack-pings on each milestone.

## 2. Deploy to lia-test-27b via InfraController

```bash
# port-forward (skip if already running)
kubectl -n pegasus-platform port-forward svc/infracontroller 18092:8080 &

# Submit deploy. Edit `image:` in deploy.json to your new sha first.
curl -X POST http://localhost:18092/model-deployments \
  -H 'Content-Type: application/json' \
  -d @deploy.json
```

Key fields in `deploy.json`:
- `image`: your fresh `lia-vllm-chat-template-kwargs-<sha>` tag
- `env.EXTRA_ARGS`: must include `--reasoning-parser qwen3` for thinking gates
- `env.VLLM_REASONING_CONFIG`: JSON `{"reasoning_start_str":"<think>","reasoning_end_str":"</think>"}`. Passed via env (not CLI flag) to dodge unquoted `${EXTRA_ARGS}` shell-expansion breakage.

Wait for pod ready (typically ~5 min for Qwen3-VL-32B):

```bash
kubectl get pod -n kserve-models \
  -l 'app.kubernetes.io/name=lia-test-27b,kserve.io/component=workload' \
  -w
```

Pod label is `app.kubernetes.io/name=lia-test-27b` (NOT `serving.kserve.io/inferenceservice=lia-test-27b` — that's a CRD-driven label that doesn't apply here).

## 3. Port-forward + run sweep

```bash
kubectl -n kserve-models port-forward svc/lia-test-27b-kserve-workload-svc 18093:8000 &
curl -s http://localhost:18093/v2/health/ready   # should be 200 once model loaded
python3 test_sweep.py --host http://localhost:18093 --out result.json
```

`test_sweep.py` runs 5 scenarios:

| Scenario | `chat_template_kwargs` | Expected |
|---|---|---|
| A_both | `{enable_thinking: T, supports_thinking: T}` | 200, `think_blocks` populated |
| B_enable_only | `{enable_thinking: T}` | **Same output as A** (normalize) |
| C_supports_only | `{supports_thinking: T}` | **Same output as A** (normalize) |
| D_reject_tokenize | `{tokenize: T}` (reserved kwarg) | **HTTP 400** (allowlist) |
| E_reject_tools | `{tools: "anything"}` (unknown kwarg) | **HTTP 400** (allowlist) |

With `temperature=0.0`, B and C should produce **byte-identical** text + think_blocks to A. Any drift means normalize isn't working.

## 4. Verify

`result.json` contains the per-scenario response. Eyeball the verdict block at end of `test_sweep.py` stdout. Sample passing run (PR #80 sha 8428c34):

```
[PASS] A_both: both flags → think_blocks populated
[PASS] B_enable_only: normalize: enable_thinking only → thinking still happens
[PASS] C_supports_only: normalize: supports_thinking only → thinking still happens
[PASS] D_reject_tokenize: allowlist: tokenize → 400
[PASS] E_reject_tools: allowlist: tools → 400
5/5 checks passed
```

Strong signal that B/C are truly normalized: their `text_head` and `think_blocks` head match A character-for-character.

D/E error body has the exact server message:

```
{"detail":"chat_template_kwargs contains unsupported keys: ['tokenize']. Supported: ['enable_thinking', 'supports_thinking']"}
```

## 5. Tear down

```bash
curl -X DELETE http://localhost:18092/model-deployments/lia-test-27b
# or leave it for the next iteration
```

## Files

| file | purpose |
|---|---|
| `build.sh` | Pull base → build → push, with slack notify on each step |
| `deploy.json` | InfraController submission body (update `image:` per sha) |
| `test_sweep.py` | 5-scenario sweep + automatic verdict |
| `result.json` | Captured result from the 2026-06-02 run on sha `8428c34` |

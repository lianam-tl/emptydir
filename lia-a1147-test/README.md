# A-1147: vllm-direct `n>1` end-to-end test harness

Reproducible test infrastructure for the xplatform vllm-video worker's
multi-rollout (`n>1`) support. Paired with branch
[`lia/A-1147-vllm-direct-n-rollouts`](https://github.com/twelvelabs-io/xplatform/tree/lia/A-1147-vllm-direct-n-rollouts)
and Linear [A-1147](https://linear.app/twelve-labs/issue/A-1147).

This README is intended as a complete drop-in: anyone who reads it should be
able to (1) deploy the worker, (2) verify `n>1` works against synthetic frames,
(3) drive it with a real video, and (4) iterate over a HuggingFace dataset.

> ⚠️ Paths assume the user is `long8v` and worktrees live under `~`. Adjust as needed.

---

## 1. What we found and fixed

The PR's first commit added `n` plumbing through:
- `vllm-direct.spec` (params_schema + `init_args`)
- `vllm_common.py` (`_build_vllm_transport_request` + `call_vllm`)
- `vllm_infer_direct.py` (task arg + `call_vllm(n=n)`)
- `vllm_video/server.py` (`GenerateRequest.n`, V2 parser, `SamplingParams(n=n)`,
  and `if len(output.outputs) > 1: text = json.dumps([o.text for o in ...])`)

**Smoke testing exposed a second bug**: vLLM v1 `AsyncLLM.generate()` defaults
to `RequestOutputKind.CUMULATIVE`. When `n>1`, `RequestOutputCollector.put()`
only merges multiple `RequestOutput`s if the consumer hasn't drained between
puts. The consumer drains on every `ready` event, so the awaited generator
yields **partial** `RequestOutput`s — each carrying only one `CompletionOutput`.
Our code then takes `output.outputs[0]` for the finished yield and never sees
the other n-1 completions.

Symptom: `n=4` requested, server returns a single string (the last completion).

Fix (commit `2b257d4` on the branch): when `request.n > 1`, set
`sp_kwargs["output_kind"] = RequestOutputKind.FINAL_ONLY` in
`_build_sampling_params_for_request`. The engine then coalesces all `n`
completions into a single final `RequestOutput`, the existing `if len(...) > 1`
branch fires, and the worker returns `json.dumps([...])`.

**Heads-up for SJ:** [pegasus PR #818](https://github.com/twelvelabs-io/pegasus/pull/818)'s
test `test_run_generate_serializes_multiple_outputs_when_n_gt_1` uses a
`_DummyOutput` mock that bypasses the AsyncLLM race entirely, so the test
passes on the unfixed code path. Real vLLM 0.19 needs the `FINAL_ONLY` override.

---

## 2. Prerequisites

| Tool | Purpose | Install |
|---|---|---|
| `colima` + `docker` + `docker buildx` | Build the worker image on macOS | `brew install colima docker docker-buildx` |
| `aws` CLI (SSO) | ECR login, S3 download via `boto3` | `brew install awscli` |
| `kubectl` | Talk to the training cluster | already configured |
| `ffmpeg` + `ffprobe` | Local video decode | `brew install ffmpeg` |
| Python 3.13 at `~/.venv/bin/python` | Run smoke + iterator scripts | `uv venv ~/.venv` |

Also required:

- `AWS_PROFILE=training` SSO session refreshed: `aws sso login --profile training`
- `HF_TOKEN` exported (or present in `/Users/long8v/pegasus/.env`)
- kubectl context pointing at the **training cluster** (`tl-data-training-cluster`)
- Python packages: `pandas`, `huggingface_hub`, `numpy`, `requests`, `boto3` (already in `~/.venv`)

---

## 3. End-to-end workflow

### Step 1 — build the worker image (one-time, ~25 min)

Uses your existing build script at `~/emptydir/build_vllm_video_local.sh`.
The `run_build.sh` wrapper in this folder adds Slack progress (`#fun-lia-trashcan`).

```bash
mkdir -p /tmp/a1147
nohup bash ~/emptydir/lia-a1147-test/run_build.sh > /tmp/a1147/wrapper.log 2>&1 &
# Slack will tell you when it's done. Final image URI lands in /tmp/a1147/image_uri.
```

> 💡 If your branch is already checked out in a worktree (recommended),
> `~/xplatform`'s `git checkout` of the same branch will fail. The wrapper
> handles this by pointing `REPO_DIR` at `~/xplatform-A-1147-n-rollouts`.

After the wrapper exits, you should see:
```bash
cat /tmp/a1147/image_uri
# 476114115052.dkr.ecr.us-west-2.amazonaws.com/tl-data-training-pegasus-vllm-video:lia-A-1147-vllm-direct-n-rollouts-<sha>
```

### Step 2 — deploy the pod

Two flavors:

**A) Pure image (no live patching):** `test_pod.yaml` deploys a Pod that runs
straight from the image you just built. Patch the image tag and apply.

```bash
IMG=$(cat /tmp/a1147/image_uri)
sed "s|PLACEHOLDER|${IMG#*:}|" ~/emptydir/lia-a1147-test/test_pod.yaml | kubectl apply -f -
```

**B) ConfigMap-mounted `server.py` (iterate without rebuild):** `test_pod_v2.yaml`
mounts a ConfigMap over `/app/.../server.py` so you can edit the local file,
update the ConfigMap, and redeploy in ~4 min instead of rebuilding (~25 min).

```bash
WORKTREE=~/xplatform-A-1147-n-rollouts
kubectl create configmap -n pegasus-platform lia-a1147-server-py \
  --from-file=server.py=$WORKTREE/services/pipeline/model-workers/shared/model_worker/vllm_video/server.py \
  --dry-run=client -o yaml | kubectl apply -f -

kubectl apply -f ~/emptydir/lia-a1147-test/test_pod_v2.yaml
```

> The pod has `restartPolicy: Never` and **no infracontroller labels**, so
> infracontroller doesn't touch it. Resources: 1 GPU, `b300-pegasus` nodepool.

### Step 3 — wait for ready (~3-5 min)

Qwen3.5-0.8B loads quickly via run:ai streamer.

```bash
bash ~/emptydir/lia-a1147-test/wait_ready.sh
# heartbeats Slack every 2 min, exits 0 when ready=true
```

### Step 4 — port-forward

```bash
kubectl port-forward -n pegasus-platform pod/lia-a1147-vllm-video-test 18000:8000 \
  > /tmp/a1147/pf.log 2>&1 &
echo "pf_pid=$!" > /tmp/a1147/pf.pid
sleep 3
curl -s http://localhost:18000/v2/health/ready  # {"ready":true}
```

> macOS may sleep the port-forward when the laptop lid closes. If subsequent
> requests fail with "Connection refused", just rerun this step.

### Step 5a — synthetic-frames smoke test

Fastest sanity check. No video download, just zeros as frames.

```bash
~/.venv/bin/python ~/emptydir/lia-a1147-test/smoke_test_a1147_n_rollouts.py \
  --url http://localhost:18000 \
  --n 4 --temperature 0.7 --max-tokens 64
# Expected: PASS: n=4 returned a JSON array of 4 strings
```

### Step 5b — real-video smoke test (single shot)

```bash
~/.venv/bin/python ~/emptydir/lia-a1147-test/smoke_test_a1147_n_rollouts.py \
  --url http://localhost:18000 \
  --video s3://tl-data-training-pegasus-us-west-2/raw_media/.../some.mp4 \
  --prompt "Describe what you see in this video." \
  --n 8 --temperature 0.7 --max-tokens 256 \
  --duration 30 --max-side 512 --fps 2.0
```

Flag notes:
- `--video` accepts `s3://...` (downloads via boto3 + cached at `/tmp/a1147_<basename>`) or a local path.
- `--max-side` rounds down to a multiple of **32**. Qwen3VL needs `H % 32 == 0` and `W % 32 == 0` (patch_size 16 × merge_size 2); otherwise the processor reshape fails.
- `--duration` clips to first N seconds (default 10).

### Step 5c — iterate an HF dataset

Loops `(video, prompt)` pairs from a HuggingFace dataset and writes
one JSON file per row to `--output-dir`. Resumable.

```bash
~/.venv/bin/python ~/emptydir/lia-a1147-test/iterate_a1147_dataset.py \
  --hf-dataset twelvelabs/tl_soccer_h16_sme_tdf \
  --hf-config H16_SOCCER \
  --hf-split train \
  --limit 5 \
  --n 4 --temperature 0.7 --max-tokens 256 \
  --duration 30 --max-side 384 --fps 2.0 \
  --output-dir /tmp/a1147_results
```

What it does per row:
1. Read `media[0].media_path` → video s3 URL
2. Read first `messages[0].content[i].text` (where `type=="text"`) → prompt
3. Skip if `<output-dir>/<row-id>.json` exists (resume support)
4. Download + ffmpeg-decode the video chunk
5. POST to `/v2/models/vllm-video/infer` with `n>1`
6. Parse the response, write `<output-dir>/<row-id>.json`

Output schema:
```json
{
  "id": "<row hash>",
  "row_index": 0,
  "video_url": "s3://...",
  "prompt": "Find every corner kick. ...",
  "n": 4,
  "temperature": 0.7,
  "max_tokens": 256,
  "decoded_frames": 60,
  "input_resolution": [384, 192],
  "decode_seconds": 0.43,
  "infer_seconds": 4.0,
  "result": {
    "completions": ["...", "...", "...", "..."],
    "video_frames": 60,
    "output_tokens": 72,
    "input_tokens": 2785,
    "finish_reason": "stop",
    "worker_elapsed_ms": 1451.5,
    "vllm_generate_ms": 1442.9,
    "retry_count": 0
  }
}
```

Resumability: re-running the same command skips any row whose `<id>.json`
exists. To force redo a row: `rm /tmp/a1147_results/<id>.json` first.

Failures are written to `<output-dir>/<row-id>.error.json` with traceback.

### Step 6 — teardown

```bash
kubectl delete pod -n pegasus-platform lia-a1147-vllm-video-test
kubectl delete configmap -n pegasus-platform lia-a1147-server-py
kill $(cat /tmp/a1147/pf.pid | cut -d= -f2) 2>/dev/null
```

The ECR image stays. Skip rebuild on the next iteration unless your branch advanced.

---

## 4. Iterating on `server.py` patches without a rebuild

Critical workflow for debugging the worker:

1. Edit `~/xplatform-A-1147-n-rollouts/services/pipeline/model-workers/shared/model_worker/vllm_video/server.py` locally.
2. Refresh the ConfigMap from the edited file:
   ```bash
   kubectl create configmap -n pegasus-platform lia-a1147-server-py \
     --from-file=server.py=$WORKTREE/services/pipeline/model-workers/shared/model_worker/vllm_video/server.py \
     --dry-run=client -o yaml | kubectl apply -f -
   ```
3. Recreate the pod (the existing one has the old file pinned by its mount):
   ```bash
   kubectl delete pod -n pegasus-platform lia-a1147-vllm-video-test
   kubectl apply -f ~/emptydir/lia-a1147-test/test_pod_v2.yaml
   bash ~/emptydir/lia-a1147-test/wait_ready.sh
   ```
4. Re-run port-forward + your test.

Total time per iteration: ~4 min model load + a few seconds for ConfigMap. Compare to ~25 min for a full image rebuild.

---

## 5. Files in this folder

| File | What it does |
|---|---|
| `README.md` | This document. |
| `run_build.sh` | Wraps the parent `build_vllm_video_local.sh` with Slack heartbeats. |
| `wait_ready.sh` | Polls pod readiness; Slack heartbeat every 2 min. |
| `test_pod.yaml` | Plain Pod (no ConfigMap mount). Image tag placeholder. |
| `test_pod_v2.yaml` | Pod with ConfigMap mount over `server.py`. **Use this for iteration.** |
| `smoke_test_a1147_n_rollouts.py` | Sends a single `(video, prompt, n)` to the worker and validates the response is a length-N JSON array. |
| `iterate_a1147_dataset.py` | Loops HF dataset rows → smoke test per row → JSON output per row. |
| `probe_pod.yaml` + `probe_script.py` | One-off standalone pod that runs `vllm.LLM.generate(prompt, SamplingParams(n=4))` offline. Used to prove vLLM 0.19 honors `n>1` independently. Keep for future similar debugging. |
| `raw_inspect.py` | Dumps the raw V2 binary response from the worker. Used to catch the UINT8-vs-BYTES decoder mistake. |

---

## 6. Gotchas we hit (read before debugging)

| Symptom | Root cause | Fix |
|---|---|---|
| `n=4` returns a single string | vLLM v1 `AsyncLLM` default `output_kind=CUMULATIVE` races with consumer drain | Set `output_kind=FINAL_ONLY` when `n>1` in `_build_sampling_params_for_request`. See PR commit `2b257d4`. |
| Smoke test sees `text` truncated by 4 chars | `text` response is a `UINT8` tensor (raw utf-8), not KServe `BYTES` (which would have a 4-byte length prefix) | In the smoke test, decode UINT8 chunks as `chunk.decode("utf-8")` directly. |
| Worker fails with `RuntimeError: shape '[...]' is invalid for input of size N` | Qwen3VL processor requires H and W divisible by `patch_size × merge_size = 16 × 2 = 32` | In the smoke test's `_load_video_to_numpy`, `out_w -= out_w % 32; out_h -= out_h % 32`. |
| `s5cmd cp ... ERROR: SharedCredsLoad: failed to get profile` | s5cmd doesn't read AWS SSO sessions | Use boto3 (`session = boto3.Session(profile_name='training')`). It honors SSO. |
| Iterator fails with "Token has expired and refresh failed" | AWS SSO token aged out | `aws sso login --profile training` in your terminal, then rerun. |
| `Failed to establish a new connection: [Errno 61] Connection refused` | macOS slept the kubectl port-forward | Rerun the port-forward step. |
| Image build aborts: `'<branch>' is already checked out at '<worktree>'` | Building from `~/xplatform` while the branch is checked out in a worktree | The wrapper script sets `REPO_DIR=<worktree path>` to avoid this. |

---

## 7. Future work / what this does NOT test

This harness only exercises the **worker** of the vllm-direct path. It does
**not** exercise:

- The `vllm-direct.spec` pipeline spec validation (covered by xplatform CI)
- The `vllm_infer_direct` task in the workflow-engine
- The `vllm_common.call_vllm` wrapper
- The `batch-request` service's deploy/run/teardown lifecycle
- S3 result upload from the workflow-engine

For a true end-to-end test of the PR, you want to:

1. Build a `workflow-engine` image from the same branch (or use the existing
   one if it imports `tasks/vllm_common` at runtime via the deployed package).
2. Either:
   - Submit a job via `POST /jobs` to the orchestrator (single job), or
   - Submit a batch via `POST /batch-runs` to the `batch-request` service
     (handles deploy + manifest + teardown around the same path).
3. Verify the per-request result JSON in the configured S3 output path
   contains a list of `n` strings (or however the final contract is defined).

A reasonable next test would be to submit one job via the orchestrator with
`pipeline_id=vllm-direct`, `params.n=8`, `url=<one of the soccer mp4s>`, and a
copy of the SME prompt as `params.prompt`, then inspect the orchestrator's
output URL.

---

## 8. Performance notes (Qwen3.5-0.8B baseline)

Pod resources: 1× H100 on `b300-pegasus`, tp=1, dp=1, `GPU_MEM=0.85`.

On a 30-second clip at fps=2 (60 frames, 384×192):
- Decode (ffmpeg, local): ~0.4 s
- Network upload + V2 binary protocol: <0.1 s
- vLLM inference (n=4): ~1.4 s
- vLLM inference (n=8): ~2.5 s

So you can do roughly **20-30 rows per minute** with this configuration. For
8947 rows: ~5-8 hours single-pod. A bigger model (Qwen3.5-27B) will be 5-10×
slower; consider parallelism (multiple pods, batch-request load-balancer) for
real production runs.

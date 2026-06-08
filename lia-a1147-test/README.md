# A-1147 vllm-direct n>1 smoke test artifacts

Test infrastructure used to verify the n>1 sampling fix for the xplatform
vllm-video worker. Paired with branch
[`lia/A-1147-vllm-direct-n-rollouts`](https://github.com/twelvelabs-io/xplatform/tree/lia/A-1147-vllm-direct-n-rollouts)
and Linear ticket [A-1147](https://linear.app/twelve-labs/issue/A-1147).

## Files

| File | What it does |
|---|---|
| `run_build.sh` | Wraps `~/emptydir/build_vllm_video_local.sh` with `#fun-lia-trashcan` Slack heartbeats every 5 min. Outputs final image URI to `/tmp/a1147/image_uri`. |
| `wait_ready.sh` | Polls a single pod until `Ready=true`; Slack heartbeat every 2 min. |
| `test_pod.yaml` | Base smoke-test pod manifest. Standalone Pod (no infracontroller labels), b300-pegasus, tp=1, Qwen3.5-0.8B. Image tag is a placeholder. |
| `test_pod_v2.yaml` | Same as `test_pod.yaml` but mounts a ConfigMap over `/app/.../server.py`. Lets you iterate on `server.py` patches **without rebuilding the image** — just `kubectl create configmap --from-file=...` and re-apply the pod. |
| `probe_pod.yaml` + `probe_script.py` | Standalone pod that runs `vllm.LLM(...).generate(prompt, SamplingParams(n=4))` offline. Used to prove vLLM 0.19 honors `n>1` for Qwen3.5-0.8B independently of the worker server. |
| `raw_inspect.py` | Sends one V2 binary request to a port-forwarded worker and dumps the raw response header + per-output bytes. Used to find that `text` is a UINT8 tensor (not BYTES) — the kind of mistake that's invisible if you only look at decoded strings. |

The smoke test client itself is `~/emptydir/smoke_test_a1147_n_rollouts.py`
(in the repo root, not this folder).

## End-to-end (assumes worktree at `~/xplatform-A-1147-n-rollouts`)

```bash
# 1. Build image (~25 min, Slack notifies completion)
mkdir -p /tmp/a1147
nohup bash ~/emptydir/lia-a1147-test/run_build.sh > /tmp/a1147/wrapper.log 2>&1 &

# 2. Deploy pod (~4 min model load)
IMG=$(cat /tmp/a1147/image_uri)
sed "s|PLACEHOLDER|${IMG#*:}|" ~/emptydir/lia-a1147-test/test_pod.yaml | kubectl apply -f -
bash ~/emptydir/lia-a1147-test/wait_ready.sh

# 3. Port-forward + smoke
kubectl port-forward -n pegasus-platform pod/lia-a1147-vllm-video-test 18000:8000 &
~/.venv/bin/python ~/emptydir/smoke_test_a1147_n_rollouts.py --url http://localhost:18000 --n 4 --temperature 0.7
```

### Iterating on `server.py` without a rebuild

```bash
# Edit your local server.py, then:
kubectl create configmap -n pegasus-platform lia-a1147-server-py \
  --from-file=server.py=$WORKTREE/services/pipeline/model-workers/shared/model_worker/vllm_video/server.py \
  --dry-run=client -o yaml | kubectl apply -f -
kubectl delete pod -n pegasus-platform lia-a1147-vllm-video-test
kubectl apply -f ~/emptydir/lia-a1147-test/test_pod_v2.yaml
bash ~/emptydir/lia-a1147-test/wait_ready.sh
```

## What this test found

vLLM v1's `AsyncLLM.generate()` defaults to `RequestOutputKind.CUMULATIVE`. When
`n>1`, the `RequestOutputCollector` races with the consumer — partial
`RequestOutput`s containing only one `CompletionOutput` each can be drained
between the engine's puts, silently losing n-1 outputs. Fix in the branch:
set `sp_kwargs["output_kind"] = RequestOutputKind.FINAL_ONLY` when `request.n > 1`
in `_build_sampling_params_for_request`.

Note for anyone porting: SJ's [pegasus PR #818](https://github.com/twelvelabs-io/pegasus/pull/818)
uses a `_DummyOutput` mock in its test that bypasses the AsyncLLM race entirely,
so its test would pass on the unfixed code path. Real vLLM 0.19 will drop
outputs without the `FINAL_ONLY` override.

## Caveats / paths assumed

- Slack helpers source `SLACK_BOT_TOKEN` from `/Users/long8v/pegasus/.env`. Channel: `#fun-lia-trashcan`.
- `run_build.sh` assumes the worktree is at `/Users/long8v/xplatform-A-1147-n-rollouts` (otherwise the parent `~/xplatform` checkout collides with the worktree's checkout of the same branch).
- All k8s objects assume kubeconfig points at the **training cluster** (`tl-data-training-cluster`).
- AWS profile `training` is used for ECR login.

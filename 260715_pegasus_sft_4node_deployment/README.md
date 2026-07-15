# Pegasus SFT: 4-node deployment

This deployment keeps the existing `pegasus-15-sft` worker unchanged and
adds `pegasus-sft-4node` on four `b300-ens` nodes.

- 4 replicas on any available `b300-ens` nodes
- 8 GPUs per replica, 32 GPUs total
- `TP=2`, `DP=4` per replica
- concurrency 4 per replica
- checkpoint: `checkpoint-1000-safetensors`
- v2 generation default: 128,000 tokens
- v2 structured-output string cap: 102,000 characters

Each replica requests all 8 GPUs on a B300 node, so Kubernetes necessarily
places the four replicas on four distinct nodes.

Register `b300-ens` if it is not already present, register the model, and mark
it active:

```bash
kubectl port-forward -n pegasus-platform svc/speccenter 18080:8080

curl -L -X POST http://127.0.0.1:18080/nodegroups \
  -H 'Content-Type: application/json' \
  --data-binary @node_group_b300_ens.json

curl -L -X POST http://127.0.0.1:18080/models \
  -H 'Content-Type: application/json' \
  --data-binary @model_register.json

curl -L -X PATCH http://127.0.0.1:18080/status \
  -H 'Content-Type: application/json' \
  --data-binary @status_active.json
```

To stop it, patch `status_inactive.json`. Deactivation removes the generated
deployment and its pods.

`model_register.json` now points v1 at the branch-specific 128K/102K image.
The v2 payloads preserve the attempted versioned rollout configuration.

## Operational Notes From 2026-07-15

### GPU topology

- `gpu_count: 8`, `TP=2`, and `DP=4` consume one complete B300 node.
- Four replicas therefore consume 32 GPUs on four distinct nodes without
  explicit node-name pinning.
- Keep model concurrency and
  `PEGASUS_VLLM_VIDEO_REQUEST_SEMAPHORE_COUNT` at 4 to expose all four DP
  engines per replica.

### Output limits live in different layers

- `pegasus15-sme` owns the normal pipeline request default for `max_tokens`.
- The vLLM worker owns the fallback used when a direct request omits
  `max_tokens`.
- The worker also recursively adds or lowers JSON Schema string `maxLength`.
  This is a guided-decoding constraint, not the `postprocess_segments` task.
- `postprocess_segments` has no character truncation guard; it handles
  deduplication, overlaps, short-segment absorption, and duration warnings.
- Raising output tokens reduces the input space remaining inside the model's
  262,144-token context. A 128,000-token output reservation leaves roughly
  134,000 tokens for video, ASR, prompt, schema, and expansion reserve.

The branch-specific image contains:

```text
GENERATION_MAX_TOKENS_DEFAULT = 128000
STRING_SCHEMA_MAX_LENGTH = 102000
```

Image:

```text
476114115052.dkr.ecr.us-west-2.amazonaws.com/tl-data-training-pegasus-vllm-video:lia-128k-102k-eaa7c96@sha256:521400b4881ae44c769c9f39d845c7a0924a6b1d0c53da5f0f312b823729fef3
```

### Building a training worker image

The `BUILD` workflow needs a digest-pinned training base image. The successful
build used:

```text
476114115052.dkr.ecr.us-west-2.amazonaws.com/tl-data-training-x-platform-model-workers:vllm-v0.19.0-20260411@sha256:c4bca9570657a0b75284fbaf725740a206fb0c30c4d233402bb4c0ffb4f8d5cc
```

Use `service_group=x-platform-vllm-video-worker` and
`deploy_model_worker=false` for a custom legacy SpecCenter worker. The build
uses a unique tag in the shared training ECR repository and does not alter a
live deployment by itself.

Build run:
https://github.com/twelvelabs-io/xplatform/actions/runs/29393971161

### SpecCenter and infra-controller behavior

- Registering a new pipeline spec is not enough; its version must also be
  activated through `PATCH /pipelines/status`.
- The 128K experiment registered `pegasus15-sme` `1.0.11`, then restored the
  live shared pipeline to `1.0.4` with its 32,000-token default during teardown.
- The legacy infra-controller was OOM-killed at its 1 GiB limit while
  reconciling 141 active models. SpecCenter accepted state changes, but their
  Kubernetes deployment events were delayed until a controller restart.
- Before assuming a model activation failed, check both SpecCenter active
  state and `kubectl get pods`; they can temporarily disagree when the
  infra-controller is unhealthy.

### Final teardown state

- `pegasus-sft-4node` v2 is inactive.
- No `pegasus-sft-4node` deployments or pods remain.
- The shared `pegasus15-sme` pipeline is back on version `1.0.4` with
  `max_tokens=32000`.

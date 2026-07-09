# H0 Entity Augment CK2200 Pegasus15-SME vLLM Worker

This folder contains the payloads for deploying this checkpoint as a persistent Pegasus15-SME vLLM worker:

```text
worker_type: h0-entity-augment-ck2200
model_id: h0-entity-augment-ck2200
version: v1
deployment: model-h0-entity-augment-ck2200-v1
node_pool: b300-pegasus
gpu: 2
TP: 2
MODEL: s3://tl-data-training-pegasus-us-west-2/checkpoints/jeongyeon-nam/consol_260416_clean_filter_less_aug_highres_h0_mn_sme_2x_lr2e-6_qwen3_5_27b-base/checkpoint-2200-safetensors/
```

This uses the same `vllm-video` worker image as the previous Pegasus15-SME persistent worker, but registers a different worker identity and loads a different checkpoint.

## What To Call

Use orchestrator `/jobs` with:

```json
{
  "pipelineId": "pegasus15-sme",
  "params": {
    "worker_type": "h0-entity-augment-ck2200",
    "worker_call_mode": "loadbalancer",
    "audio_worker_call_mode": "loadbalancer"
  }
}
```

`pipelineId` selects the Pegasus15-SME pipeline. `worker_type` selects this deployed checkpoint.

## Deploy

Open two port-forwards:

```bash
kubectl port-forward -n pegasus-platform svc/speccenter 18080:8080
kubectl port-forward -n pegasus-platform svc/infracontroller 18081:8080
```

Register the model:

```bash
curl -L -X POST \
  http://127.0.0.1:18080/models \
  -H 'Content-Type: application/json' \
  --data-binary @legacy_model_register.json
```

Mark it active:

```bash
curl -L -X PATCH \
  http://127.0.0.1:18080/status \
  -H 'Content-Type: application/json' \
  --data-binary @status_active.json
```

Scale to one replica:

```bash
curl -L -X PATCH \
  'http://127.0.0.1:18081/deployments/h0-entity-augment-ck2200/scale?version=v1' \
  -H 'Content-Type: application/json' \
  --data-binary @scale_one.json
```

Wait for readiness:

```bash
kubectl get pods -n pegasus-platform -o wide | rg 'model-h0-entity-augment-ck2200'
```

Expected:

```text
model-h0-entity-augment-ck2200-v1-...   2/2   Running
```

## Smoke Test

Open an orchestrator port-forward:

```bash
kubectl port-forward -n pegasus-platform svc/orchestrator 18082:8080
```

Submit:

```bash
curl -L -s -X POST \
  http://127.0.0.1:18082/jobs \
  -H 'Content-Type: application/json' \
  --data-binary @orchestrator_smoke_job.json
```

Poll:

```bash
curl -L -s http://127.0.0.1:18082/jobs/{jobId}
```

Fetch output after completion:

```bash
curl -L -s http://127.0.0.1:18082/jobs/{jobId}/output
```

## Difference From `vllm-video-b300`

Both are `vllm-video` servers. The difference is which worker name the pipeline routes to and which checkpoint is loaded:

| Worker | Checkpoint | Purpose |
| --- | --- | --- |
| `vllm-video-b300` | shared/default model | Generic shared vLLM video worker |
| `lia-soccer-mtp-ck2000-persistent` | soccer MTP checkpoint-2000 | Previous persistent Pegasus15-SME worker |
| `h0-entity-augment-ck2200` | H0 entity augment checkpoint-2200 | This persistent Pegasus15-SME vLLM worker |

If a `/jobs` payload uses `worker_type: vllm-video-b300`, it will not call this worker. To call this checkpoint, use `worker_type: h0-entity-augment-ck2200`.

## Manual Teardown

Mark inactive:

```bash
curl -L -X PATCH \
  http://127.0.0.1:18080/status \
  -H 'Content-Type: application/json' \
  --data '{"id":"h0-entity-augment-ck2200","version":"v1","status":"inactive","reason":"manual teardown"}'
```

Scale to zero:

```bash
curl -L -X PATCH \
  'http://127.0.0.1:18081/deployments/h0-entity-augment-ck2200/scale?version=v1' \
  -H 'Content-Type: application/json' \
  --data '{"replicas":0}'
```

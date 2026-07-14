# Pegasus15 SME Persistent Worker

This folder contains the JSON payloads used to keep one `vllm-video` worker alive for the Pegasus15 SME checkpoint:

```text
worker_type: lia-soccer-mtp-ck2000-persistent
model_id: lia-soccer-mtp-ck2000-persistent
version: v1
deployment: model-lia-soccer-mtp-ck2000-persistent-v1
node_pool: b300-pegasus
gpu: 2
TP: 2
```

This is **one worker pod using two GPUs**, not two separate worker pods.

## Deployment Flow

There are three separate concepts:

```text
pipelineId   = pegasus15-sme
worker_type  = lia-soccer-mtp-ck2000-persistent
MODEL        = s3://.../checkpoint-2000-safetensors/
```

- `pipelineId` is the pipeline recipe. It decides which workflow logic runs.
- `worker_type` is the model worker name. It decides which live model server the pipeline calls.
- `MODEL` is the checkpoint path loaded inside that model worker.

Deploying this worker means:

1. Register a model record in spec-center with `legacy_model_register.json`.
2. Mark that model record active with `status_active.json`.
3. Ask infra-controller to scale that model to one replica with `scale_one.json`.
4. Wait until Kubernetes shows `model-lia-soccer-mtp-ck2000-persistent-v1-...` as `2/2 Running`.
5. Submit jobs through orchestrator `/jobs` with `pipelineId: pegasus15-sme` and `worker_type: lia-soccer-mtp-ck2000-persistent`.

In beginner terms:

- spec-center is the registry/address book.
- infra-controller starts or stops the actual Kubernetes model pod.
- orchestrator `/jobs` is the API entrypoint users call.
- workflow-engine later picks up the accepted job and runs the pipeline.
- the model pod is only one part of the pipeline; it is not the whole pipeline by itself.

## Difference From Generic vLLM Video

This deployment still uses the `vllm-video` worker image internally:

```text
476114115052.dkr.ecr.us-west-2.amazonaws.com/tl-data-training-pegasus-vllm-video:main-fb19772
```

The difference is the registered worker identity and the checkpoint it loads.

| Case | `worker_type` | What it calls | Lifecycle |
| --- | --- | --- | --- |
| Shared/default vLLM video | `vllm-video-b300` | Existing shared/default vLLM video worker | Managed separately by platform |
| This deployment | `lia-soccer-mtp-ck2000-persistent` | Our checkpoint-2000 SME worker | Stays alive until manual teardown |
| Batch-request model deployment | Generated per batch run | Temporary worker for that batch | Batch-request tears it down when done |

So this is not a new pipeline. It is a custom persistent `vllm-video` worker that the existing `pegasus15-sme` pipeline can route to.

To use the deployed checkpoint, the important request field is:

```json
"worker_type": "lia-soccer-mtp-ck2000-persistent"
```

If this is changed back to:

```json
"worker_type": "vllm-video-b300"
```

then the request uses the shared/default vLLM video worker instead of this checkpoint.

## Message For Owen

Send this if someone needs to call the deployed Pegasus15 SME API:

```text
Pegasus15-SME API is called through orchestrator /jobs.

Model worker:
- pipelineId: pegasus15-sme
- worker_type: lia-soccer-mtp-ck2000-persistent
- worker_call_mode: loadbalancer
- audio_worker_call_mode: loadbalancer

If calling from a laptop, first port-forward orchestrator:

kubectl port-forward -n pegasus-platform svc/orchestrator 18082:8080

Then submit a job:

curl -L -s -X POST http://127.0.0.1:18082/jobs \
  -H 'Content-Type: application/json' \
  --data-binary @orchestrator_smoke_job.json

Example payload:
orchestrator_smoke_job.json in this same repo directory

Poll status:

curl -L -s http://127.0.0.1:18082/jobs/{jobId}

When status becomes completed, fetch output:

curl -L -s http://127.0.0.1:18082/jobs/{jobId}/output

Important: this is not batch-request. Batch-request currently creates and tears down its own model deployment, so for the persistent Pegasus15 SME worker, use /jobs directly.

If the job stays JOB_STATUS_PENDING, the request was accepted but workflow-engine has not started it yet. This usually means workflow-engine is saturated or one of its admission gates is closed.
```

## Important

`batch-request` currently creates its own temporary model deployment for each `/batch-runs` request and tears it down at the end. So a new call to:

```bash
POST http://xplatform-training.twelve.labs/batch-request/batch-runs
```

will **not automatically reuse** this persistent worker.

To use this persistent worker, submit jobs directly to orchestrator and set:

```json
"worker_type": "lia-soccer-mtp-ck2000-persistent",
"worker_call_mode": "loadbalancer",
"audio_worker_call_mode": "loadbalancer"
```

## Files

```text
legacy_model_register.json  # Register the persistent legacy model in spec-center.
legacy_model_register_node2.json # Register the second node-pinned model.
status_active.json          # Mark the model active.
status_active_node2.json    # Mark the second node-pinned model active.
scale_one.json              # Scale the deployment to 1 replica.
orchestrator_smoke_job.json # Example pegasus15-sme job routed to this worker.
```

## Check Current Worker

```bash
kubectl get pods -n pegasus-platform -o wide | rg 'model-lia-soccer-mtp-ck2000-persistent'
```

Expected when alive:

```text
model-lia-soccer-mtp-ck2000-persistent-v1-...   2/2   Running
```

## Submit A Request

Open a port-forward to orchestrator:

```bash
kubectl port-forward -n pegasus-platform svc/orchestrator 18082:8080
```

From this directory, submit the smoke job:

```bash
curl -L -X POST \
  http://127.0.0.1:18082/jobs \
  -H 'Content-Type: application/json' \
  --data-binary @orchestrator_smoke_job.json
```

Response shape:

```json
{"jobId":"<job-id>"}
```

Poll the job:

```bash
curl -L http://127.0.0.1:18082/jobs/<job-id>
```

If it completes, the response will include an `outputUrl` like:

```text
s3://tl-data-training-pegasus-us-west-2/sme-2pass-postprocess/<hash>.json
```

Read the output:

```bash
aws s3 cp --profile training \
  s3://tl-data-training-pegasus-us-west-2/sme-2pass-postprocess/<hash>.json -
```

## Recreate The Persistent Worker

Only do this if the worker was torn down or disappeared.

### Two-node variant

To guarantee two independent `TP=2` workers on different nodes, register and
activate both `legacy_model_register.json` and
`legacy_model_register_node2.json`. Scale each model ID to one replica with
`scale_one.json`. Both records advertise the same
`worker_type: lia-soccer-mtp-ck2000-persistent`, so loadbalancer calls use them
as one worker pool. Each worker uses two GPUs, for four GPUs total.

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
  'http://127.0.0.1:18081/deployments/lia-soccer-mtp-ck2000-persistent/scale?version=v1' \
  -H 'Content-Type: application/json' \
  --data-binary @scale_one.json
```

Then wait until the pod becomes `2/2 Running`:

```bash
kubectl get pods -n pegasus-platform -o wide | rg 'model-lia-soccer-mtp-ck2000-persistent'
```

Startup can take several minutes because the worker loads the checkpoint, loads the MTP drafter, compiles vLLM graphs, and warms CUDA graphs.

## Tear Down Manually

Open port-forwards:

```bash
kubectl port-forward -n pegasus-platform svc/speccenter 18080:8080
kubectl port-forward -n pegasus-platform svc/infracontroller 18081:8080
```

Mark the model inactive:

```bash
curl -L -X PATCH \
  http://127.0.0.1:18080/status \
  -H 'Content-Type: application/json' \
  --data '{"id":"lia-soccer-mtp-ck2000-persistent","version":"v1","status":"inactive","reason":"manual teardown"}'
```

Scale to zero:

```bash
curl -L -X PATCH \
  'http://127.0.0.1:18081/deployments/lia-soccer-mtp-ck2000-persistent/scale?version=v1' \
  -H 'Content-Type: application/json' \
  --data '{"replicas":0}'
```

Verify it is gone or no longer running:

```bash
kubectl get pods -n pegasus-platform -o wide | rg 'model-lia-soccer-mtp-ck2000-persistent'
```

## Notes

- The sample video currently looks like baseball even though the prompt says soccer.
- If a submitted job stays `JOB_STATUS_PENDING`, check workflow-engine capacity/rollout first:

```bash
curl -L http://127.0.0.1:18082/jobs/stats
kubectl get pods -n pegasus-platform -o wide | rg 'workflow-workflow-engine-v1-pegasus'
```

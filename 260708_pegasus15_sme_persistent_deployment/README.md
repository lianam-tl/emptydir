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
status_active.json          # Mark the model active.
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


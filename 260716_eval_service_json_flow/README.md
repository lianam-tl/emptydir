# Eval-Service JSON Submission Flow

This note explains how this payload is actually submitted:

```json
{
  "name": "entity-cov-v3-chunk05m",
  "dataset": "twelvelabs/entity_cov_v0_tdf",
  "config": "chunk_05m",
  "split": "test",
  "maxSamples": 1,
  "pipelineId": "vllm-direct",
  "workerType": "vllm-video-v1",
  "modelPath": "s3://tl-data-training-pegasus-us-west-2/releases/Pegasus1.5-2604/",
  "nodePool": "b300-pegasus",
  "minReplicas": 1,
  "maxReplicas": 1,
  "concurrency": 1,
  "tp": 1,
  "dp": 1,
  "maxTokens": 16384,
  "temperature": 0
}
```

Dataset link: https://huggingface.co/datasets/twelvelabs/entity_cov_v0_tdf

## TL;DR

You send this JSON to eval-service with:

```bash
HOST=http://xplatform-training.twelve.labs
curl -sS -X POST "${HOST}/sme-studio/api/eval/runs" \
  -H "Content-Type: application/json" \
  --data-binary @payload.json | jq
```

eval-service returns an `evalRun.id`. Use that id to check status:

```bash
RUN_ID="<evalRun.id from POST response>"
curl -sS "${HOST}/sme-studio/api/eval/runs/${RUN_ID}" | jq
```

## Copy-paste payload file

```bash
cat > /tmp/entity-cov-v3-chunk05m.json <<'JSON'
{
  "name": "entity-cov-v3-chunk05m",
  "dataset": "twelvelabs/entity_cov_v0_tdf",
  "config": "chunk_05m",
  "split": "test",
  "maxSamples": 1,
  "pipelineId": "vllm-direct",
  "workerType": "vllm-video-v1",
  "modelPath": "s3://tl-data-training-pegasus-us-west-2/releases/Pegasus1.5-2604/",
  "nodePool": "b300-pegasus",
  "minReplicas": 1,
  "maxReplicas": 1,
  "concurrency": 1,
  "tp": 1,
  "dp": 1,
  "maxTokens": 16384,
  "temperature": 0
}
JSON

HOST=http://xplatform-training.twelve.labs
RUN_ID="$(
  curl -sS -X POST "${HOST}/sme-studio/api/eval/runs" \
    -H "Content-Type: application/json" \
    --data-binary @/tmp/entity-cov-v3-chunk05m.json \
    | jq -r '.evalRun.id'
)"

echo "RUN_ID=${RUN_ID}"
curl -sS "${HOST}/sme-studio/api/eval/runs/${RUN_ID}" | jq
```

## What happens inside

1. Your `curl` posts the JSON to:

```text
POST http://xplatform-training.twelve.labs/sme-studio/api/eval/runs
```

2. `eval-service` validates the JSON with `CreateRunRequest`.

Important field mapping:

| JSON field | Meaning |
| --- | --- |
| `dataset`, `config`, `split`, `maxSamples` | Which Hugging Face dataset rows to load. Here it loads at most 1 sample from `twelvelabs/entity_cov_v0_tdf`, config `chunk_05m`, split `test`. |
| `pipelineId` | Which x-platform/orchestrator pipeline to run. Here: `vllm-direct`. |
| `workerType` | Worker identity used by the model deployment config. Here: `vllm-video-v1`. |
| `modelPath` | S3 model/release path used by the deployed model worker. |
| `nodePool`, `minReplicas`, `maxReplicas`, `concurrency`, `tp`, `dp` | GPU scheduling and model serving capacity knobs. |
| `maxTokens`, `temperature` | Generation parameters passed into each inference request. |

3. eval-service loads the dataset rows and turns each eval item into an `EvalTask`.

For this payload, `maxSamples: 1` usually means one dataset row is loaded. If that row contains multiple eval metadata entries, eval-service may create more than one task from that one row.

4. In the normal current path, eval-service creates Eval V3 DB rows, then publishes a RabbitMQ planner wake-up message.

The immediate POST response is only:

```json
{
  "evalRun": {
    "id": "<run_id>",
    "status": "pending"
  }
}
```

At this point, GPU inference has not necessarily started yet. The request is recorded and queued.

5. The Eval V3 planner builds the x-platform BatchRequest payload.

The BatchRequest payload is not identical to your original JSON. It is compiled into roughly this shape:

```json
{
  "name": "entity-cov-v3-chunk05m-<run_id_prefix>",
  "model_deployment_template": {
    "deploymentMode": "legacy",
    "image": "<eval-service env image or imageUrl override>",
    "gpu": 1,
    "port": 8000,
    "modelPath": "s3://tl-data-training-pegasus-us-west-2/releases/Pegasus1.5-2604/",
    "env": {
      "TP": "1",
      "DP": "1"
    },
    "concurrency": 1
  },
  "job_requests": [
    {
      "request_id": "<stable request id>",
      "job_request": {
        "url": "<video url from TDF row>",
        "pipeline_id": "vllm-direct",
        "params": {
          "segment_definition": "<query/user prompt from TDF metadata>",
          "metadata_schema": "<optional metadata schema>",
          "max_tokens": 16384,
          "temperature": 0.0
        }
      }
    }
  ],
  "capacity_policy": {
    "node_pool": "b300-pegasus",
    "min_replicas": 1,
    "max_replicas": 1,
    "placement_policy": "compact_hard_pinning"
  },
  "execution_policy": {
    "max_in_flight": 1
  },
  "lifecycle_policy": {
    "readiness": {
      "mode": "partial",
      "min_ready_replicas": 1
    },
    "teardown": {
      "policy": "immediate"
    }
  },
  "metadata": {
    "source": "eval-v3-worker",
    "eval_v3_eval_run_id": "<run_id>",
    "eval_v3_idempotency_key": "<stable batch idempotency key>",
    "prediction_unit_id": "prediction",
    "dataset": "twelvelabs/entity_cov_v0_tdf"
  }
}
```

6. The Eval V3 submitter sends that compiled payload to BatchRequest:

```text
POST ${BATCH_REQUEST_BASE_URL}/batch-runs
```

In training-prod this is exposed publicly as:

```text
http://xplatform-training.twelve.labs/batch-request/batch-runs
```

But for normal evals, do not post there directly. Post to eval-service first, because eval-service creates the Eval V3 run, tasks, output rows, scoring flow, and leaderboard publishing state.

7. BatchRequest deploys or reuses the model worker, sends each `job_request` to the selected pipeline, and writes the prediction outputs.

8. Eval V3 watcher, ingester, scorer, and publisher workers pick up the completed BatchRequest output, store predictions/metrics, and publish results.

## How to inspect the underlying BatchRequest

After the eval run starts, the eval run detail may expose `batchId`:

```bash
BATCH_ID="$(
  curl -sS "${HOST}/sme-studio/api/eval/runs/${RUN_ID}" \
    | jq -r '.evalRun.batchId // empty'
)"

echo "BATCH_ID=${BATCH_ID}"
curl -sS "${HOST}/batch-request/batch-runs/${BATCH_ID}" | jq
curl -sS "${HOST}/batch-request/batch-runs/${BATCH_ID}/settings" | jq
```

If `batchId` is empty, the run may still be in the Eval V3 planning/submitting stage.

## Notes and gotchas

- `name` is a human-readable eval run name. Add `batchNamePrefix` if you want the BatchRequest UI row to have an easy prefix too.
- `maxSamples: 1` is good for smoke testing. Remove it or set it to `null` for a full run.
- `concurrency` is per deployed worker capacity. `maxInFlight` can separately cap total in-flight requests; when omitted, the planner defaults conservatively.
- `tp * dp` becomes the GPU count in the model deployment template. Here `1 * 1 = 1 GPU`.
- `temperature: 0` makes generation deterministic as much as the serving stack allows.
- `includeAsrData` defaults to `true` in the eval-service request schema. In the older BatchRequest compiler path, existing ASR data from the TDF sample can be forwarded as `params.asr_data`. In the current Eval V3 planner code I inspected, the compiled `job_request.params` includes segment definition, metadata schema, max tokens, and temperature; I did not see `asr_data` wired there. So if ASR is critical for this exact path, verify the deployed eval-service version or inspect the BatchRequest settings for the submitted run.

## Source files checked

- `eval/eval-service/eval_service/api/routes/runs.py`
- `eval/eval-service/eval_service/api/models.py`
- `eval/eval-v3/eval_v3/control_plane.py`
- `eval/eval-v3/eval_v3/execution/planning_payloads.py`
- `eval/eval-v3/eval_v3/execution/batch_request.py`
- `eval/eval-v3/eval_v3/execution/submitter/handler.py`
- `eval/eval-service/docs/eval-quickstart.md`

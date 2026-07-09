# Pegasus15 Named Persistent Workers

This folder contains deployment payloads for two persistent Pegasus15-SME vLLM workers.

## Workers

| Worker | S3 checkpoint | GPUs |
| --- | --- | --- |
| `pegasus-15` | `s3://tl-data-training-pegasus-us-west-2/checkpoints/kian-kim/soce_rl_260516_all_pair_p15_a1mckpt1000s60_w7030/` | 2 |
| `pegasus-15-sft` | `s3://tl-data-training-pegasus-us-west-2/checkpoints/jeongyeon-nam/ablation_260416_soccer_clean_filter_low_aug-highres_lr2e-6_qwen3_5_27b_soccer_dc_sme_low_filter_mtp_0513-base/checkpoint-1000-safetensors/` | 2 |

Both use:

```text
pipelineId: pegasus15-sme
image: 476114115052.dkr.ecr.us-west-2.amazonaws.com/tl-data-training-pegasus-vllm-video:main-fb19772
node_group_id: b300-pegasus
TP: 2
DP: 1
```

These payloads do not hard-pin `node_names`. Kubernetes/Kueue can choose available B300 capacity.

## Call Through `/jobs`

Use the orchestrator `/jobs` API. The main routing field is `params.worker_type`.

For the RL checkpoint:

```json
{
  "pipelineId": "pegasus15-sme",
  "params": {
    "worker_type": "pegasus-15",
    "worker_call_mode": "loadbalancer",
    "audio_worker_call_mode": "loadbalancer"
  }
}
```

For the SFT checkpoint:

```json
{
  "pipelineId": "pegasus15-sme",
  "params": {
    "worker_type": "pegasus-15-sft",
    "worker_call_mode": "loadbalancer",
    "audio_worker_call_mode": "loadbalancer"
  }
}
```

Example smoke payloads:

- `pegasus_15_smoke_job.json`
- `pegasus_15_sft_smoke_job.json`

## Deploy Commands

Open two port-forwards:

```bash
kubectl port-forward -n pegasus-platform svc/speccenter 18080:8080
kubectl port-forward -n pegasus-platform svc/infracontroller 18081:8080
```

Register and activate `pegasus-15`:

```bash
curl -L -X POST http://127.0.0.1:18080/models \
  -H 'Content-Type: application/json' \
  --data-binary @pegasus_15_model_register.json

curl -L -X PATCH http://127.0.0.1:18080/status \
  -H 'Content-Type: application/json' \
  --data-binary @pegasus_15_status_active.json

curl -L -X PATCH \
  'http://127.0.0.1:18081/deployments/pegasus-15/scale?version=v1' \
  -H 'Content-Type: application/json' \
  --data-binary @scale_one.json
```

Register and activate `pegasus-15-sft`:

```bash
curl -L -X POST http://127.0.0.1:18080/models \
  -H 'Content-Type: application/json' \
  --data-binary @pegasus_15_sft_model_register.json

curl -L -X PATCH http://127.0.0.1:18080/status \
  -H 'Content-Type: application/json' \
  --data-binary @pegasus_15_sft_status_active.json

curl -L -X PATCH \
  'http://127.0.0.1:18081/deployments/pegasus-15-sft/scale?version=v1' \
  -H 'Content-Type: application/json' \
  --data-binary @scale_one.json
```

Check readiness:

```bash
kubectl get pods -n pegasus-platform -o wide | rg 'model-pegasus-15|model-pegasus-15-sft'
```

Expected:

```text
model-pegasus-15-v1-...       2/2   Running
model-pegasus-15-sft-v1-...   2/2   Running
```

## Manual Teardown

Scale to zero:

```bash
curl -L -X PATCH \
  'http://127.0.0.1:18081/deployments/pegasus-15/scale?version=v1' \
  -H 'Content-Type: application/json' \
  --data '{"replicas":0}'

curl -L -X PATCH \
  'http://127.0.0.1:18081/deployments/pegasus-15-sft/scale?version=v1' \
  -H 'Content-Type: application/json' \
  --data '{"replicas":0}'
```

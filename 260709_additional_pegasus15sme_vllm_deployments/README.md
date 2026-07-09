# Additional Pegasus15-SME vLLM Persistent Workers

This folder contains deployment payloads for two additional persistent Pegasus15-SME vLLM workers.

## Workers

| Worker | S3 checkpoint | GPUs |
| --- | --- | --- |
| `rl-consol-mtp-alpha1-gs60` | `s3://tl-data-training-pegasus-us-west-2/checkpoints/jeongyeon-nam/rl_rl_consol_0516_mtp_alpha1_bs_224_8k_lr5e_7_ckpt1000-base/global_step_60/actor/huggingface/` | 2 |
| `lia-soccer-mtp-ck1000-persistent` | `s3://tl-data-training-pegasus-us-west-2/checkpoints/jeongyeon-nam/ablation_260416_soccer_clean_filter_low_aug-highres_lr2e-6_qwen3_5_27b_soccer_dc_sme_low_filter_mtp_0513-base/checkpoint-1000-safetensors/` | 2 |

Both use:

```text
pipelineId: pegasus15-sme
image: 476114115052.dkr.ecr.us-west-2.amazonaws.com/tl-data-training-pegasus-vllm-video:main-fb19772
node_group_id: b300-pegasus
TP: 2
DP: 1
```

Unlike the earlier two persistent workers, these payloads do not hard-pin `node_names`. That lets Kueue/Kubernetes choose available B300 capacity instead of forcing every 2-GPU worker onto the same node.

## Call Through `/jobs`

Use the same orchestrator `/jobs` API. The only routing field that changes is `params.worker_type`.

For the RL checkpoint:

```json
{
  "pipelineId": "pegasus15-sme",
  "params": {
    "worker_type": "rl-consol-mtp-alpha1-gs60",
    "worker_call_mode": "loadbalancer",
    "audio_worker_call_mode": "loadbalancer"
  }
}
```

For soccer checkpoint-1000:

```json
{
  "pipelineId": "pegasus15-sme",
  "params": {
    "worker_type": "lia-soccer-mtp-ck1000-persistent",
    "worker_call_mode": "loadbalancer",
    "audio_worker_call_mode": "loadbalancer"
  }
}
```

## Deploy Commands

Open two port-forwards:

```bash
kubectl port-forward -n pegasus-platform svc/speccenter 18080:8080
kubectl port-forward -n pegasus-platform svc/infracontroller 18081:8080
```

Register and activate the RL worker:

```bash
curl -L -X POST http://127.0.0.1:18080/models \
  -H 'Content-Type: application/json' \
  --data-binary @rl_consol_mtp_alpha1_gs60_model_register.json

curl -L -X PATCH http://127.0.0.1:18080/status \
  -H 'Content-Type: application/json' \
  --data-binary @rl_consol_mtp_alpha1_gs60_status_active.json

curl -L -X PATCH \
  'http://127.0.0.1:18081/deployments/rl-consol-mtp-alpha1-gs60/scale?version=v1' \
  -H 'Content-Type: application/json' \
  --data-binary @scale_one.json
```

Register and activate the soccer checkpoint-1000 worker:

```bash
curl -L -X POST http://127.0.0.1:18080/models \
  -H 'Content-Type: application/json' \
  --data-binary @soccer_mtp_ck1000_model_register.json

curl -L -X PATCH http://127.0.0.1:18080/status \
  -H 'Content-Type: application/json' \
  --data-binary @soccer_mtp_ck1000_status_active.json

curl -L -X PATCH \
  'http://127.0.0.1:18081/deployments/lia-soccer-mtp-ck1000-persistent/scale?version=v1' \
  -H 'Content-Type: application/json' \
  --data-binary @scale_one.json
```

Check readiness:

```bash
kubectl get pods -n pegasus-platform -o wide | rg 'model-rl-consol-mtp-alpha1-gs60|model-lia-soccer-mtp-ck1000-persistent'
```

Expected:

```text
model-rl-consol-mtp-alpha1-gs60-v1-...          2/2   Running
model-lia-soccer-mtp-ck1000-persistent-v1-...   2/2   Running
```

## Manual Teardown

Scale to zero:

```bash
curl -L -X PATCH \
  'http://127.0.0.1:18081/deployments/rl-consol-mtp-alpha1-gs60/scale?version=v1' \
  -H 'Content-Type: application/json' \
  --data '{"replicas":0}'

curl -L -X PATCH \
  'http://127.0.0.1:18081/deployments/lia-soccer-mtp-ck1000-persistent/scale?version=v1' \
  -H 'Content-Type: application/json' \
  --data '{"replicas":0}'
```

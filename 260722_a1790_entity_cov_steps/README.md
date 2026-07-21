# A1790 Entity Coverage v0.2 steps 400–1600

This directory tracks four Eval V3 submissions for A1790 Entity-SME-4x.

- Dataset: https://huggingface.co/datasets/twelvelabs/entity_cov_v02_tdf
- Dataset revision: `5caf5ebd1ce03b6b6bb28a50504a8c36542d9433`
- Config and split: `default/test`
- Expected samples: 18
- Steps: 400, 800, 1200, 1600
- Pipeline and worker: `vllm-direct`, `vllm-video-v1`
- Node pool: `b300-pegasus`
- Tensor parallelism and data parallelism: 1 and 1
- Replicas: minimum 2, maximum 4
- Concurrency per replica: 8
- Maximum in-flight requests: 20
- Maximum output tokens: 65,536
- Temperature: 0
- Tensor cache: enabled

All four safetensors paths were verified in S3 before submission. Each path
contains eight top-level objects totaling 54,733,665,753 bytes.

Submit from a host that can reach the internal Eval V3 API:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 \
  260720_wandb_7n_entity_cov_eval/submit_verified_evals.py \
  --state 260722_a1790_entity_cov_steps/state.json \
  --eval-api-base \
  http://eval-v3-api-owen-2.pegasus-eval.svc.cluster.local:8090
```

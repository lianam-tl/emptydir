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

## Submitted runs

| Step | Eval run ID | Batch ID |
|---:|---|---|
| 400 | `03699e73-6bbd-5412-aff4-487c61bd08b2` | `batch-a0223f2d-9190-4903-a34d-d2b8f9cf4819` |
| 800 | `82c18432-2844-59bf-a3b8-b6d913bd1390` | `batch-f5b1e91d-2d6f-460b-8e96-ab3dad48286e` |
| 1200 | `a2d0ef8b-5ef2-5033-aec1-587e23a6b0a4` | `batch-1fb77893-77fa-4d19-9c5f-397246b4f432` |
| 1600 | `a47d832e-1ee3-502a-aebd-66692900c339` | `batch-4176ee5b-ba77-461e-b6b2-630a20529738` |

The CPU-node Slack monitor runs as PID `1836500` and posts to
`#fun-lia-trashcan` every 20 minutes or whenever status changes.

Submit from a host that can reach the internal Eval V3 API:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 \
  260720_wandb_7n_entity_cov_eval/submit_verified_evals.py \
  --state 260722_a1790_entity_cov_steps/state.json \
  --eval-api-base \
  http://eval-v3-api-owen-2.pegasus-eval.svc.cluster.local:8090
```

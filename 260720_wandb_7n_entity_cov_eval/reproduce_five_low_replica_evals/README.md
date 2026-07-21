# Reproduce the five low-replica entity coverage submissions

This directory contains a clean submission template for the five Eval V3 runs
submitted and cancelled on 2026-07-21. The template has no `eval_run_id`, so
`submit_verified_evals.py` will submit every entry instead of skipping it.

## Exact evaluation configuration

- Dataset: https://huggingface.co/datasets/twelvelabs/entity_cov_v02_tdf
- Dataset config: `default`
- Dataset split: `test`
- Dataset revision resolved on 2026-07-21: `5caf5ebd1ce03b6b6bb28a50504a8c36542d9433`
- Expected samples: 18
- Pipeline: `vllm-direct`
- Worker: `vllm-video-v1`
- Node pool: `b300-pegasus`
- Tensor parallelism: 1
- Data parallelism: 1
- Replicas: minimum 2, maximum 4
- Concurrency per replica: 8
- Maximum in-flight requests: 20
- Maximum output tokens: 65,536
- Temperature: 0
- Checkpoint conversion: disabled
- Tensor cache: enabled

The current Eval V3 create-run API does not accept a dataset-revision field.
It resolves the current Hugging Face revision during submission and saves that
resolved SHA in the Eval V3 database. Therefore, first confirm that the dataset
still resolves to the SHA above when exact dataset reproduction matters.

## Preview the complete wire payloads without submitting

Run this from `260720_wandb_7n_entity_cov_eval`:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 - <<'PY'
import json
from pathlib import Path

from monitor_and_submit import eval_payload

template_path = Path(
    "reproduce_five_low_replica_evals/submission.template.json"
)
items = json.loads(template_path.read_text(encoding="utf-8"))
payloads = [eval_payload(item) for item in items]
print(json.dumps(payloads, indent=2))
PY
```

The preview prints every API field, including every complete S3 checkpoint
path. It does not contact Eval V3.

## Submit from the CPU node

The Eval V3 API address is internal to the Kubernetes network, so run the
submission on `cpu`:

```bash
ssh cpu
reproduction_checkout=$(mktemp -d)
git clone \
  --branch lia/no-ticket-wandb-7n-entity-cov-eval-260720 \
  --single-branch \
  git@github.com:lianam-tl/emptydir.git \
  "$reproduction_checkout/emptydir"
cd "$reproduction_checkout/emptydir/260720_wandb_7n_entity_cov_eval"
submission_directory=$(mktemp -d)
cp reproduce_five_low_replica_evals/submission.template.json \
  "$submission_directory/state.json"
PYTHONDONTWRITEBYTECODE=1 python3 submit_verified_evals.py \
  --state "$submission_directory/state.json" \
  --eval-api-base \
  http://eval-v3-api-owen-2.pegasus-eval.svc.cluster.local:8090
python3 -m json.tool "$submission_directory/state.json"
```

The submission script sends one `POST /eval/runs` request per entry. Each
successful response has HTTP status 202 and an Eval V3 run ID. The script saves
those IDs in the temporary `state.json` and creates `status.html` next to it.

Using a temporary working copy is important: the script deliberately adds run
IDs to its state file. The checked-in template remains clean and reusable.

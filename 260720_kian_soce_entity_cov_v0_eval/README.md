# Kian SOCE entity coverage v0 evaluation

- Dataset: https://huggingface.co/datasets/twelvelabs/entity_cov_v0_tdf
- Config: `chunk_10m`
- Split: `test`
- Eval service: `eval-v3-api-lia` in namespace `pegasus-eval`
- Run ID: `7c2a3a9b-d814-5944-9243-19a34c971fa8`
- Batch ID: `batch-eb4225e9-1e4c-4da3-8ebb-eb3a266edc48`
- Checkpoint: `s3://tl-data-training-pegasus-us-west-2/checkpoints/kian-kim/soce_rl_260516_all_pair_p15_a1mckpt1000s60_w7030/`

The Lia sandbox was rebuilt from
[`lia/no-ticket-entity-cov-v02-benchmark-260716`](https://github.com/twelvelabs-io/pegasus/tree/lia/no-ticket-entity-cov-v02-benchmark-260716)
at `e0e07743d798416dcf23c4e74f4e2af9c4ae21b0`. The fresh run contains all 20
rows from `chunk_10m`.

The first request included `pollTimeoutSeconds` and was rejected with HTTP 422.
No run was created by that request. The accepted request in `payload.json` omits
that unsupported field.

## Monitor

The CPU-node monitor reads `SLACK_BOT_TOKEN` from the existing bot `.env` and
posts start and terminal notifications to `#fun-lia-trashcan`.

```bash
python3 poll_eval.py \
  --run-id 7c2a3a9b-d814-5944-9243-19a34c971fa8 \
  --api-base http://eval-v3-api-lia.pegasus-eval.svc.cluster.local:8090
```

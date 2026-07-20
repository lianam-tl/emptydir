# Kian SOCE entity coverage v0 evaluation

- Dataset: https://huggingface.co/datasets/twelvelabs/entity_cov_v0_tdf
- Config: `chunk_10m`
- Split: `test`
- Eval service: `eval-v3-api-lia` in namespace `pegasus-eval`
- Run ID: `c9676b7f-6433-5523-acaf-8d30ea602536`
- Batch ID: `batch-1f65209b-d152-49cf-97c8-9142bbb6be85`
- Checkpoint: `s3://tl-data-training-pegasus-us-west-2/checkpoints/kian-kim/soce_rl_260516_all_pair_p15_a1mckpt1000s60_w7030/`

The deployed Lia Eval V3 API already supports the legacy v0 entity coverage
benchmark, so no image rebuild was required. The run contains all 20 rows from
`chunk_10m`.

The first request included `pollTimeoutSeconds` and was rejected with HTTP 422.
No run was created by that request. The accepted request in `payload.json` omits
that unsupported field.

## Monitor

The CPU-node monitor reads `SLACK_BOT_TOKEN` from the existing bot `.env` and
posts start and terminal notifications to `#fun-lia-trashcan`.

```bash
python3 poll_eval.py \
  --run-id c9676b7f-6433-5523-acaf-8d30ea602536 \
  --api-base http://eval-v3-api-lia.pegasus-eval.svc.cluster.local:8090
```


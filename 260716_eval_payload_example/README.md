# Eval payload example

This payload runs the 20-sample, 10-minute entity-coverage v0 evaluation on
checkpoint 400 through the combined retry-fix and entity-cov v0.2 sandbox.

Target service:

- Kubernetes service: `eval-v3-api-pr-1729-cov-v02`
- Eval project: `pegasus-eval-service-sandbox-pr-1729-cov-v02`
- Local port-forward: `http://localhost:17291`

Submit:

```bash
curl -sS -X POST http://localhost:17291/eval/runs \
  -H 'Content-Type: application/json' \
  --data @260716_eval_payload_example/entity_cov_v0_ck0400.json | jq
```

Change `name`, `experimentName`, and `idempotencyKey` together when creating a
new run. Reusing the same idempotency key returns the existing run instead of
submitting duplicate work.

Dataset: https://huggingface.co/datasets/twelvelabs/entity_cov_v0_tdf

The separate v0.2 dataset payload is not used here because its current run
creation reaches PostgreSQL's indexed-row size limit before an eval run is
created.

# Kian one-node B300 Pegasus deployment

This deployment runs the Kian checkpoint on one `b300-pegasus` node:

- worker type: `kian-socerl-s60-1node`
- replicas: 1
- GPUs: 8
- tensor parallelism: 2
- data parallelism: 4
- image: `main-fb19772`

The deployed checkpoint is:

```text
s3://tl-data-training-pegasus-us-west-2/checkpoints/kian-kim/soce_rl_260516_all_pair_p15_a1mckpt1000s60_w7030/
```

Register and activate it through SpecCenter:

```bash
kubectl port-forward -n pegasus-platform svc/speccenter 18080:8080

curl -X POST http://127.0.0.1:18080/models \
  -H 'Content-Type: application/json' \
  --data-binary @model_register.json

curl -X PATCH http://127.0.0.1:18080/status \
  -H 'Content-Type: application/json' \
  --data-binary @status_active.json
```

Use `status_inactive.json` with the same status endpoint to tear it down.

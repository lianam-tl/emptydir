# Two-model B300 ENS deployment

This deployment runs two checkpoint-1000 model workers on `b300-ens`:

- `kian-socerl-s60-4node`: 4 replicas, 8 GPUs each, 32 GPUs total
- `pegasus-sft-2node`: 2 replicas, 8 GPUs each, 16 GPUs total
- 48 GPUs total
- each replica uses `TP=2`, `DP=4`, and concurrency 4

The Kian checkpoint path supplied in chat did not exist. The matching existing
prefix used here is:

```text
s3://tl-data-training-pegasus-us-west-2/checkpoints/kian-kim/soce_rl_260516_all_pair_p15_a1mckpt1000s60_w7030/
```

The workers use the previous `main-fb19772` image. Therefore the direct worker
fallback remains 16,384 output tokens, the structured-output string cap remains
26,000 characters, and the live `pegasus15-sme` pipeline default remains 32,000
tokens.

Register and activate both models through SpecCenter:

```bash
kubectl port-forward -n pegasus-platform svc/speccenter 18080:8080

for model in kian pegasus_sft; do
  curl -L -X POST http://127.0.0.1:18080/models \
    -H 'Content-Type: application/json' \
    --data-binary "@model_register_${model}.json"
  curl -L -X PATCH http://127.0.0.1:18080/status \
    -H 'Content-Type: application/json' \
    --data-binary "@status_active_${model}.json"
done
```

Each pod requests all 8 GPUs on a B300 node, so replicas necessarily occupy
distinct nodes without explicit node-name pinning.

To tear down, patch the matching `status_inactive_*.json` payloads.


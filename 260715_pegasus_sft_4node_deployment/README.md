# Pegasus SFT: 4-node deployment

This deployment keeps the existing `pegasus-15-sft` worker unchanged and
adds `pegasus-sft-4node` on four `b300-ens` nodes.

- 4 replicas on any available `b300-ens` nodes
- 8 GPUs per replica, 32 GPUs total
- `TP=2`, `DP=4` per replica
- concurrency 4 per replica
- checkpoint: `checkpoint-1000-safetensors`
- v2 generation default: 128,000 tokens
- v2 structured-output string cap: 102,000 characters

Each replica requests all 8 GPUs on a B300 node, so Kubernetes necessarily
places the four replicas on four distinct nodes.

Register `b300-ens` if it is not already present, register the model, and mark
it active:

```bash
kubectl port-forward -n pegasus-platform svc/speccenter 18080:8080

curl -L -X POST http://127.0.0.1:18080/nodegroups \
  -H 'Content-Type: application/json' \
  --data-binary @node_group_b300_ens.json

curl -L -X POST http://127.0.0.1:18080/models \
  -H 'Content-Type: application/json' \
  --data-binary @model_register.json

curl -L -X PATCH http://127.0.0.1:18080/status \
  -H 'Content-Type: application/json' \
  --data-binary @status_active.json
```

To stop it, patch `status_inactive.json`. Deactivation removes the generated
deployment and its pods.

`model_register_v2_128k.json` upgrades the same worker to the branch-specific
128K/102K image. Activate it with `status_active_v2.json`; v1 remains available
as an inactive rollback version.

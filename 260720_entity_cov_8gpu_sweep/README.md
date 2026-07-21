# Entity coverage 8-GPU checkpoint sweep

- Dataset: https://huggingface.co/datasets/twelvelabs/entity_cov_v0_tdf
- Config: `chunk_10m`
- Split: `test`
- Eval service: `eval-v3-api-lia`
- Topology per checkpoint: TP=1, DP=1, 8 replicas, 8 B300 GPUs
- Steps per family: 400, 800, 1200, 1600, 2000

The sweep evaluates the `consol-h0mn2x` and `soccer-lvreason-mcq` checkpoint
families. Every model path uses the converted `checkpoint-<step>-safetensors/`
directory.

## Submit

```bash
kubectl -n pegasus-eval port-forward svc/eval-v3-api-lia 18091:8090
python3 submit_sweep.py --api-base http://127.0.0.1:18091
```

Refresh the JSON and HTML status without submitting new runs:

```bash
python3 refresh_status.py \
  --results submission_results.json \
  --api-base http://127.0.0.1:18091
```

## Monitor

```bash
python3 poll_sweep.py \
  --results submission_results.json \
  --api-base http://eval-v3-api-lia.pegasus-eval.svc.cluster.local:8090
```

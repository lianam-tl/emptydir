# H0 Entity v1.2 step 100 Entity Coverage v0.2

This directory tracks the DCP export and Eval V3 submission for
`sft-h0-entity-v1-2-7n-qwen3-5-27b-base` step 100.

- Dataset: https://huggingface.co/datasets/twelvelabs/entity_cov_v02_tdf
- Dataset revision: `5caf5ebd1ce03b6b6bb28a50504a8c36542d9433`
- Config and split: `default/test` (18 samples)
- Eval sandbox: `owen-2`
- Replicas: minimum 2, maximum 4
- Concurrency per replica: 8
- Tensor parallelism and data parallelism: 1 and 1
- Maximum in-flight requests: 20
- Maximum output tokens: 65,536
- Tensor cache: enabled

The source is a DCP checkpoint. `export.yaml` converts it to safetensors.
`monitor_and_submit.py` waits for the export, submits the evaluation, and
posts updates and 20-minute heartbeats to `#fun-lia-trashcan`.

Export job: `export-h0-entity-v1-2-s100-tikkv4`

CPU monitor: PID `1902504` under
`/home/jeongyeon-nam/eval-monitor-export-h0-entity-v1-2-s100-tikkv4`.

# H0 Entity v1.2 step 200 Entity Coverage v0.2

This directory tracks the DCP export and Eval V3 submission for
`sft-h0-entity-v1-2-7n-qwen3-5-27b-base` step 200.

- Dataset: https://huggingface.co/datasets/twelvelabs/entity_cov_v02_tdf
- Dataset revision: `5caf5ebd1ce03b6b6bb28a50504a8c36542d9433`
- Config and split: `default/test` (18 samples)
- Eval sandbox: `owen-2`
- Replicas: minimum 8, maximum 8
- Concurrency per replica: 8
- Tensor parallelism and data parallelism: 1 and 1
- Maximum in-flight requests: 20
- Maximum output tokens: 65,536
- Tensor cache: enabled

The source is a DCP checkpoint. `export.yaml` converts it to safetensors.
`monitor_and_submit.py` waits for the export, submits the evaluation, and
posts updates and 20-minute heartbeats to `#fun-lia-trashcan`.

Export job: `export-h0-entity-v1-2-s200-tikx2m`

The first Eval V3 run, `2b95ae9d-bd6e-546e-ac51-43e1a5dece6c`, produced 8 of
18 predictions before GPU reservation contention terminated the batch.

Current Eval V3 run: `368487fc-51c1-5f11-a1a0-297d833c092a`

XPlatform batch: `batch-7039effa-6b4b-4217-bb82-ad7033d49193`

CPU monitor: PID `1929341` under
`/home/jeongyeon-nam/eval-retry2-h0-entity-v1-2-s200-260723`.

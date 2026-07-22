# A1790 Entity-SME-4x step 2200 Entity Coverage v0.2

This directory tracks the DCP export and Eval V3 submission for
`sft_a1790_entity_sme_4x_consol_7n_qwen3_5_27b-base` step 2200.

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

Export job: `export-a1790-entity-sme4x-s2200-tikl46`

The first Eval V3 run, `e69c7e97-7a9a-53ac-839d-267ebf505ef8`, failed before
inference because its submitter lease expired before the XPlatform request ID
was saved. No matching request was found among the latest 1,000 XPlatform
batches, so it was resubmitted with a new idempotency key.

Retry Eval V3 run: `9076d70b-c407-5004-b9b9-a32f99fc8c65`

CPU monitor: PID `1906164` under
`/home/jeongyeon-nam/eval-retry-a1790-entity-sme4x-s2200-260722`.

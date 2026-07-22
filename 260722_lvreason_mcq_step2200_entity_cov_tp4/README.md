# LVReason MCQ step 2200 Entity Coverage v0.2

This directory tracks the DCP export and Eval V3 submission for LVReason MCQ
step 2200.

- Dataset: https://huggingface.co/datasets/twelvelabs/entity_cov_v02_tdf
- Dataset revision: `5caf5ebd1ce03b6b6bb28a50504a8c36542d9433`
- Config and split: `default/test` (18 samples)
- Eval sandbox: `owen-2`
- Replicas: minimum 2, maximum 4
- Concurrency per replica: 8
- Tensor parallelism: 4 GPUs per replica
- Data parallelism: 1
- Maximum in-flight requests: 20
- Maximum output tokens: 65,536
- Tensor cache: enabled

The source is a DCP checkpoint. `export.yaml` converts it to safetensors.
`monitor_and_submit.py` waits for the export, submits the evaluation, and
posts updates to `#fun-lia-trashcan`.


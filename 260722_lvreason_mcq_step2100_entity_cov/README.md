# LVReason MCQ step 2100 Entity Coverage v0.2

This directory tracks the DCP export and Eval V3 submission for LVReason MCQ
step 2100.

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

The source is a DCP checkpoint. `export.yaml` converts it to safetensors;
`monitor_and_submit.py` under `260720_wandb_7n_entity_cov_eval/` submits the
evaluation after the export succeeds.

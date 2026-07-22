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

The source is a DCP checkpoint. `export.yaml` converts it to safetensors.
After the model path is verified, `submit_verified_evals.py` under
`260720_wandb_7n_entity_cov_eval/` submits the evaluation.

Export job `export-sft-lvreason-mcq-s2100-tik5wl` produced and uploaded the
55,563,009,392-byte model. It was marked failed only because the generic
manifest requested nonexistent `vision_process.py` after the upload. The four
LVReason-specific metadata files were copied from the source checkpoint, and
`export.yaml` now lists the correct files for reproduction.

Eval run: `9d185735-eb19-5463-b86b-3959ab619f27`  
Batch: `batch-2498e936-1e29-40cb-b5fa-c9d5b77cd034`

The CPU-node Slack monitor is PID `1876233` and posts status changes and
20-minute heartbeats to `#fun-lia-trashcan`.

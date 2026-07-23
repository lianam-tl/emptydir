# A1865 Entity-SME Whisper Entity Coverage v0.2

This directory tracks Entity Coverage v0.2 evaluations for steps 400 and 800
of `sft_a1865_entity_sme_whisper_lr2e-6_qwen3_5_27b-base`.

- Checkpoint root: `s3://tl-data-training-pegasus-us-west-2/checkpoints/chanho-ahn/sft_a1865_entity_sme_whisper_lr2e-6_qwen3_5_27b-base/`
- Dataset: https://huggingface.co/datasets/twelvelabs/entity_cov_v02_tdf
- Dataset revision: `5caf5ebd1ce03b6b6bb28a50504a8c36542d9433`
- Config and split: `default/test` (18 samples)
- Eval sandbox: `owen-2`
- Replicas: steps 400 and 800 fixed at 2; remaining steps fixed at 8
- Concurrency per replica: 8
- Maximum output tokens: 65,536
- Tensor cache: enabled
- Training Git SHA: `3d52392097e448b8d4b4c9872c5d59c156f5e388`

Step 400 already contains `model.safetensors`. Step 800 is a DCP checkpoint,
so `export_step800.yaml` exports it before its evaluation is submitted.

Step 800 export job: `export-a1865-entity-sme-whisper-s800-timegl`

Step 400 Eval V3 run: `f6631367-1f85-5b77-b947-fd073a0cab9a`

Step 400 XPlatform batch: `batch-84a6a11c-2762-402e-b9e5-810d6b875062`

Step 400 CPU monitor: PID `1953939` under
`/home/jeongyeon-nam/eval-a1865-entity-sme-whisper-s400-260723`.

Step 400 completed 18 of 18 predictions. Overall naming + appearance IoU is
`0.276020`; half naming + appearance IoU is `0.343128`.

Step 800 CPU monitor: PID `1954090` under
`/home/jeongyeon-nam/eval-a1865-entity-sme-whisper-s800-260723`.

Step 800 completed 18 of 18 predictions. Overall naming + appearance IoU is
`0.263756`; half naming + appearance IoU is `0.297342`.

Steps 100, 200, 300, 500, 600, and 700 use one sequential export job. After
the export completes, `monitor_sequential.py` submits fixed-8-replica
evaluations in that exact order and waits for each one to finish before
submitting the next.

Sequential export job: `export-a1865-entity-sme-whisper-sequential-timezn`

Sequential CPU monitor: PID `1957209` under
`/home/jeongyeon-nam/eval-a1865-entity-sme-whisper-sequential-260723`.

Step 100 Eval V3 run: `b44f8811-87f8-5f92-b3b4-b3f712c104c3`

Step 100 XPlatform batch: `batch-a44a1a01-9e0b-42bd-a0b6-127fb7209b03`

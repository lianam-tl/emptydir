# Entity Coverage v0.2: A1790 and SFT LVReason checkpoints

This directory tracks five Eval V3 targets requested on 2026-07-22.

- Dataset: https://huggingface.co/datasets/twelvelabs/entity_cov_v02_tdf
- Dataset revision: `5caf5ebd1ce03b6b6bb28a50504a8c36542d9433`
- Config and split: `default/test` (18 samples)
- A1790 Entity-SME-4x: steps 1900, 2000, 2100
- SFT LVReason MCQ: steps 1600, 2000
- Eval sandbox: `owen-2`
- Replicas: minimum 2, maximum 4
- Concurrency per replica: 8
- Tensor parallelism and data parallelism: 1 and 1
- Maximum in-flight requests: 20
- Maximum output tokens: 65,536
- Tensor cache: enabled

The LVReason checkpoints already have safetensors exports. The A1790 targets
exist only as DCP checkpoints, so the three manifests in `a1790/exports/`
export them first. `monitor_and_submit.py` from
`260720_wandb_7n_entity_cov_eval/` submits each A1790 eval after its export
succeeds.

See `status.html` for the compact visual summary.

## Current jobs

| Family | Step | Export/Eval ID | State |
|---|---:|---|---|
| A1790 Entity-SME-4x | 1900 | `export-a1790-entity-sme4x-s1900-tijsfi` | Export running; eval auto-submit enabled |
| A1790 Entity-SME-4x | 2000 | `export-a1790-entity-sme4x-s2000-tijsfr` | Export running; eval auto-submit enabled |
| A1790 Entity-SME-4x | 2100 | `export-a1790-entity-sme4x-s2100-tijsg4` | Export running; eval auto-submit enabled |
| SFT LVReason MCQ | 1600 | `918e01d5-cd55-501d-bfc2-09b53e5d5f34` | Processing |
| SFT LVReason MCQ | 2000 | `d10c50da-66a2-59e7-b81a-21ac4a26fafd` | Processing |

The CPU-node A1790 export/eval watcher is PID `1841028` and posts updates to
`#fun-lia-trashcan`. The LVReason eval watcher is PID `1843012` and posts to
the same channel.

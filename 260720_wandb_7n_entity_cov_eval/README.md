# W&B 7-node checkpoint entity coverage evaluation

- W&B: https://wandb.ai/twelvelabs/pegasus-sme/runs/kp1ju1r1
- W&B: https://wandb.ai/twelvelabs/pegasus-sme/runs/bqm74hdf
- Dataset: https://huggingface.co/datasets/twelvelabs/entity_cov_v0_tdf
- Config: `chunk_10m`
- Split: `test`
- Eval topology: TP=1, DP=1, 8 replicas
- Current 400-step targets: 400, 800, 1200 from each run

The training jobs store DCP checkpoints, so every target first requires a
one-GPU export to a sibling `checkpoint-<step>-safetensors/` directory. The
CPU monitor submits Eval V3 automatically after each export succeeds.

Generate manifests:

```bash
python3 generate_export_manifests.py
```

Submit each manifest from `exports/` with:

```bash
tlab submit <manifest> --priority high --yes
```

After recording the resulting PyTorchJob names in `export_submissions.json`,
run the monitor on the CPU node:

```bash
python3 monitor_and_submit.py \
  --submissions export_submissions.json \
  --state monitor_state.json \
  --eval-api-base http://eval-v3-api-lia.pegasus-eval.svc.cluster.local:8090
```

Step 1600 is tracked separately in `a1790_step1600_submission.json`. Its export
manifest converts the DCP checkpoint to `checkpoint-1600-safetensors`; the
monitor then submits https://huggingface.co/datasets/twelvelabs/entity_cov_v02_tdf
with 65,536 output tokens, TP=1, and eight replicas.

Export job: `export-a1790-entity-sme4x-s1600-tiicoj`. The first attempt,
`export-a1790-entity-sme4x-s1600-tiici0`, failed during a transient GitHub DNS
lookup and was replaced.

CPU handoff monitor: PID `1726652`, polling every 60 seconds and notifying
`#fun-lia-trashcan`.

## Requested v0.2 follow-up runs

`requested_eval_state.json` tracks the inference-ready Pegasus 1.5 SFT,
Pegasus 1.5 RL, Kian SOCE, and A1790 step-1600 checkpoints. Kian is submitted
again because the earlier completed result used dataset revision `96d4b609`,
while these requested runs use revision `5caf5ebd`.

A1790 step 1800 was available only as a DCP training checkpoint, so
`exports/a1790-entity-sme4x-step1800.yaml` exports it to safetensors first.
`a1790_step1800_submission.json` is then consumed by `monitor_and_submit.py`,
which automatically submits the evaluation against
https://huggingface.co/datasets/twelvelabs/entity_cov_v02_tdf after export.

Submitted direct eval runs:

- Pegasus 1.5 SFT: `968ce189-f02c-5c4f-9b91-2172edd43714`
- Pegasus 1.5 RL: `d84e4056-2195-57b9-889f-b4655e2461f4`
- Pegasus 1.5 Kian SOCE: `8f941a94-0cd6-5243-b3e9-f56f83423631`
- A1790 step 1600 rerun: `5a38f661-4868-55ef-89f6-2d1e6d037915`

A1790 step 1800 export job:
`export-a1790-entity-sme4x-s1800-tiitci`.

All new evals resolve to dataset revision
`5caf5ebd1ce03b6b6bb28a50504a8c36542d9433` (18 samples). CPU-node Slack
pollers are PID `1812568` for the four inference-ready runs and PID `1812264`
for the step-1800 export-to-eval pipeline.

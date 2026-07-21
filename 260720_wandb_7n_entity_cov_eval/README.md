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
Pegasus 1.5 RL, and A1790 step-1600 checkpoints. The existing completed Kian
SOCE run is reused rather than submitted again.

A1790 step 1800 was available only as a DCP training checkpoint, so
`exports/a1790-entity-sme4x-step1800.yaml` exports it to safetensors first.
`a1790_step1800_submission.json` is then consumed by `monitor_and_submit.py`,
which automatically submits the evaluation against
https://huggingface.co/datasets/twelvelabs/entity_cov_v02_tdf after export.

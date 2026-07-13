# A-1548 metadata reward rollout analysis

This analysis checks whether W&B run `q0om466c` improves metadata values or exploits the rule-based reward.

## Inputs

- W&B history: `twelvelabs/pegasus-rl/q0om466c`
- Rollout snapshots: steps 1, 5, 10, 20, 30, 40, 50, 60, 70, 80, and 87
- S3 prefix:
  `s3://tl-data-training-pegasus-us-west-2/checkpoints/jeongyeon-nam/rl_consol_260515_soccer_lvreason_mcq_sft1300_mtp_a05_nothink_8k-meta-reward-v2-nothink-7n-a1-mtp01-260713/rollout_logs/`

Rollouts must be downloaded with `s5cmd`; the Python analysis does not download data.

## Run

```bash
/Users/long8v/.venv/bin/python analyze_rollouts.py \
  --data-dir data \
  --output-dir output \
  --env-file ~/pegasus/.env
```

Generated outputs:

- `output/report.html`
- `output/summary.json`
- `output/per_step.csv`
- `output/wandb_history.csv`
- `output/dtype_comparison.csv`
- `output/dataset_mix.csv`
- `output/dataset_comparison.csv`
- `output/repeated_value_concentration.csv`

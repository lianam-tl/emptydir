# Kian SOCE-RL Entity Coverage v0.2 Eval

Checkpoint:

```text
s3://tl-data-training-pegasus-us-west-2/checkpoints/kian-kim/soce_rl_260516_all_pair_p15_a1mckpt1000s60_w7030/
```

Benchmark dataset:

```text
https://huggingface.co/datasets/twelvelabs/entity_cov_v02_tdf
```

Submit through the `eval-v3-api-lia` sandbox port-forward:

```bash
python3 260716_kian_soce_entity_cov_v02_eval/run_eval.py
```

Poll the submitted run:

```bash
python3 260716_kian_soce_entity_cov_v02_eval/run_eval.py --poll
```

The script writes:

- `outputs/payload.json`
- `outputs/submission_record.json`
- `outputs/status.html`

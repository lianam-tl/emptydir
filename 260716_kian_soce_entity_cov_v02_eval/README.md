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

## Reproduce the PR 1729 cov-v02 DB failure

This reproduces the failed submission against `eval-v3-api-pr-1729-cov-v02`.
The API accepts `twelvelabs/entity_cov_v02_tdf`, but run creation fails before
GPU scheduling because Postgres cannot index the long `segment_definition` value.

```bash
./260716_kian_soce_entity_cov_v02_eval/reproduce_pr1729_cov_v02_failure.sh
```

Expected failure:

```text
ProgramLimitExceeded: index row size 3960 exceeds btree version 4 maximum 2704
for index "eval_tasks_eval_run_id_sample_external_id_segment_definitio_key"
```

The original failed attempt is recorded in:

- `outputs_pr_1729_cov_v02/payload.json`
- `outputs_pr_1729_cov_v02/submit_error.json`

# Entity Coverage Eval for W&B Checkpoints

W&B run: https://wandb.ai/twelvelabs/pegasus-sme/runs/06h8x4z6

Run name from W&B: `consol-260416-clean-filter-less-aug-highres-h0-th8i69`

Checkpoint base:

```text
s3://tl-data-training-pegasus-us-west-2/checkpoints/jeongyeon-nam/consol_260416_clean_filter_less_aug_highres_h0_mn_sme_2x_lr2e-6_qwen3_5_27b-base
```

I submitted the 10-minute entity coverage eval for every available 400-step safetensors checkpoint:

- `checkpoint-400-safetensors`
- `checkpoint-800-safetensors`
- `checkpoint-1200-safetensors`
- `checkpoint-1600-safetensors`

`checkpoint-2000-safetensors` was created after converting the raw DCP checkpoint, then added as the next eval target. `checkpoint-2200-safetensors` exists but is not a 400-step checkpoint, so it is not part of this run.

Dataset link: https://huggingface.co/datasets/twelvelabs/entity_cov_v0_tdf

Eval payload defaults:

- dataset: `twelvelabs/entity_cov_v0_tdf`
- config: `chunk_10m`
- split: `test`
- pipeline: `vllm-direct`
- worker: `vllm-video-v1`
- node pool: `b300-pegasus`
- replicas: `1`
- TP/DP: `2/1`
- max tokens: `16384`
- temperature: `0`
- image: eval-service deployment default; no `imageUrl` override, because training-prod rejects images outside its workload policy

Submit:

```bash
python3 260716_entity_cov_wandb_ckpt_eval/submit_entity_cov_wandb_ckpts.py
```

Poll:

```bash
python3 260716_entity_cov_wandb_ckpt_eval/submit_entity_cov_wandb_ckpts.py --poll
```

Poll with Slack notification:

```bash
SLACK_BOT_TOKEN=<token> \
python3 260716_entity_cov_wandb_ckpt_eval/submit_entity_cov_wandb_ckpts.py \
  --poll \
  --poll-seconds 120 \
  --timeout-seconds 21600 \
  --slack-channel '#fun-lia-trashcan'
```

Payloads and submission responses are under `payloads/`.

## TL;DR

This folder tracks eval-service submissions for W&B run `06h8x4z6`, checkpoints `400/800/1200/1600/2000`, entity coverage `chunk_10m`.

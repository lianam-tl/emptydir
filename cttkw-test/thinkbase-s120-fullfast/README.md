# thinkbase step 120 — full `sme_eval_v3.1_fast` (1167 samples)

End-to-end run with PR #80 final image (`lia-vllm-chat-template-kwargs-7d0418d`), `chat_template_kwargs={enable_thinking: true, supports_thinking: true}`, on the RL `think-base` checkpoint at `global_step_120`. Full mixed-subset eval (30 subsets, 1167 samples total).

## Setup

| field | value |
|---|---|
| run_id | `3001f85a-7b11-4ea3-888a-8973e03b91e1` |
| batch_id | `batch-40ee3487-cfb8-487d-ad96-981dacb9cc6e` |
| dataset | `twelvelabs/sme_eval_v3.1_fast` (all subsets) |
| modelPath | `s3://tl-data-training-pegasus-us-west-2/checkpoints/jeongyeon-nam/rl_rl_v1.5g_wo_a0a1_vc-consol_8k_from_qwen27b_alpha1_think-base/global_step_120/actor/huggingface/` |
| chat_template_kwargs | `{"enable_thinking": true, "supports_thinking": true}` |
| worker image | `lia-vllm-chat-template-kwargs-7d0418d` |
| concurrency × replicas | 4 × 4 |
| maxTokens | 16384 |

## Aggregate results

| metric | value |
|---|---|
| samples | 1167 |
| `think_blocks` populated | **1167 / 1167 (100.0 %)** |
| `text` starts with ```` ```json ```` block | **1165 / 1167 (99.8 %)** |
| `finish_reason = stop` | 1158 / 1167 (99.2 %) |
| `finish_reason = length` | 9 / 1167 (0.8 %) |

Token distribution (min / median / p95 / max):

| | min | median | p95 | max |
|---|---|---|---|---|
| input_tokens | 7,801 | 90,107 | 144,422 | 182,737 |
| output_tokens | 331 | 2,779 | 10,133 | 24,271 |
| think_block_chars | 882 | 5,116 | 18,302 | 63,108 |

(Input variance is huge because the dataset spans short A0_* clips and long H16_* sport halves.)

## Length-truncated samples (9)

7 of the 9 finished with the `</think>` already emitted plus the start of a ```` ```json ```` answer that ran out of tokens mid-array — the assistant_think_prefix partition still works (think_blocks captures the full reasoning, text contains the partial JSON).

The remaining **2** samples (0.2 % of the dataset) hit `max_tokens=16384` while still in the reasoning phase, so `</think>` was never emitted. In that case the PR fix puts the entire raw output into `think_blocks[0]` and leaves `text=""` — reasoning is preserved instead of silently lost.

## File layout

- `tasks.json` — per-sample mapping: `request_key` → `sample_id`, `video`, `segment_definition`, `metadata_schema`.
- `summary.csv` — one row per sample: input_tokens, output_tokens, finish_reason, think_block_chars, text_chars, text_starts_json_block.
- `outputs/<run_id>.<run_id>.<request_key>.json` × 1167 — raw worker output JSON (text, input_tokens, output_tokens, finish_reason, think_blocks).

## How to query

```bash
# pick any sample's request_key and view the reasoning vs the JSON answer
KEY=$(awk -F, 'NR==2 {print $1}' summary.csv)
jq -r '.think_blocks[0]' < outputs/*.${KEY}*.json | head -40
jq -r '.text'           < outputs/*.${KEY}*.json | head -40

# how many samples have JSON-conformant answers
awk -F, 'NR>1 && $7=="True" {n++} END {print n}' summary.csv

# distribution of output_tokens
awk -F, 'NR>1 {print $3}' summary.csv | sort -n | awk 'BEGIN{n=0} {a[n++]=$1} END {print "p50:", a[int(n*0.5)], "p95:", a[int(n*0.95)], "max:", a[n-1]}'
```

## Note on eval-service status

The eval-service UI reports this run as `failed` because the batch-request → eval-service status callback didn't sync cleanly. The actual model inference completed every sample (orchestrator and S3 outputs both confirm 1167/1167 successful); only the eval-service-side ingest state is stale. The raw outputs in this folder are the source of truth.

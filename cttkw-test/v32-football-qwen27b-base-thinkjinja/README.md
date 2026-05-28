# v32 — stock Qwen3.5-27B base + Pegasus video processor + RL `think` chat_template

Same setup as v29 (H16_FOOTBALL_r1, 10 samples, `chat_template_kwargs={enable_thinking: true, supports_thinking: true}`, PR #80 commit `1d54373` worker image) but the model weights come from the **stock Qwen3.5-27B instruct** under `hf_models/Qwen/Qwen3.5-27B/` rather than any RL checkpoint. Only the `chat_template.jinja` and the Pegasus video processor files (`processing_pegasus_qwen3_vl.py`, `processor_config.json`, `ffmpeg_fetch_video.py`) were swapped in from the `think-base` RL checkpoint so the base instruct weights run on our video pipeline.

The goal is to isolate whether literal `</think>` emission is a base-instruct capability or an RL-learned behavior.

## Setup

| field | value |
|---|---|
| run_id | `ac2b739f-4266-4c6c-88ad-188a6ddf79c8` |
| batch_id | `batch-d4722104-67ab-4274-8728-2ead06f67bc0` |
| dataset | `twelvelabs/sme_eval_v3.1_fast` |
| config | `H16_FOOTBALL_r1` |
| modelPath | `s3://tl-data-training-pegasus-us-west-2/checkpoints/jeongyeon-nam/lia-cttkw-qwen3.5-27b-think-template/` |
| chat_template_kwargs | `{"enable_thinking": true, "supports_thinking": true}` |
| worker image | `lia-vllm-chat-template-kwargs-1d54373` (post auto-drop fix) |

The hybrid checkpoint was built by:
1. `aws s3 sync` of `hf_models/Qwen/Qwen3.5-27B/` (52 GB, 24 files — 11 safetensors shards, tokenizer, config, base chat_template, …)
2. Overwriting `chat_template.jinja` with the RL `think-base/global_step_120/actor/huggingface/chat_template.jinja` (11 287 bytes — the one that handles `enable_thinking` / `supports_thinking` and the `json_schema` content item).
3. Copying the missing Pegasus video files (`processing_pegasus_qwen3_vl.py`, `processor_config.json`, `ffmpeg_fetch_video.py`) from the same RL checkpoint.

## Findings

| metric | v29 (RL think-base step 120) | v32 (stock 27B + same templates) |
|---|---|---|
| think_blocks populated | 0/10 | 0/10 |
| `</think>` in text | 0/10 | 0/10 |
| ```` ```json ```` block in text | 10/10 | **6/10** |
| finish=`stop` | 10/10 | 6/10 |
| finish=`length` (hit max_tokens=16384) | 0/10 | **4/10** |
| avg output_tokens (stop only) | ~1 800 | ~3 700 |

Take-aways:
- **`</think>` is not a stock-instruct capability.** Even fresh Qwen3.5-27B with the same chat template, schema in prompt, and `<think>\n` assistant prefix never emits the literal `</think>` token. It is therefore something RL would need to teach (reward-shape) — not something the base model "knows" and the current RL is forgetting.
- **Schema-driven JSON emission IS a stock capability.** When the schema is rendered as text in the user message (chat template's `json_schema` handler), 6/10 stock-27B runs produce a properly formatted ```` ```json ```` answer block — the same pattern as the RL output. RL pushes this from 6/10 → 10/10 and shortens the rambling (no length-truncation, smaller avg output).
- The 4 length-truncated v32 samples show what happens when the base model doesn't know when to stop reasoning — it rambles in the reasoning prefix until `max_tokens=16384` runs out, never emitting `</think>` or JSON.

## Implication for PR #80

Plumbing-side the PR is correct: `think_blocks` would populate as soon as the model emits the `<think>...</think>` pair. But no available Qwen3.5-27B variant (stock instruct, RL think-base step 20, RL think-base step 120, RL think-7n step 120) actually emits the closing tag. Marker emission needs reward-shaping in a follow-up training run — it is not a regression in this PR. Document in the PR description as a known limitation; the schema-as-text plumbing fix (commit `1d54373`) is the part of this work that is necessary for the broader thinking-mode flow.

## Files

- `tasks.json` — per-sample mapping.
- `<run_id>.<run_id>.<request_key>.json` × 10 — raw worker output JSON.

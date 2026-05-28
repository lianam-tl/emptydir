# v27 — football think outputs (RL think-base step 120, H16_FOOTBALL_r1)

10 sample eval through eval-service → batch-request → orchestrator → wf-engine → batch-worker (v2 binary infer), with `chatTemplateKwargs={enable_thinking: true, supports_thinking: true}`.

## Setup

| field | value |
|---|---|
| run_id | `38f2c57f-fdcd-4a6f-99e8-e87b53603370` |
| batch_id | `batch-f52017fd-aa96-4686-96d9-2f9235a99cfa` |
| dataset | `twelvelabs/sme_eval_v3.1_fast` |
| config | `H16_FOOTBALL_r1` |
| modelPath | `s3://tl-data-training-pegasus-us-west-2/checkpoints/jeongyeon-nam/rl_rl_v1.5g_wo_a0a1_vc-consol_8k_from_qwen27b_alpha1_think-base/global_step_120/actor/huggingface/` |
| chat_template_kwargs | `{"enable_thinking": true, "supports_thinking": true}` |
| worker image | `lia-vllm-chat-template-kwargs-515c305` (PR #80 build with think_blocks plumbing) |

## Findings

- All 10 outputs contain reasoning (numbered steps, "I will analyze...", etc.) — `enable_thinking` system instruction is honored.
- **No sample emits the literal `<think>` or `</think>` markers.** Chat template prepends `<think>\n` as the assistant prefix, but the model never closes with `</think>`.
- 9/10 finish naturally (`finish_reason=stop`); 1 hits `max_tokens=16384` while still in reasoning.
- `think_blocks` is `[]` on every sample — the worker regex needs the `<think>...</think>` pair, so no extraction happens. The reasoning text is **preserved in the `text` field** (no data loss).

This is a model-training issue, not a PR plumbing issue: `chat_template_kwargs` reaches `processor.apply_chat_template` correctly (verified via `CTTKW_V2` log on the worker pod and the +30-token signature in `input_tokens`), but the RL checkpoint at step 120 doesn't reward `</think>` emission. The xplatform-side plumbing in PR #80 is correct — `think_blocks` would populate as soon as the checkpoint emits the closing tag.

## Files

- `tasks.json` — per-sample mapping: request_key → sample_id, video URL, segment_definition, metadata_schema.
- `<run_id>.<run_id>.<request_key>.json` — raw worker output JSON per sample. Schema: `text`, `request_id`, `video_frames`, `input_tokens`, `output_tokens`, `finish_reason`, `think_blocks`.
- `summary.md` — quick per-sample table (output_tokens, finish_reason, video, segment-definition snippet).

## How to read

```bash
# Pick a key and inspect text
KEY=2QZMS7477VNS
jq -r '.text' < 38f2c57f-fdcd-4a6f-99e8-e87b53603370.38f2c57f-fdcd-4a6f-99e8-e87b53603370.${KEY}*.json

# Match key to its segment definition
jq '.[] | select(.request_key | startswith("2QZMS"))' tasks.json
```

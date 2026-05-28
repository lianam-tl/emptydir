# v29 — football think outputs with `schema-as-text` fix

Same setup as `v27-football-think-base-s120/`, but with PR #80 commit `1d54373` (auto-drop fix). v27 vs v29 isolates the effect of keeping the schema text in the prompt while dropping only the vLLM guided-decoding tensor.

## Setup

| field | value |
|---|---|
| run_id | `bf320613-e724-4382-9c62-7e77a12c114f` |
| batch_id | `batch-31b9b075-ab46-47a2-a437-0466eba9809d` |
| dataset | `twelvelabs/sme_eval_v3.1_fast` |
| config | `H16_FOOTBALL_r1` |
| modelPath | `s3://tl-data-training-pegasus-us-west-2/checkpoints/jeongyeon-nam/rl_rl_v1.5g_wo_a0a1_vc-consol_8k_from_qwen27b_alpha1_think-base/global_step_120/actor/huggingface/` |
| chat_template_kwargs | `{"enable_thinking": true, "supports_thinking": true}` |
| worker image | `lia-vllm-chat-template-kwargs-1d54373` |

## What the fix does

PR #80 originally dropped the entire `structured_output_schema` tensor on the v2 binary path when `chat_template_kwargs.enable_thinking` was truthy, to prevent xgrammar from constraining the `<think>` block. That dropped two separate things at once:

1. **Schema-as-text in the user message** — the chat template (`chat_template.jinja` line 35-37) renders any `json_schema` content item as ```` ```json {schema} ``` ```` + "Important: ..." instructions. This is the exact training-prompt pattern the RL checkpoints learned from.
2. **vLLM `StructuredOutputsParams` guided decoding** — token-level JSON constraint.

We only needed to drop (2). v27 dropped both → model had no idea what structure to emit. Fix:

- `services/pipeline/tasks/shared/tasks/vllm_common.py`: always forward the schema tensor.
- `services/pipeline/model-workers/shared/model_worker/vllm_video/server.py:_build_sampling_params_for_request`: skip `StructuredOutputsParams` when `request.chat_template_kwargs.enable_thinking` is truthy; schema-as-text still flows into the prompt via `build_llm_inputs(json_schema=request.structured_output_schema)`.

## v27 vs v29 diff (same checkpoint, same query, same video)

| Property | v27 (pre-fix) | v29 (post-fix) |
|---|---|---|
| schema text in prompt | ✗ (dropped) | ✓ (chat template renders it) |
| vLLM guided decoding | ✗ (dropped) | ✗ (dropped — correct) |
| input_tokens (sample 0) | 136,694 | 137,090 (+396 for schema text + Important: ...) |
| reasoning in output | ✓ (~300-16384 tokens) | ✓ (more structured, 492-3743 tokens) |
| ```` ```json ```` code block | 0/10 | **10/10** |
| schema-conformant fields (start_time, end_time, etc.) | 0/10 | **10/10** |
| `</think>` marker in output | 0/10 | 0/10 |
| `think_blocks` populated | 0/10 | 0/10 |

## Findings

- **Fix works**: the model recovered the training output pattern — detailed reasoning followed by a ```` ```json ```` code block that matches the schema (start_time, end_time, player_name, play_result, etc.).
- **`</think>` still absent**: the RL checkpoint at step 120 has learned the schema-driven JSON emission but has not yet learned to emit the literal `</think>` marker. This is a model-training matter, not a plumbing matter. Training rollouts at step 120 (see `rollout_logs/120.jsonl`) *do* contain `</think>` in the rewarded outputs, so further training (or reward-shaping on marker emission) should close that gap.
- **Occasional duplicate `\`\`\`json` block** at end of output (model emits the same JSON twice with different indent). Worth a follow-up but not blocking.

## Files

- `tasks.json` — per-sample mapping: request_key → sample_id, video, segment_definition, metadata_schema.
- `<run_id>.<run_id>.<request_key>.json` × 10 — raw worker output JSON (text, input_tokens, output_tokens, finish_reason, think_blocks).
- `summary.md` — quick table.

## How to read

```bash
KEY=$(jq -r '.[0].request_key' tasks.json)
jq -r '.text' < *.${KEY}*.json
jq '.[] | select(.request_key | startswith($k))' --arg k "$KEY" tasks.json
```

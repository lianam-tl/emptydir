# thinking_token_budget PoC — RESULT

**Outcome: PASS** ✅

`thinking_token_budget` enforced `</think>` insertion at the requested token count on the lia-test-27b worker with PR #80's `supports_thinking` prompt-prepend path.

## Setup

- worker image: `476114115052.dkr.ecr.us-west-2.amazonaws.com/tl-data-training-pegasus-vllm-video:lia-vllm-chat-template-kwargs-a39bd70`
- worker spec: `09-infracontroller-llm-worker-thinkbudget.json`
- engine args added:
  - `EXTRA_ARGS=--guided-decoding-backend guidance --reasoning-parser qwen3`
  - `VLLM_REASONING_CONFIG={"reasoning_start_str":"<think>","reasoning_end_str":"</think>"}` (env var; server.py reads it because `--reasoning-config` JSON cannot survive unquoted `${EXTRA_ARGS}` expansion)
- video: `s3://tl-data-training-pegasus-us-west-2/raw_media/private/tl_gemini_sports_soccer_h16/00276801_9min_01687_02244.mp4` decoded to `[557, 224, 448, 3]` UINT8 at fps=1
- prompt: `Find every foul in the video.`
- chat_template_kwargs: `{enable_thinking: true, supports_thinking: true}`

## Result

| budget | completion_tokens | finish_reason | `</think>` in text | char_idx of `</think>` | text length |
|---|---|---|---|---|---|
| None (baseline) | **4096** | `length` | False | — | 0 |
| 10 | 428 | `stop` | **True** | 1145 | 1279 |
| 100 | 508 | `stop` | **True** | 984 | 1198 |

The baseline reproduces JOURNEY.md Run 3 (`completion_tokens=4096`, no `</think>`). With a budget, vLLM forces `</think>` after the configured reasoning-token count, and the model then emits a normal answer (the foul analysis).

Note: the worker streams the prepended `<think>` block over SSE delta.content too, so `text` in the result contains the reasoning + `</think>` + the answer concatenated. In the eval-service path this gets split into `think_blocks` vs `text` server-side; here we collect the raw stream.

## Implication

Production rollout to enable per-request thinking budget would need three things:

1. **xplatform commits** `c003840` + `fd8f11d` + `a39bd70` on branch `lia/vllm-chat-template-kwargs` (server.py: CLI flag + engine kwarg + env-var fallback + SamplingParams plumb + chat-completions parse + GenerateRequest field).
2. **worker env** must include both `EXTRA_ARGS=...--reasoning-parser qwen3` and `VLLM_REASONING_CONFIG={...}`.
3. **clients** that want to use it can pass top-level `thinking_token_budget: N` in the chat completions body.

v2 binary infer path is **not** wired. If batch worker pipeline ever needs this, a separate PR has to plumb the field through `_build_vllm_transport_request` → `_build_generate_request_from_v2`.

## Files

- `10-test-think-budget.py` — sweep script
- `11-soccer-sweep-result.json` — raw JSON of the 3-run sweep above
- `10-RUNBOOK-think-budget.md` — deploy + run instructions

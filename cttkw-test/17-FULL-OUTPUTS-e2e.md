# E2E orchestrator test — budget + schema + think_blocks all applied

Job submitted via `POST /jobs` to orchestrator (port-forward) with the soccer mp4 + `chat_template_kwargs=enable_thinking+supports_thinking` + `thinking_token_budget=50` + `metadata_schema` in params. Spec `1.1.6-thinkbudget` (activated at speccenter). Worker `lia-vllm-chat-template-kwargs-87edc21`, wf-engine `lia-vllm-chat-template-kwargs-87edc21`. Path = v1 chat completions (`use_chat_completions=true`).

S3 output: `s3://tl-data-training-pegasus-us-west-2/vllm-direct/b023ea5af0f93eaeee4375b966d282bc9ff284c40f04e26398642477fad09682.json`

## Result

| metric | value |
|---|---|
| input_tokens | 118749 |
| output_tokens | 252 |
| finish_reason | `stop` |
| **think_blocks** | **1 item, 157 chars** (separate top-level field) |
| **text starts with `{`** | **True** |
| **text parses as JSON** | **True** |

## think_blocks[0]

```text
The user wants to identify fouls in the video.

1.  **First Foul (00:00 - 00:15):** At the very beginning, a Manchester United player (Ibrahimovic) is fouled
```

## text

```json
{
  "results": [
    {
      "end_time": 15.0,
      "fouls": [
        {
          "description": "A Manchester United player is fouled by a Liverpool player, leading to a free kick.",
          "end": "00:15",
          "start": "00:00"
        }
      ],
      "start_time": 0.0
    },
    {
      "end_time": 106.0,
      "fouls": [
        {
          "description": "A Liverpool player is fouled by a Manchester United player, resulting in a free kick.",
          "end": "01:46",
          "start": "01:28"
        }
      ],
      "start_time": 88.0
    }
  ]
}
```

## What this proves

Every link in the chain works:

1. **orchestrator** validated `thinking_token_budget=50` against spec `1.1.6-thinkbudget` (params_schema accepts the int)
2. **wf-engine** (`workflow-engine:lia-vllm-chat-template-kwargs-87edc21`) resolved init_args incl. `thinking_token_budget` and called `vllm_infer_direct(..., thinking_token_budget=50)` — fn signature accepted it (no silent drop)
3. **wf-engine's vllm_common.py** built a chat completions body that included `thinking_token_budget` AND `response_format` (schema gate dropped: client no longer auto-drops response_format under enable_thinking)
4. **worker** (`vllm-video:lia-vllm-chat-template-kwargs-87edc21`) routed through `_build_sampling_params_for_request` — passed both `thinking_token_budget` and `structured_outputs=StructuredOutputsParams(...)` to vLLM SamplingParams (gate also dropped on worker side when `--reasoning-parser qwen3` is on)
5. **vLLM** enforced `</think>` after the budget then applied guided decoding for the JSON-schema part
6. **worker** then ran PR #80's `_extract_visible_text_and_think_blocks` (assistant_think_prefix=True from `supports_thinking`) — partitioned on the first `</think>`, placed reasoning in `think_blocks[0]` and JSON in `text`
7. **wf-engine** uploaded the result dict to S3 with both fields intact

## Caveat (v2 path)

This test used `use_chat_completions=true` so it went through the v1 chat completions HTTP path. The v2 binary infer plumbing in commit `87edc21` (tensor `thinking_token_budget` BYTES alongside `chat_template_kwargs`) is code-ready but was not exercised in this run. To test v2 binary path e2e would require either:
- a kserve LLMInferenceService route that exposes `/v2/models/.../infer` (JOURNEY v11 confirmed the current route 404s)
- or a batch worker setup that the workflow goes through

The schema gate and thinking_token_budget handling in `_build_sampling_params_for_request` is shared between both paths, so any caller passing `chat_template_kwargs` + `thinking_token_budget` + `structured_output_schema` via either transport will get the same behavior once the request reaches the worker.

# v37 — final e2e run after the `</think>` strip fix

Same setup as v22/v25/v26 (`sme_eval_v3.1_fast` 10 samples, no config override → mixed subsets) but with the post-fix worker image `lia-vllm-chat-template-kwargs-7d0418d`. This is the run that confirms PR #80 ends in a clean state: every sample has a populated `think_blocks` *and* a `text` field that starts with a JSON code block, with no `</think>` artifact leaking through.

## Setup

| field | value |
|---|---|
| run_id | `48b9364a-6dd9-4664-b621-80c90480d7d0` |
| batch_id | `batch-dae1289f-ea40-40f1-b8d7-0671296728bd` |
| dataset | `twelvelabs/sme_eval_v3.1_fast` |
| config | (none — mixed subsets) |
| modelPath | `s3://tl-data-training-pegasus-us-west-2/checkpoints/jeongyeon-nam/rl_rl_v1.5g_wo_a0a1_vc-consol_8k_from_qwen27b_alpha1_think-base/global_step_120/actor/huggingface/` |
| chat_template_kwargs | `{"enable_thinking": true, "supports_thinking": true}` |
| worker image | `lia-vllm-chat-template-kwargs-7d0418d` |

## Result

```
in=9166  out=4832 tb=1 </th>=False clean_json=True fin=stop
in=9377  out=5482 tb=1 </th>=False clean_json=True fin=stop
in=10842 out=4125 tb=1 </th>=False clean_json=True fin=stop
in=8707  out=7140 tb=1 </th>=False clean_json=True fin=stop
in=25479 out=6371 tb=1 </th>=False clean_json=True fin=stop
in=9288  out=3925 tb=1 </th>=False clean_json=True fin=stop
in=25618 out=3986 tb=1 </th>=False clean_json=True fin=stop
in=7801  out=1788 tb=1 </th>=False clean_json=True fin=stop
in=11081 out=3006 tb=1 </th>=False clean_json=True fin=stop
in=37358 out=2795 tb=1 </th>=False clean_json=True fin=stop

summary: 10 samples, 10 populated tb, 10 clean json starts
```

- **10/10 `think_blocks` populated** with the full reasoning trace.
- **10/10 `text` starts cleanly with ```` ```json ``` ```` / `{`** — no reasoning prefix, no `</think>` artifact.
- **10/10 `finish_reason=stop`** — no length truncation.
- Input tokens vary 7-37k (mixed subsets: short A0_OTHERS, longer H16_*); output tokens 1.8-7k (think mode → longer than schema-only output).

## What this proves

End-to-end the PR works as documented:

1. **Plumbing**: `chatTemplateKwargs` flows eval-service → batch-request → orchestrator → wf-engine → worker → `apply_chat_template`.
2. **Schema-as-text**: chat template's `json_schema` content item renders the schema as text in the user message; guided decoding is skipped because `enable_thinking` is truthy. Model has the structural cue it was trained against, so JSON emission is recovered.
3. **`</think>` extraction**: `assistant_think_prefix=True` partition logic captures the reasoning between the prompt-side `<think>\n` prefix and the model-emitted `</think>`; `text` contains only the final JSON answer; `think_blocks` carries the reasoning trace.
4. **Persistence**: `think_blocks` lands in the orchestrator's `output_url` JSON for downstream consumers.

This is the run referenced in the PR description's Verification §B + §D.

## Files

- `tasks.json` — per-sample mapping (request_key → sample_id, video, segment_definition, metadata_schema).
- `<run_id>.<run_id>.<request_key>.json` × 10 — raw worker output JSON.

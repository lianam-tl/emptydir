# IFStruct v1.0 snapshot and notes

This directory preserves the public IFStruct repository and its frozen benchmark data for later Pegasus evaluation work.

- Upstream repository: https://github.com/Liquid4All/ifstruct
- Benchmark dataset: https://huggingface.co/datasets/LiquidAI/ifstruct-v1.0
- Technical post: https://www.liquid.ai/blog/ifstruct-v1.0
- Snapshot commit: `81dbaf26eddf20a0e36038a0a5b139bad4765eda` (2026-06-30)
- License: Apache-2.0; see `upstream/LICENSE`

`upstream/` is an unmodified copy of that commit, without its nested `.git` directory.

## What the benchmark measures

IFStruct is a text-only benchmark for structured-output instruction following. A model receives a text request describing a target structure and must return JSON or YAML. It does not contain image, video, or audio inputs.

The evaluator is deterministic and rule-based. It does not use an LLM judge. For each response, it checks:

1. Whether the response parses as the requested JSON or YAML format.
2. Whether a requested Markdown code block is present.
3. Whether forbidden commentary appears outside the payload.
4. Whether the top level is a bare list or a list under the exact requested wrapper key.
5. Whether the list contains the requested number or range of items.
6. Whether nested fields satisfy the supplied schema: required fields, primitive types, enum membership, numeric bounds, nested-list lengths, and absence of unexpected fields.

Each sample receives `1` only when every checked rule passes; otherwise it receives `0`. Content accuracy and quality are deliberately not scored.

## Public test-set summary

The public artifact is a frozen test set, not a supervised training dataset. It contains prompts and validation specifications but no reference assistant answers.

| Property | Count |
| --- | ---: |
| Test examples | 2,000 |
| JSON / YAML | 1,000 / 1,000 |
| Entity taxonomies | 24 |
| Bare list / wrapper object | 989 / 1,011 |
| Code block requested | 1,268 |
| No external commentary requested | 998 |
| Escaping-focused examples | 973 |
| Rows with at least one optional schema field | 203 |
| Exact unique schemas | 1,881 |

All underlying schemas have an array at their root. The response may expose that array directly or wrap it under one requested key. Prompts present schemas through natural prose, raw JSON Schema, field tables, annotated examples, path glossaries, and deliberately difficult strings containing quotes, paths, stack traces, or multiline text.

Do not place `upstream/data/test.jsonl` in SFT or RL training data. That would contaminate this public benchmark. Liquid AI's separate generated GRPO training split and generation recipe are not included in the public repository.

## Repository layout

```text
260724_ifstruct/
├── README.md
└── upstream/
    ├── data/test.jsonl          # frozen 2,000-row public test set
    ├── ifstruct/validator.py    # deterministic parsing and validation rules
    ├── ifstruct/eval.py         # concurrent OpenAI-compatible evaluation CLI
    ├── ifstruct/client.py       # chat-completions HTTP client
    ├── tests/test_validator.py
    └── LICENSE
```

The reusable rule-based implementation is primarily `upstream/ifstruct/validator.py`. `validate_response(...)` accepts a model response and the row's schema/format requirements, then returns `ValidationResult` with binary pass/fail, errors, and diagnostic details.

## Running the upstream tests

From this directory:

```bash
cd upstream
uv run --with pytest pytest -q
```

Running the full benchmark requires an OpenAI-compatible chat-completions endpoint:

```bash
cd upstream
BASE_URL=http://your-endpoint/v1 \
API_KEY=your-key \
uv run ifstruct-eval \
  --model your-model \
  --dataset data/test.jsonl \
  --results-file results/latest.json \
  --temperature 0 \
  --n-threads 32
```

This sends only a single text user message. It does not pass the schema through an API-level guided-decoding or `response_format` parameter, so the score measures the model's native structured-output behavior.

## Audit findings and limitations

The copied source is intentionally unchanged, but one evaluator issue should be fixed before treating scores as authoritative:

- `require_code_block=False` does not reject a response that uses a code block, even when the prompt explicitly forbids one.
- When a code block is required, generic or incorrectly labelled fences such as ```` ```python ```` can pass if their contents parse.

The validator only checks the positive condition `require_code_block and not uses_code_block`. This weakens the claimed code-fence instruction check but does not invalidate the core JSON/YAML schema checks.

The benchmark also does not validate semantic correctness or many advanced JSON Schema features, including regex patterns, date/email formats, string lengths, uniqueness, conditional dependencies, and duplicate keys. Structurally valid but meaningless content can therefore pass.

The complete public test set and its specifications are available to anyone, so future models can also be contaminated by it. Treat the result as one diagnostic rather than a definitive model-quality score.

## Pegasus relevance

IFStruct can diagnose the text decoder's ability to obey changing schemas, including required versus optional fields, wrapper keys, value types, enum constraints, and unexpected-field suppression. It does not evaluate video understanding, temporal grounding, or whether generated descriptions are correct.

For Pegasus, the useful comparison would be:

1. Native Qwen3.5-27B generation without guided decoding, measuring learned schema following.
2. Production generation with guided decoding, measuring actual serving reliability.
3. A separate Pegasus-specific paired test where the same field is alternately required, optional, or absent from the schema.

Only 203 public IFStruct examples contain optional fields, so the official benchmark alone is not a sufficient evaluation of optional-schema robustness.

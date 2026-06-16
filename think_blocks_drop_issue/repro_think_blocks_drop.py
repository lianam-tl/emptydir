"""Standalone reproduction of the eval-service ``think_blocks`` drop + the fix.

Self-contained: it inlines the BEFORE behavior of
``eval/eval-service/eval_service/prediction/response_normalizer.py`` and the
AFTER ``extract_think_blocks`` helper, then asserts the observed regression and
the fix on a realistic orchestrator output payload.

Run:  python repro_think_blocks_drop.py
"""

import json
from typing import Any


# --- copied from response_normalizer.py (unchanged by the fix) ----------------
def _load_jsonish(value: str) -> Any:
    stripped = value.strip()
    if not stripped:
        return value
    try:
        return json.loads(stripped)
    except (json.JSONDecodeError, TypeError):
        return value


def normalize_response_for_eval(response: Any) -> Any:
    """Unwrap stored response payloads into the evaluator's expected format."""
    current = response
    while True:
        if isinstance(current, str):
            parsed = _load_jsonish(current)
            if parsed is not current:
                current = parsed
                continue
            return current
        if isinstance(current, dict):
            if "results" in current:
                current = current["results"]
                continue
            text_value = current.get("text")
            if isinstance(text_value, str):
                parsed_text = _load_jsonish(text_value)
                if parsed_text is not text_value:
                    current = parsed_text
                    continue
        return current


# --- the fix: a new sibling-extraction helper ---------------------------------
def extract_think_blocks(output: Any) -> Any:
    """Pull ``think_blocks`` off a raw output payload before it is unwrapped."""
    current = output
    if isinstance(current, str):
        current = _load_jsonish(current)
    if isinstance(current, dict):
        think_blocks = current.get("think_blocks")
        if think_blocks:
            return think_blocks
    return None


def main() -> None:
    think_blocks = [{"type": "thinking", "thinking": "let me reason about the scenes..."}]
    orchestrator_output = {
        "request_id": "req-1",
        "text": json.dumps({"results": [{"start_time": 3.0, "end_time": 4.0}]}),
        "think_blocks": think_blocks,
        "finish_reason": "stop",
        "video_frames": [],
    }

    # BEFORE: normalization unwraps to just the answer; think_blocks is gone.
    response = normalize_response_for_eval(orchestrator_output)
    print("normalized response :", response)
    assert response == [{"start_time": 3.0, "end_time": 4.0}]
    assert not (isinstance(response, dict) and "think_blocks" in response), (
        "regression: think_blocks survived inside response (only happened on parse failure)"
    )
    print("=> think_blocks DROPPED by the normalizer (matches the 18,705-sample finding)")

    # AFTER: extract_think_blocks recovers them so callers can store them
    # as a top-level sibling field in predictions.jsonl.
    recovered = extract_think_blocks(orchestrator_output)
    print("extracted think_blocks:", recovered)
    assert recovered == think_blocks

    prediction_record = {"response": response, "think_blocks": recovered}
    print("predictions.jsonl row :", json.dumps(prediction_record))
    assert prediction_record["think_blocks"] == think_blocks

    # extract returns None when there is nothing to preserve
    assert extract_think_blocks({"text": "answer"}) is None
    assert extract_think_blocks({"text": "answer", "think_blocks": []}) is None
    assert extract_think_blocks("plain text") is None
    assert extract_think_blocks(None) is None

    print("\nALL ASSERTIONS PASSED")


if __name__ == "__main__":
    main()

"""[cc-generated] PR-snippet style PoC: verify thinking_token_budget enforces </think>.

Sweeps thinking_token_budget across [None, 10, 100] against the chat completions
endpoint of the lia-test-27b worker (image lia-vllm-chat-template-kwargs-c003840
with --reasoning-parser qwen3). Prints completion_tokens + presence/position of
</think> in the output text for each run.

Setup (in another shell):

    kubectl -n kserve-models port-forward svc/lia-test-27b-predictor-default 18093:80 &
    # or whichever kserve isvc service exposes lia-test-27b

Usage:

    python 10-test-think-budget.py \
        --host http://localhost:18093 \
        --video-npy /path/to/sports_soccer.npy \
        --video-fps 2.5 \
        --start 0.0 \
        --end 4.0
"""

from __future__ import annotations

import argparse
import base64
import json
import time
from typing import Any

import numpy as np
import requests

PROMPT_TEXT = "Find every foul in the video."
BUDGETS: list[int | None] = [None, 10, 100]


def load_or_synth_video(video_npy: str | None, synthetic_shape: str | None) -> np.ndarray:
    if video_npy:
        arr = np.load(video_npy)
        if arr.dtype != np.uint8 or arr.ndim != 4:
            raise ValueError(f"expected UINT8 [T,H,W,C] tensor, got dtype={arr.dtype} shape={arr.shape}")
        return arr
    if not synthetic_shape:
        raise ValueError("either --video-npy or --synthetic-shape T,H,W is required")
    t, h, w = [int(x) for x in synthetic_shape.split(",")]
    rng = np.random.default_rng(42)
    return rng.integers(0, 256, size=(t, h, w, 3), dtype=np.uint8)


def build_body(
    arr: np.ndarray,
    video_fps: float,
    start: float,
    end: float,
    thinking_token_budget: int | None,
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": "vllm-video",
        "stream": True,
        "max_tokens": 4096,
        "temperature": 0.0,
        "chat_template_kwargs": {"enable_thinking": True, "supports_thinking": True},
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "x-tl-video-tensor",
                        "tensor_b64": base64.b64encode(arr.tobytes()).decode(),
                        "shape": list(arr.shape),
                        "dtype": "uint8",
                        "fps": video_fps,
                        "start": start,
                        "end": end,
                    },
                    {"type": "text", "text": PROMPT_TEXT},
                ],
            }
        ],
    }
    if thinking_token_budget is not None:
        body["thinking_token_budget"] = thinking_token_budget
    return body


def run(host: str, body: dict[str, Any]) -> dict[str, Any]:
    t0 = time.time()
    resp = requests.post(
        f"{host}/v1/chat/completions", json=body, stream=True, timeout=600
    )
    resp.raise_for_status()
    text_parts: list[str] = []
    last_usage: dict[str, Any] | None = None
    finish_reason: str | None = None
    for raw in resp.iter_lines():
        if not raw.startswith(b"data: "):
            continue
        data = raw[len(b"data: ") :]
        if data == b"[DONE]":
            break
        chunk = json.loads(data)
        choice = chunk["choices"][0]
        delta = choice.get("delta") or {}
        if "content" in delta and delta["content"]:
            text_parts.append(delta["content"])
        if choice.get("finish_reason"):
            finish_reason = choice["finish_reason"]
        if chunk.get("usage"):
            last_usage = chunk["usage"]
    text = "".join(text_parts)
    return {
        "elapsed_sec": round(time.time() - t0, 1),
        "finish_reason": finish_reason,
        "usage": last_usage,
        "text": text,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="http://localhost:18093")
    p.add_argument("--video-npy", default=None, help="pre-decoded UINT8 [T,H,W,3] tensor")
    p.add_argument(
        "--synthetic-shape",
        default=None,
        help="comma-sep T,H,W to generate a random UINT8 tensor (e.g. 16,224,224). Smoke-test only.",
    )
    p.add_argument("--video-fps", type=float, default=2.5)
    p.add_argument("--start", type=float, default=0.0)
    p.add_argument("--end", type=float, default=4.0)
    p.add_argument("--out", default=None, help="optional path to dump full JSON results")
    args = p.parse_args()

    arr = load_or_synth_video(args.video_npy, args.synthetic_shape)
    print(f"[video] shape={arr.shape} dtype={arr.dtype} (synthetic={args.video_npy is None})")

    results: dict[str, Any] = {}
    for budget in BUDGETS:
        label = f"budget={budget}" if budget is not None else "budget=None"
        print(f"\n=== {label} ===")
        body = build_body(arr, args.video_fps, args.start, args.end, budget)
        result = run(args.host, body)
        usage = result["usage"] or {}
        text = result["text"]
        close_idx = text.find("</think>")
        print(
            f"  elapsed={result['elapsed_sec']}s "
            f"finish={result['finish_reason']} "
            f"prompt_tokens={usage.get('prompt_tokens')} "
            f"completion_tokens={usage.get('completion_tokens')}"
        )
        print(
            f"  text_chars={len(text)} "
            f"</think>_present={close_idx >= 0} "
            f"</think>_char_idx={close_idx}"
        )
        print(f"  text_head: {text[:200]!r}")
        results[label] = result

    if args.out:
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nWrote {args.out}")

    print("\n=========== COMPARE ===========")
    for label, r in results.items():
        u = r["usage"] or {}
        close = "</think>" in r["text"]
        print(
            f"{label:<18} len={len(r['text']):>6} </think>={close} "
            f"completion={u.get('completion_tokens')} finish={r['finish_reason']}"
        )


if __name__ == "__main__":
    main()

"""[cc-generated] Smoke test for vllm-direct n>1 (xplatform A-1147 / lia/A-1147-vllm-direct-n-rollouts).

Sends a single V2 binary infer request to a port-forwarded vllm-video pod with a
configurable `n` parameter, then asserts:
  - n=1: `text` is a plain string
  - n>1: `text` is a JSON-serialized array of length n

The video tensor is synthetic (zeros) — we only care about the shape of the
response, not generation quality.

Usage:
    # Terminal 1 — port-forward the pod with the new image
    kubectl port-forward -n pegasus-platform <pod> 8000:8000

    # Terminal 2
    python smoke_test_a1147_n_rollouts.py --n 4 --temperature 0.7
"""

from __future__ import annotations

import argparse
import json
import struct
import sys

import numpy as np
import requests


def encode_bytes_tensor(values: list[bytes]) -> bytes:
    """KServe V2 BYTES tensor wire format: 4-byte little-endian length prefix per element."""
    return b"".join(struct.pack("<I", len(v)) + v for v in values)


def decode_bytes_tensor(buf: bytes) -> str:
    """Inverse of encode_bytes_tensor for a single-element BYTES tensor."""
    n_bytes = struct.unpack_from("<I", buf, 0)[0]
    return buf[4 : 4 + n_bytes].decode("utf-8")


def build_v2_request(
    *,
    video: np.ndarray,
    prompt: str,
    n: int,
    temperature: float,
    max_tokens: int,
    fps: float,
) -> tuple[dict[str, str], bytes]:
    video_bytes = video.tobytes()
    prompt_tensor_bytes = encode_bytes_tensor([prompt.encode("utf-8")])
    header = {
        "id": "a1147-smoke-test",
        "parameters": {
            "video_fps": fps,
            "start": 0.0,
            "end": float(video.shape[0]) / fps,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "n": n,
        },
        "inputs": [
            {
                "name": "video",
                "shape": list(video.shape),
                "datatype": "UINT8",
                "parameters": {"binary_data_size": len(video_bytes)},
            },
            {
                "name": "prompt",
                "shape": [1],
                "datatype": "BYTES",
                "parameters": {"binary_data_size": len(prompt_tensor_bytes)},
            },
        ],
        "outputs": [
            {"name": "text", "parameters": {"binary_data": True}},
            {"name": "request_id", "parameters": {"binary_data": True}},
            {"name": "video_frames", "parameters": {"binary_data": True}},
        ],
    }
    header_json = json.dumps(header).encode("utf-8")
    body = header_json + video_bytes + prompt_tensor_bytes
    headers = {
        "Inference-Header-Content-Length": str(len(header_json)),
        "Content-Type": "application/octet-stream",
    }
    return headers, body


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--model", default="vllm-video")
    parser.add_argument("--n", type=int, default=4)
    parser.add_argument("--prompt", default="Describe what you see in this video.")
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help=">0 needed to get diverse n outputs (n=8 with temp=0 = 8 identical strings)",
    )
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--frames", type=int, default=4)
    parser.add_argument("--fps", type=float, default=2.0)
    parser.add_argument("--timeout", type=float, default=300.0)
    args = parser.parse_args()

    video = np.zeros((args.frames, 64, 64, 3), dtype=np.uint8)
    headers, body = build_v2_request(
        video=video,
        prompt=args.prompt,
        n=args.n,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        fps=args.fps,
    )
    url = f"{args.url}/v2/models/{args.model}/infer"
    print(f"POST {url}  n={args.n}  temperature={args.temperature}  frames={args.frames}")

    resp = requests.post(url, headers=headers, data=body, timeout=args.timeout)
    if resp.status_code != 200:
        print(f"HTTP {resp.status_code}: {resp.text[:1000]}", file=sys.stderr)
        return 1

    resp_header_len = int(resp.headers.get("Inference-Header-Content-Length", "0"))
    if resp_header_len <= 0:
        print(
            "FAIL: response missing Inference-Header-Content-Length",
            file=sys.stderr,
        )
        return 2
    resp_header = json.loads(resp.content[:resp_header_len])
    payload = resp.content[resp_header_len:]

    offset = 0
    text_output: str | None = None
    for out in resp_header["outputs"]:
        size = out["parameters"]["binary_data_size"]
        chunk = payload[offset : offset + size]
        if out["name"] == "text":
            # vllm_video returns `text` as a UINT8 tensor (raw utf-8 bytes), not
            # as a KServe BYTES tensor (which would have a 4-byte length prefix).
            if out.get("datatype") == "UINT8":
                text_output = chunk.decode("utf-8")
            else:
                text_output = decode_bytes_tensor(chunk)
        offset += size

    if text_output is None:
        print("FAIL: no 'text' output in response", file=sys.stderr)
        return 3

    preview = text_output[:300].replace("\n", " ")
    print(f"text[:300] = {preview}{'...' if len(text_output) > 300 else ''}")

    if args.n == 1:
        try:
            parsed = json.loads(text_output)
        except json.JSONDecodeError:
            print(f"PASS: n=1 returned a single string ({len(text_output)} chars)")
            return 0
        if isinstance(parsed, list):
            print(
                f"FAIL: n=1 should return a string, got JSON list of {len(parsed)} items "
                "(regression of n=1 contract)",
                file=sys.stderr,
            )
            return 4
        print(f"PASS: n=1 returned a single (non-list) value")
        return 0

    try:
        parsed = json.loads(text_output)
    except json.JSONDecodeError as exc:
        print(f"FAIL: n={args.n} but text isn't JSON: {exc}", file=sys.stderr)
        return 5
    if not isinstance(parsed, list):
        print(
            f"FAIL: n={args.n} but text decoded to {type(parsed).__name__}, not list",
            file=sys.stderr,
        )
        return 6
    if len(parsed) != args.n:
        print(
            f"FAIL: requested n={args.n} but got {len(parsed)} items",
            file=sys.stderr,
        )
        return 7
    print(f"PASS: n={args.n} returned a JSON array of {len(parsed)} strings")
    for i, t in enumerate(parsed):
        preview_i = t[:80].replace("\n", " ")
        print(f"  [{i}] {preview_i}{'...' if len(t) > 80 else ''}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

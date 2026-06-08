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
import os
import struct
import subprocess
import sys
import tempfile
from urllib.parse import urlparse

import numpy as np
import requests


def _load_video_to_numpy(
    video_arg: str,
    fps: float,
    duration: float | None,
    max_side: int | None,
) -> tuple[np.ndarray, dict]:
    """Resolve --video (local path or s3://...) to a [T,H,W,3] uint8 ndarray.

    ffprobe gives original WxH for the shape calc; ffmpeg does the actual
    decode at the requested fps and (optional) duration / scale.
    """
    if video_arg.startswith("s3://"):
        local = os.path.join(tempfile.gettempdir(), "a1147_" + os.path.basename(urlparse(video_arg).path))
        if not os.path.exists(local) or os.path.getsize(local) == 0:
            # boto3 honors AWS SSO sessions; s5cmd does not on Mac in the common case.
            import boto3  # local import so users without boto3 can still use --video <local-path>
            parsed = urlparse(video_arg)
            bucket = parsed.netloc
            key = parsed.path.lstrip("/")
            profile = os.environ.get("AWS_PROFILE", "training")
            print(f"[load] boto3 download s3://{bucket}/{key} -> {local} (profile={profile})")
            session = boto3.Session(profile_name=profile)
            session.client("s3").download_file(bucket, key, local)
        video_arg = local

    probe = subprocess.check_output([
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height,r_frame_rate,duration",
        "-of", "json",
        video_arg,
    ])
    pmeta = json.loads(probe)["streams"][0]
    orig_w, orig_h = int(pmeta["width"]), int(pmeta["height"])

    # Decide output W,H
    out_w, out_h = orig_w, orig_h
    if max_side and max(orig_w, orig_h) > max_side:
        if orig_w >= orig_h:
            out_w = max_side
            out_h = int(round(orig_h * max_side / orig_w))
        else:
            out_h = max_side
            out_w = int(round(orig_w * max_side / orig_h))
        # Qwen3VL needs dims divisible by patch_size * merge_size = 16 * 2 = 32.
        # Otherwise the processor's reshape into spatial patch grids fails with
        # `RuntimeError: shape '[...]' is invalid for input of size N`.
        out_w -= out_w % 32
        out_h -= out_h % 32

    vf = f"fps={fps},scale={out_w}:{out_h}"
    args_ = ["ffmpeg", "-v", "error", "-i", video_arg, "-vf", vf]
    if duration is not None:
        args_ += ["-t", str(duration)]
    args_ += ["-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    print(f"[load] ffmpeg {' '.join(args_[3:])}")
    raw = subprocess.check_output(args_)
    frame_bytes = out_w * out_h * 3
    if len(raw) % frame_bytes != 0:
        raise RuntimeError(
            f"ffmpeg returned {len(raw)} bytes, not a multiple of WxHx3={frame_bytes}"
        )
    n_frames = len(raw) // frame_bytes
    video = np.frombuffer(raw, dtype=np.uint8).reshape(n_frames, out_h, out_w, 3).copy()
    return video, {"orig_w": orig_w, "orig_h": orig_h, "out_w": out_w, "out_h": out_h,
                   "n_frames": n_frames, "fps": fps}


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
    parser.add_argument("--frames", type=int, default=4,
                        help="Frame count for the synthetic (zeros) fallback. Ignored when --video is given.")
    parser.add_argument("--fps", type=float, default=2.0)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--video",
                        help="s3://... or local path. Loads real frames via ffmpeg instead of zeros.")
    parser.add_argument("--duration", type=float, default=10.0,
                        help="Seconds of video to decode (from start). Default 10s.")
    parser.add_argument("--max-side", type=int, default=512,
                        help="Resize so max(W,H) <= this. Default 512 to keep request size small.")
    args = parser.parse_args()

    if args.video:
        video, meta = _load_video_to_numpy(args.video, args.fps, args.duration, args.max_side)
        print(f"[load] decoded {meta['n_frames']} frames at "
              f"{meta['out_w']}x{meta['out_h']} (orig {meta['orig_w']}x{meta['orig_h']}), fps={meta['fps']}")
    else:
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

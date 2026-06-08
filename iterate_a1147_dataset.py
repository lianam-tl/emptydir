"""[cc-generated] Iterate an HF dataset, send each (video, prompt) to the vllm-video
worker with n>1, save the n completions to <output-dir>/<row-id>.json.

Smoke-test driver for xplatform A-1147 (vllm-direct n>1). Assumes:
  - worker pod is port-forwarded to localhost:18000 (or wherever --worker-url points)
  - HF_TOKEN is exported (or in /Users/long8v/pegasus/.env)
  - AWS_PROFILE=training works for boto3 to download videos from S3

Usage:
  ~/.venv/bin/python ~/emptydir/iterate_a1147_dataset.py \
    --hf-dataset twelvelabs/tl_soccer_h16_sme_tdf \
    --hf-config H16_SOCCER \
    --limit 5 \
    --n 4 --temperature 0.7 --max-tokens 256 \
    --duration 30 --max-side 384 --fps 2.0

Re-runs are resumable: rows whose output file already exists are skipped.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from smoke_test_a1147_n_rollouts import (
    _load_video_to_numpy,
    build_v2_request,
    decode_bytes_tensor,
)


def _slack(text: str, channel: str = "#fun-lia-trashcan") -> None:
    """Best-effort Slack post; never raises."""
    token = os.environ.get("SLACK_BOT_TOKEN", "")
    if not token:
        return
    try:
        requests.post(
            "https://slack.com/api/chat.postMessage",
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json; charset=utf-8"},
            data=json.dumps({"channel": channel, "text": text}).encode("utf-8"),
            timeout=10,
        )
    except Exception:
        pass


def _extract_video_url(row) -> str:
    """media[0].media_path."""
    media = row["media"]
    return str(media[0]["media_path"])


def _extract_prompt(row) -> str:
    """First text-type content part across messages[0].content."""
    messages = row["messages"]
    content = messages[0]["content"]
    for part in content:
        if isinstance(part, dict) and part.get("type") == "text" and part.get("text"):
            return part["text"]
    raise ValueError("no text content part found in messages[0]")


def _load_hf_parquet_rows(
    dataset: str, config: str, split: str, hf_token: str
) -> pd.DataFrame:
    """Download all parquet shards for a (dataset, config, split) and concat."""
    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi(token=hf_token)
    info = api.dataset_info(dataset)
    pattern = f"{config}/{split}-"
    shards = [s.rfilename for s in info.siblings if s.rfilename.startswith(pattern)]
    if not shards:
        raise ValueError(
            f"no parquet shards matching {pattern!r} found in {dataset}; "
            f"available: {[s.rfilename for s in info.siblings if s.rfilename.endswith('.parquet')]}"
        )
    shards.sort()
    print(f"[hf] loading {len(shards)} shard(s) from {dataset}/{config}/{split}")
    frames = []
    for shard in shards:
        local = hf_hub_download(dataset, shard, repo_type="dataset", token=hf_token)
        frames.append(pd.read_parquet(local))
    df = pd.concat(frames, ignore_index=True)
    print(f"[hf] {len(df)} total rows")
    return df


def _call_worker(
    url: str,
    model: str,
    video,
    prompt: str,
    n: int,
    temperature: float,
    max_tokens: int,
    fps: float,
    timeout: float,
) -> dict:
    """Send a single V2 infer request, return parsed result."""
    headers, body = build_v2_request(
        video=video,
        prompt=prompt,
        n=n,
        temperature=temperature,
        max_tokens=max_tokens,
        fps=fps,
    )
    resp = requests.post(f"{url}/v2/models/{model}/infer", headers=headers, data=body, timeout=timeout)
    if resp.status_code != 200:
        raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:500]}")
    hlen = int(resp.headers["Inference-Header-Content-Length"])
    hdr = json.loads(resp.content[:hlen])
    payload = resp.content[hlen:]

    out: dict = {}
    offset = 0
    for o in hdr["outputs"]:
        size = o["parameters"]["binary_data_size"]
        chunk = payload[offset : offset + size]
        name = o["name"]
        dtype = o.get("datatype")
        if name == "text":
            text = chunk.decode("utf-8") if dtype == "UINT8" else decode_bytes_tensor(chunk)
            if n > 1:
                try:
                    out["completions"] = json.loads(text)
                    if not isinstance(out["completions"], list):
                        raise ValueError("text field decoded to non-list for n>1")
                except Exception as exc:
                    out["completions"] = None
                    out["text_raw"] = text
                    out["parse_error"] = str(exc)
            else:
                out["text"] = text
        elif name in ("video_frames", "output_tokens", "input_tokens", "retry_count"):
            import struct
            out[name] = struct.unpack("<i", chunk)[0]
        elif name in ("worker_elapsed_ms", "vllm_generate_ms"):
            import struct
            out[name] = struct.unpack("<f", chunk)[0]
        elif name == "finish_reason":
            out[name] = chunk.decode("utf-8")
        offset += size
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf-dataset", required=True, help="e.g. twelvelabs/tl_soccer_h16_sme_tdf")
    parser.add_argument("--hf-config", required=True, help="e.g. H16_SOCCER")
    parser.add_argument("--hf-split", default="train")
    parser.add_argument("--worker-url", default="http://localhost:18000")
    parser.add_argument("--model", default="vllm-video")
    parser.add_argument("--output-dir", default="/tmp/a1147_results")
    parser.add_argument("--limit", type=int, default=5, help="Process first N rows only (default 5)")
    parser.add_argument("--start", type=int, default=0, help="Row index to start from")
    parser.add_argument("--n", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--duration", type=float, default=30.0,
                        help="Seconds to decode per video")
    parser.add_argument("--fps", type=float, default=2.0)
    parser.add_argument("--max-side", type=int, default=384)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--slack-every", type=int, default=5)
    args = parser.parse_args()

    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        env_path = "/Users/long8v/pegasus/.env"
        if os.path.exists(env_path):
            with open(env_path) as f:
                for line in f:
                    if line.startswith("HF_TOKEN="):
                        hf_token = line.split("=", 1)[1].strip().strip('"')
                        os.environ["HF_TOKEN"] = hf_token
                        break
    if not hf_token:
        print("ERROR: HF_TOKEN not set", file=sys.stderr)
        return 2

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[iter] output dir: {out_dir}")

    df = _load_hf_parquet_rows(args.hf_dataset, args.hf_config, args.hf_split, hf_token)
    end = min(args.start + args.limit, len(df))
    print(f"[iter] processing rows [{args.start}, {end}) of {len(df)}")
    _slack(f":arrow_forward: [A-1147 iter] starting rows [{args.start},{end}) of {len(df)} on {args.worker_url}, n={args.n}, model={args.model}")

    n_done = 0
    n_skipped = 0
    n_failed = 0
    started_at = time.time()

    for idx in range(args.start, end):
        row = df.iloc[idx]
        row_id = str(row["id"])
        out_file = out_dir / f"{row_id}.json"
        if out_file.exists():
            n_skipped += 1
            print(f"[iter {idx}/{end - 1}] skip (already done) id={row_id[:12]}")
            continue
        try:
            video_url = _extract_video_url(row)
            prompt = _extract_prompt(row)
            print(f"[iter {idx}/{end - 1}] id={row_id[:12]} video={video_url} prompt[:80]={prompt[:80]!r}")
            t0 = time.time()
            video, vmeta = _load_video_to_numpy(video_url, args.fps, args.duration, args.max_side)
            t_decode = time.time() - t0
            t1 = time.time()
            result = _call_worker(
                url=args.worker_url,
                model=args.model,
                video=video,
                prompt=prompt,
                n=args.n,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                fps=args.fps,
                timeout=args.timeout,
            )
            t_infer = time.time() - t1
            payload = {
                "id": row_id,
                "row_index": idx,
                "video_url": video_url,
                "prompt": prompt,
                "n": args.n,
                "temperature": args.temperature,
                "max_tokens": args.max_tokens,
                "decoded_frames": int(vmeta["n_frames"]),
                "input_resolution": [int(vmeta["out_w"]), int(vmeta["out_h"])],
                "decode_seconds": round(t_decode, 2),
                "infer_seconds": round(t_infer, 2),
                "result": result,
            }
            out_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
            n_done += 1
            comps = result.get("completions") or ([result.get("text")] if "text" in result else [])
            preview = (comps[0] if comps else "")[:120].replace("\n", " ")
            print(f"  -> {len(comps)} completion(s) in {t_infer:.1f}s. preview={preview!r}")
        except Exception as exc:
            n_failed += 1
            err_file = out_dir / f"{row_id}.error.json"
            err_file.write_text(
                json.dumps(
                    {"id": row_id, "row_index": idx, "error": str(exc), "traceback": traceback.format_exc()},
                    ensure_ascii=False,
                    indent=2,
                )
            )
            print(f"  -> FAILED: {exc}")

        if (n_done + n_failed) and (n_done + n_failed) % args.slack_every == 0:
            elapsed = time.time() - started_at
            _slack(
                f":hourglass: [A-1147 iter] processed {n_done + n_failed + n_skipped}/{end - args.start}: "
                f"done={n_done}, skipped={n_skipped}, failed={n_failed}, t={elapsed:.0f}s"
            )

    elapsed = time.time() - started_at
    print(f"\n[iter] DONE done={n_done}, skipped={n_skipped}, failed={n_failed}, elapsed={elapsed:.0f}s")
    _slack(
        f":checkered_flag: [A-1147 iter] complete: done={n_done}, skipped={n_skipped}, "
        f"failed={n_failed}, elapsed={elapsed:.0f}s. results at `{out_dir}/`"
    )
    return 0 if n_failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

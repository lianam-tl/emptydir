"""[cc-generated] E2E smoke for PR #80 fixup changes.

Direct chat-completions sweep against lia-test-27b (current sha 8428c34):

  A. Both flags (sanity) → think_blocks populated
  B. enable_thinking only → normalize sets both → thinking happens
  C. supports_thinking only → same as B
  D. tokenize=True (reserved kwarg)  → HTTP 400 (allowlist)
  E. tools=True (unknown kwarg)      → HTTP 400 (allowlist)

Setup (in another shell): kubectl port-forward svc/lia-test-27b-... 18093:80
"""
from __future__ import annotations
import argparse, base64, json, sys, time
import numpy as np
import requests

PROMPT = "Briefly describe what is happening in the video."

def synth(shape="8,160,160"):
    t, h, w = [int(x) for x in shape.split(",")]
    rng = np.random.default_rng(7)
    return rng.integers(0, 256, size=(t, h, w, 3), dtype=np.uint8)

def build_body(arr, fps, start, end, ctk):
    body = {
        "model": "vllm-video",
        "stream": True,
        "max_tokens": 256,
        "temperature": 0.0,
        "messages": [{"role": "user", "content": [
            {"type": "x-tl-video-tensor",
             "tensor_b64": base64.b64encode(arr.tobytes()).decode(),
             "shape": list(arr.shape), "dtype": "uint8",
             "fps": fps, "start": start, "end": end},
            {"type": "text", "text": PROMPT},
        ]}],
    }
    if ctk is not None:
        body["chat_template_kwargs"] = ctk
    return body

def run(host, body):
    t0 = time.time()
    resp = requests.post(f"{host}/v1/chat/completions", json=body, stream=True, timeout=600)
    if resp.status_code >= 400:
        body_text = resp.text[:500]
        return {"status": resp.status_code, "body": body_text, "elapsed": round(time.time()-t0, 1)}
    text_parts, think_blocks, finish, usage = [], [], None, None
    for raw in resp.iter_lines():
        if not raw.startswith(b"data: "): continue
        data = raw[len(b"data: "):]
        if data == b"[DONE]": break
        chunk = json.loads(data)
        choice = chunk["choices"][0]
        delta = choice.get("delta") or {}
        if delta.get("content"):
            text_parts.append(delta["content"])
        if choice.get("finish_reason"):
            finish = choice["finish_reason"]
        if chunk.get("usage"):
            usage = chunk["usage"]
        tb = chunk.get("x_tl_think_blocks")
        if isinstance(tb, list) and tb:
            think_blocks = [str(item) for item in tb]
    text = "".join(text_parts)
    return {
        "status": 200,
        "elapsed": round(time.time()-t0, 1),
        "finish": finish,
        "usage": usage,
        "text_chars": len(text),
        "text_head": text[:200],
        "tb_count": len(think_blocks),
        "tb_chars": sum(len(b) for b in think_blocks),
        "tb_head": think_blocks[0][:200] if think_blocks else "",
    }

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="http://localhost:18093")
    p.add_argument("--out", default="/tmp/cttkw_test_results.json")
    args = p.parse_args()

    arr = synth()
    print(f"[video] synth shape={arr.shape}")

    cases = [
        ("A_both",           {"enable_thinking": True, "supports_thinking": True}),
        ("B_enable_only",    {"enable_thinking": True}),
        ("C_supports_only",  {"supports_thinking": True}),
        ("D_reject_tokenize", {"tokenize": True}),
        ("E_reject_tools",    {"tools": "anything"}),
    ]

    results = {}
    for label, ctk in cases:
        print(f"\n=== {label} ctk={ctk} ===")
        body = build_body(arr, 2.5, 0.0, 4.0, ctk)
        r = run(args.host, body)
        results[label] = r
        if r["status"] != 200:
            print(f"  status={r['status']} body={r['body'][:200]}")
        else:
            usage = r["usage"] or {}
            print(f"  elapsed={r['elapsed']}s finish={r['finish']} "
                  f"prompt_tokens={usage.get('prompt_tokens')} "
                  f"completion_tokens={usage.get('completion_tokens')}")
            print(f"  text_chars={r['text_chars']} text_head={r['text_head'][:120]!r}")
            print(f"  think_blocks={r['tb_count']} chars={r['tb_chars']} head={r['tb_head'][:120]!r}")

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {args.out}")

    print("\n=========== VERDICT ===========")
    def expect_400(r): return r["status"] == 400
    def expect_thinking(r): return r["status"] == 200 and r["tb_count"] >= 1 and r["tb_chars"] >= 1
    checks = [
        ("A_both",          expect_thinking(results["A_both"]),         "both flags → think_blocks populated"),
        ("B_enable_only",   expect_thinking(results["B_enable_only"]),  "normalize: enable_thinking only → thinking still happens"),
        ("C_supports_only", expect_thinking(results["C_supports_only"]),"normalize: supports_thinking only → thinking still happens"),
        ("D_reject_tokenize", expect_400(results["D_reject_tokenize"]), "allowlist: tokenize → 400"),
        ("E_reject_tools",    expect_400(results["E_reject_tools"]),    "allowlist: tools → 400"),
    ]
    passed = 0
    for label, ok, msg in checks:
        mark = "PASS" if ok else "FAIL"
        print(f"  [{mark}] {label}: {msg}")
        passed += int(ok)
    print(f"\n{passed}/{len(checks)} checks passed")
    return 0 if passed == len(checks) else 1

if __name__ == "__main__":
    sys.exit(main())

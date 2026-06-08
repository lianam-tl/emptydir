"""Dump raw response from worker to see what bytes look like."""
import json, struct, sys
import numpy as np, requests
sys.path.insert(0, "/Users/long8v/emptydir")
from smoke_test_a1147_n_rollouts import build_v2_request

video = np.zeros((4, 64, 64, 3), dtype=np.uint8)
headers, body = build_v2_request(video=video, prompt="Describe this.", n=4, temperature=0.7, max_tokens=48, fps=2.0)
r = requests.post("http://localhost:18000/v2/models/vllm-video/infer", headers=headers, data=body, timeout=300)
print("status", r.status_code, "header-len:", r.headers.get("Inference-Header-Content-Length"))
hlen = int(r.headers["Inference-Header-Content-Length"])
hdr = json.loads(r.content[:hlen])
print("outputs in header:")
for o in hdr["outputs"]:
    print(" -", o)
payload = r.content[hlen:]
print("payload total len:", len(payload))
off = 0
for o in hdr["outputs"]:
    size = o["parameters"]["binary_data_size"]
    chunk = payload[off:off+size]
    print(f"output '{o['name']}': size={size}, first 16 bytes (hex)={chunk[:16].hex()}, first 80 chars (utf8 try)={chunk[:80]!r}")
    off += size

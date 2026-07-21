#!/usr/bin/env python3
import json
import sys
import urllib.request


job_id = sys.argv[1]
with urllib.request.urlopen(f"http://xplatform-training.twelve.labs/api/k8s-jobs/{job_id}", timeout=20) as response:
    payload = json.load(response)
print(payload.get("job", {}).get("status", "Unknown"))

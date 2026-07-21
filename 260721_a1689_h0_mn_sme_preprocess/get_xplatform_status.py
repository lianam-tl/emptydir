#!/usr/bin/env python3
import json
import subprocess
import sys


job_id = sys.argv[1]
response = subprocess.run(
    [
        "curl",
        "--fail",
        "--silent",
        "--show-error",
        f"http://xplatform-training.twelve.labs/api/k8s-jobs/{job_id}",
    ],
    check=True,
    capture_output=True,
    text=True,
)
payload = json.loads(response.stdout)
print(payload.get("job", {}).get("status", "Unknown"))

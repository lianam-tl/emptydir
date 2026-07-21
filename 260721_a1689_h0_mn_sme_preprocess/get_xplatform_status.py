#!/usr/bin/env python3
import json
import subprocess
import sys


job_id = sys.argv[1]
response = subprocess.run(
    [
        "kubectl",
        "--namespace",
        "pegasus-platform",
        "get",
        "job",
        f"job-{job_id}",
        "--output",
        "json",
    ],
    check=True,
    capture_output=True,
    text=True,
)
payload = json.loads(response.stdout)
status = payload.get("status", {})
if status.get("succeeded", 0):
    print("succeeded")
elif status.get("failed", 0):
    print("failed")
elif status.get("active", 0):
    print("running")
else:
    print("pending")

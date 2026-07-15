# Pegasus 1,200-second raw-output manifest

Build an index of all raw S3 artifacts returned by the Pegasus orchestrator for
one e2e run:

```bash
python3 build_manifest.py \
  --job-ids /path/to/job_ids.txt \
  --output-dir /path/to/output
```

The output includes `raw_output_manifest.json` and a browsable
`raw_output_manifest.html` table.

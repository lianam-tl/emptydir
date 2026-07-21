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

## Audit raw outputs

Download the output objects with `s5cmd`, then inspect all referenced JSON
files without calling the orchestrator again:

```bash
aws s3 cp --profile training s3://.../raw_output_manifest.json manifest.json
jq -r --arg destination /tmp/raw_outputs/ \
  '.[] | ["cp", .output_url, $destination] | join(" ")' manifest.json \
  > /tmp/s5cmd_commands.txt
eval "$(aws configure export-credentials --profile training --format env)"
s5cmd run /tmp/s5cmd_commands.txt
python3 audit_raw_outputs.py \
  --manifest manifest.json \
  --raw-output-dir /tmp/raw_outputs \
  --output-dir /tmp/raw_output_audit
```

This writes `raw_output_audit.json` and a browsable `raw_output_audit.html`.
It checks raw-output readability, final segment bounds and coverage, metadata
presence, definition errors, generation finish reasons, and postprocessing
repair counters. It does not judge semantic quality.

## Browse every model output

After the same `s5cmd` download, create an expandable, searchable viewer of
the model-produced summaries, entities, relationships, and shot metadata:

```bash
python3 build_raw_output_viewer.py \
  --manifest manifest.json \
  --raw-output-dir /tmp/raw_outputs \
  --output /tmp/raw_output_model_viewer.html
```

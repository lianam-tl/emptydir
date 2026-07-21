# slack_poll — generic long-running job Slack poller

One `poll_generic.sh` you can point at anything: tlab/kubectl, ray, HTTP endpoint,
a shell one-liner, whatever. Sends init + heartbeat + terminal messages to Slack
(`#fun-lia-trashcan` by default).

## Contract

Required env vars:

| var | meaning |
|---|---|
| `JOB_ID` | unique id used in messages and log filename (`~/poll_$JOB_ID.log`) |
| `STATUS_CMD` | shell command that echoes current status; runs via `eval`, may reference `$JOB_ID` |
| `SUCCESS_PATTERN` | extended regex; match → exit 0, :white_check_mark: |
| `FAILURE_PATTERN` | extended regex; match → exit 1, :x: |

Optional:

| var | default |
|---|---|
| `LABEL` | `$JOB_ID` |
| `ARTIFACT_URL` | (empty) — mentioned in success message |
| `CHANNEL` | `fun-lia-trashcan` |
| `SLACK_BOT_TOKEN` | preferred token source; falls back to `TOKEN_FILE` |
| `TOKEN_FILE` | `~/tmp/.slack_bot_token` |
| `POLL_SEC` | 120 |
| `HEARTBEAT_SEC` | 1200 (20 min) |

## Recipes

### tlab / kubectl pytorchjob (namespace: research)

```bash
ssh cpu "JOB_ID=export-safetensors-2200-thvur7 \
  STATUS_CMD='kubectl get pytorchjob \$JOB_ID -n research -o jsonpath={.status.conditions[?(@.status==\"True\")].type}' \
  SUCCESS_PATTERN='Succeeded|Complete' \
  FAILURE_PATTERN='Failed' \
  LABEL='DCP -> safetensors' \
  ARTIFACT_URL='s3://.../checkpoint-2200-safetensors/' \
  nohup ~/poll_generic.sh > ~/poll.out 2>&1 &"
```

### ray job

```bash
JOB_ID=raysubmit_abc123 \
  STATUS_CMD='ray job status $JOB_ID 2>&1 | grep -oE "SUCCEEDED|FAILED|RUNNING|PENDING|STOPPED"' \
  SUCCESS_PATTERN='SUCCEEDED' \
  FAILURE_PATTERN='FAILED|STOPPED' \
  nohup ~/poll_generic.sh &
```

### plain shell / file-based

```bash
# poll for a done marker file
JOB_ID=my-batch \
  STATUS_CMD='[ -f /tmp/my-batch.done ] && echo done || echo running' \
  SUCCESS_PATTERN='done' \
  FAILURE_PATTERN='fail' \
  nohup ~/poll_generic.sh &
```

### HTTP endpoint (JSON)

```bash
JOB_ID=my-http-job \
  STATUS_CMD='curl -sS http://svc/jobs/$JOB_ID | jq -r .state' \
  SUCCESS_PATTERN='completed|success' \
  FAILURE_PATTERN='failed|error' \
  nohup ~/poll_generic.sh &
```

## Install on cpu node

```bash
scp ~/emptydir/slack_poll/poll_generic.sh cpu:~/poll_generic.sh
ssh cpu chmod +x ~/poll_generic.sh
```

## Gotchas

- `STATUS_CMD` is `eval`-ed. Use single quotes at the outer shell so `$JOB_ID`
  is expanded inside the script, not by the caller's shell.
- If `STATUS_CMD` returns empty (e.g. job not yet visible), status becomes
  `Unknown`. First-iteration transitions are suppressed to avoid a spurious
  `"" -> Unknown` message.
- Success is checked before failure — if a status string could match both,
  fix your patterns.
- Log at `~/poll_$JOB_ID.log` contains every kubectl/curl response body,
  useful for debugging when Slack messages don't arrive.

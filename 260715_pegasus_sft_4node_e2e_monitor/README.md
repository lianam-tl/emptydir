# Pegasus SFT 4-node e2e monitor

`collect_job_status.sh` reads the Pegasus orchestrator status for every job ID
recorded by an e2e run. It produces one compact status line suitable for the
generic Slack poller.

Example:

```bash
ORCHESTRATOR_URL=http://xplatform-training.twelve.labs/orchestrator \
  ./collect_job_status.sh ~/pegasus-sft-4node-e2e-job-ids.txt
```

#!/usr/bin/env bash
set -euo pipefail

comparison_root="${1:-/var/tmp/jeongyeon-nam/260724_compare_qwen35_weights}"
script_directory=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

source /home/jeongyeon-nam/lia-ooo-bot/.env
export AWS_PROFILE=training

send_slack() {
  curl --silent --show-error --fail \
    -H "Authorization: Bearer $SLACK_BOT_TOKEN" \
    --data-urlencode "channel=C0ATJME17EK" \
    --data-urlencode "text=$1" \
    https://slack.com/api/chat.postMessage >/dev/null
}

handle_exit() {
  exit_code=$?
  if [[ $exit_code -eq 0 ]]; then
    send_slack ":white_check_mark: Qwen3.5 checkpoint-200 exact weight comparison finished."
  else
    send_slack ":x: Qwen3.5 checkpoint-200 exact weight comparison failed (exit $exit_code)."
  fi
}
trap handle_exit EXIT

mkdir -p "$comparison_root/base_manifests"
send_slack ":mag: Started exact Qwen3.5 checkpoint-200 vs release weight comparison."

checkpoint_uri='s3://tl-data-training-pegasus-us-west-2/checkpoints/jeongyeon-nam/sft_h0_entity_v1_2_7n_qwen3_5_27b-base/checkpoint-200-safetensors/model.safetensors'
s5cmd --numworkers 1 cat "$checkpoint_uri" | python3 "$script_directory/stream_tensor_hashes.py" \
  --source "$checkpoint_uri" --output "$comparison_root/checkpoint_manifest.json" \
  --inline-max-bytes 4096

for shard_number in $(seq 1 11); do
  printf -v shard_name '%05d' "$shard_number"
  base_uri="s3://tl-data-training-pegasus-us-west-2/hf_models/Qwen/Qwen3.5-27B/model.safetensors-${shard_name}-of-00011.safetensors"
  s5cmd --numworkers 1 cat "$base_uri" | python3 "$script_directory/stream_tensor_hashes.py" \
    --source "$base_uri" --output "$comparison_root/base_manifests/$shard_name.json" \
    --inline-max-bytes 4096
done

python3 "$script_directory/compare_tensor_hashes.py" \
  "$comparison_root/checkpoint_manifest.json" \
  "$comparison_root"/base_manifests/*.json \
  --output-json "$comparison_root/comparison.json" \
  --output-html "$comparison_root/comparison.html"

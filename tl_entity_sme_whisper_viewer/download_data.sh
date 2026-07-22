#!/usr/bin/env bash
set -euo pipefail

destination=${1:-/tmp/tl_entity_sme_whisper_viewer_data_260722}
source_uri='s3://tl-data-training-pegasus-us-west-2/annotation/preprocessed_datasets/base/tl_entity_sme_whisper/default_sft_entity_sme_whisper_asr/sft_sme'

mkdir -p "$destination"
eval "$(aws configure export-credentials --profile training --format env)"
s5cmd cp "$source_uri/*.arrow" "$destination/"
s5cmd cp "$source_uri/manifest.json" "$destination/"
s5cmd cp "$source_uri/dataset_info.json" "$destination/"

arrow_count=$(find "$destination" -maxdepth 1 -type f -name '*.arrow' | wc -l | tr -d ' ')
if [[ "$arrow_count" != "64" ]]; then
  echo "Expected 64 Arrow shards, found $arrow_count in $destination" >&2
  exit 1
fi
echo "Downloaded 64 Arrow shards and metadata to $destination"

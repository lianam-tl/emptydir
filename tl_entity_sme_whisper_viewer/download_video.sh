#!/usr/bin/env bash
set -euo pipefail

destination_directory=${1:-"$HOME/Desktop/html"}
source_uri='s3://tl-data-training-pegasus-us-west-2/raw_media/private/dense_caption_visual/7681b797/7681b797_chunk0.mp4'
filename='260722_tl_entity_sme_whisper_sample_7681b797.mp4'

mkdir -p "$destination_directory"
eval "$(aws configure export-credentials --profile training --format env)"
s5cmd cp "$source_uri" "$destination_directory/$filename"
echo "Downloaded $destination_directory/$filename"

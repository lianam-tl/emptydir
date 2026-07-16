#!/usr/bin/env bash
set -euo pipefail

script_directory="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repository_directory="$(cd "${script_directory}/.." && pwd)"
service_name="eval-v3-api-pr-1729-cov-v02"
namespace="pegasus-eval"
local_port="18094"
output_directory="${script_directory}/outputs_repro_pr_1729_cov_v02"
port_forward_log="${output_directory}/port_forward.log"
port_forward_pid_file="${output_directory}/port_forward.pid"
dataset_cache_source="${HOME}/.cache/huggingface/datasets/twelvelabs___entity_cov_v02_tdf"
hub_cache_source="${HOME}/.cache/huggingface/hub/datasets--twelvelabs--entity_cov_v02_tdf"
pod_cache_root="/tmp/eval-v3-cache/huggingface"

mkdir -p "${output_directory}"

if ! kubectl --context training -n "${namespace}" get service "${service_name}" >/dev/null; then
  echo "Missing service ${namespace}/${service_name}" >&2
  exit 1
fi

api_pod_name="$(
  kubectl --context training -n "${namespace}" get pod \
    -l "app=${service_name}" \
    -o jsonpath='{.items[0].metadata.name}'
)"

if [[ -z "${api_pod_name}" ]]; then
  echo "Missing API pod for app=${service_name}" >&2
  exit 1
fi

kubectl --context training -n "${namespace}" exec "${api_pod_name}" -- sh -lc \
  "mkdir -p ${pod_cache_root}/datasets ${pod_cache_root}/hub && rm -rf ${pod_cache_root}/datasets/twelvelabs___entity_cov_v02_tdf ${pod_cache_root}/hub/datasets--twelvelabs--entity_cov_v02_tdf"

if [[ -d "${dataset_cache_source}" ]]; then
  kubectl --context training -n "${namespace}" cp \
    "${dataset_cache_source}" \
    "${api_pod_name}:${pod_cache_root}/datasets/twelvelabs___entity_cov_v02_tdf"
fi

if [[ -d "${hub_cache_source}" ]]; then
  kubectl --context training -n "${namespace}" cp \
    "${hub_cache_source}" \
    "${api_pod_name}:${pod_cache_root}/hub/datasets--twelvelabs--entity_cov_v02_tdf"
fi

if ! lsof -nP -iTCP:"${local_port}" -sTCP:LISTEN >/dev/null 2>&1; then
  kubectl --context training -n "${namespace}" port-forward \
    "svc/${service_name}" "${local_port}:8090" \
    >"${port_forward_log}" 2>&1 &
  echo "$!" >"${port_forward_pid_file}"
  sleep 3
fi

curl -fsS "http://localhost:${local_port}/livez" >/dev/null
curl -fsS "http://localhost:${local_port}/readyz" >/dev/null

cd "${repository_directory}"
python3 "${script_directory}/run_eval.py" \
  --host "http://localhost:${local_port}" \
  --output-dir "${output_directory}"

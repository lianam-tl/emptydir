"""Discover and summarize training mixtures from checkpoint metadata."""

from __future__ import annotations

import json
import re
from typing import Any
from urllib.parse import urlparse

import yaml


BASE_MIXTURE_DATASETS = {
    "allava_laion_caption",
    "allava_laion_vqa",
    "allava_vflan_caption",
    "allava_vflan_vqa",
    "dd400k",
    "finevideo_sme",
    "laion_caption_gpt4_lvis",
    "laion_gpt4v",
    "lrv_version1",
    "lrv_version1_more",
    "openhermes",
    "sharegpt",
    "taxo_sft_241016_mcqa_rlsplit_40k_rl",
    "taxo_sft_250110_reannotate_132k_filtered_train",
    "taxo_sft_250110_reannotate_1m_filtered_train",
    "taxo_sft_250110_reannotate_1m_rationale_train",
    "tl_american_football_24Q2_temp_loc_qa",
    "tl_american_football_24Q2_time_conditioned_qa",
    "tl_composite_movies_and_tvshows",
    "tl_composite_sports",
    "tl_dense_caption_v1_visual",
    "tl_gemini_sports_sme",
    "tl_gemini_sports_sme_v4",
    "tl_ice_hockey_24Q3_temp_loc_qa",
    "tl_ice_hockey_24Q3_time_conditioned_qa",
    "tl_news_us_sme_c0_c1_c3",
    "tl_soccer_dense_caption",
    "tl_soccer_h16_sme",
    "tl_sports_dense_caption",
    "tl_sports_dense_varying_duration",
    "tl_sports_sme_A0",
    "tl_sports_sme_A1",
    "video_chapters_7m",
    "video_chapters_7m_A0",
    "video_chapters_7m_A1",
}
KNOWN_COMPONENTS = {
    "H0 standard",
    "H0 duration",
    "H0 2x",
    "Entity SME 4x",
    "Entity SME v1.2",
    "Entity Whisper",
    "Soccer LVReason",
}
CHECKPOINT_DIRECTORY = re.compile(
    r"^(?:checkpoint-\d+(?:-safetensors)?|global_step_\d+|step_\d+)$"
)


def split_s3_uri(uri: str) -> tuple[str, str]:
    parsed = urlparse(uri)
    if parsed.scheme != "s3" or not parsed.netloc:
        raise ValueError(f"not an S3 URI: {uri}")
    return parsed.netloc, parsed.path.lstrip("/")


def experiment_metadata_uri(model_path: str) -> str:
    bucket, key = split_s3_uri(model_path.rstrip("/"))
    parts = key.split("/")
    if parts and CHECKPOINT_DIRECTORY.fullmatch(parts[-1]):
        parts.pop()
    return f"s3://{bucket}/{'/'.join(parts)}/experiment_metadata.yaml"


def mixture_stats_uri(metadata_text: str) -> str:
    metadata = yaml.safe_load(metadata_text)
    resolved_text = metadata["training"]["resolved_config_yaml"]
    resolved = yaml.safe_load(resolved_text)
    model_input = (resolved.get("dataset_config") or {}).get("dataset_path")
    if not model_input:
        model_input = (resolved.get("data") or {}).get("train_files")
    if isinstance(model_input, list):
        model_input = model_input[0] if model_input else None
    if not isinstance(model_input, str) or not model_input.startswith("s3://"):
        raise ValueError("experiment metadata has no S3 model_input path")
    if model_input.rstrip("/").endswith(".parquet"):
        model_input = model_input.rsplit("/", 1)[0]
    return f"{model_input.rstrip('/')}/mixture_stats.json"


def wandb_url(metadata_text: str) -> str:
    metadata = yaml.safe_load(metadata_text) or {}
    url = str((metadata.get("wandb") or {}).get("url") or "")
    return url if url.startswith("https://wandb.ai/") else ""


def read_s3_text(s3_client: Any, uri: str) -> str:
    bucket, key = split_s3_uri(uri)
    response = s3_client.get_object(Bucket=bucket, Key=key)
    return response["Body"].read().decode("utf-8")


def component_name(dataset: str, template: str) -> str:
    if dataset == "tl_h0_movies_and_news_sme":
        if "duration" in template:
            return "H0 duration"
        if "2x" in template:
            return "H0 2x"
        return "H0 standard"
    if dataset == "tl_entity_sme":
        return "Entity SME 4x" if "4x" in template else "Entity SME"
    if dataset == "tl_entity_sme_v1_2":
        return "Entity SME v1.2"
    if dataset == "tl_entity_sme_whisper":
        return "Entity Whisper"
    if dataset == "longvideo_reason":
        return "Soccer LVReason"
    if dataset in BASE_MIXTURE_DATASETS:
        return "Other/base"
    return dataset


def summarize_mixture(family: str, payload: dict[str, Any]) -> dict[str, float]:
    total_tokens = int(payload["total_tokens"])
    if family == "Pegasus 1.5 RL":
        return {"P1.5 RL mix": 1.0}
    if family == "Pegasus 1.5 SFT":
        return {"P1.5 SFT mix": 1.0}

    components: dict[str, float] = {}
    for row in payload["rows"]:
        component = component_name(str(row["dataset"]), str(row["template"]))
        components[component] = components.get(component, 0.0) + (
            int(row["tokens"]) / total_tokens
        )

    for component, ratio in list(components.items()):
        if (
            component not in KNOWN_COMPONENTS
            and component != "Other/base"
            and ratio < 0.005
        ):
            components["Other/base"] = components.get("Other/base", 0.0) + ratio
            del components[component]
    return components


def discover_training_mixture(
    family: str, model_path: str, s3_client: Any
) -> dict[str, Any]:
    metadata_uri = experiment_metadata_uri(model_path)
    metadata_text = read_s3_text(s3_client, metadata_uri)
    statistics_uri = mixture_stats_uri(metadata_text)
    payload = json.loads(read_s3_text(s3_client, statistics_uri))
    return {
        "family": family,
        "model_path": model_path,
        "experiment_metadata": metadata_uri,
        "wandb_url": wandb_url(metadata_text),
        "mixture_stats": statistics_uri,
        "total_tokens": int(payload["total_tokens"]),
        "components": summarize_mixture(family, payload),
    }

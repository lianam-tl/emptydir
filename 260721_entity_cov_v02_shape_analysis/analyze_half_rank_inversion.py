#!/usr/bin/env python3
"""Explain the v0 versus v0.2 half rank inversion for two checkpoints."""

from __future__ import annotations

import html
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
COLLECTED_RUNS_PATH = ROOT / "collected_runs.json"
INFERENCE_METADATA_PATH = ROOT / "inference_metadata.json"
V0_CANDIDATES_PATH = ROOT / "entity_v0_candidates.json"
V0_PAYLOADS_PATH = ROOT / "v0_comparison_payloads.json"
ANALYSIS_PATH = ROOT / "half_rank_inversion_analysis.json"
REPORT_PATH = ROOT / "half_rank_inversion_report.html"

A1740 = "a1740-h0-duration-s400"
CONSOL = "consol-h0mn2x-s2000"
MODELS = (A1740, CONSOL)
MODEL_LABELS = {A1740: "A1740 duration s400", CONSOL: "Consol h0/mn 2x s2000"}

NAMING_KEY = "entity_coverage::naming_iou"
APPEARANCE_KEY = "entity_coverage::name_appearance_iou"


def mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def sample_source(sample_id: str) -> str:
    return sample_id.split("__")[1]


def scored_characters(samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        character
        for sample in samples
        for character in sample["character_scores"]
        if character.get("scored")
    ]


def pooled_slices(samples: list[dict[str, Any]]) -> dict[str, Any]:
    characters = scored_characters(samples)
    known = [character for character in characters if character["name_known"]]
    unknown = [character for character in characters if not character["name_known"]]
    return {
        "characters": len(characters),
        "known_characters": len(known),
        "unknown_characters": len(unknown),
        "naming": mean([character[NAMING_KEY] for character in known]),
        "appearance": mean([character[APPEARANCE_KEY] for character in characters]),
        "known_appearance": mean([character[APPEARANCE_KEY] for character in known]),
        "unknown_appearance": mean(
            [character[APPEARANCE_KEY] for character in unknown]
        ),
    }


def source_slices(samples: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        grouped[sample_source(sample["sample_id"])].append(sample)
    return {
        source: pooled_slices(source_samples)
        for source, source_samples in grouped.items()
    }


def parse_prediction(path: Path) -> dict[str, Any]:
    envelope = json.loads(path.read_text())
    try:
        payload = json.loads(envelope["text"])
    except (json.JSONDecodeError, TypeError, KeyError):
        payload = None
    if payload is None:
        return {
            "valid_json": False,
            "finish_reason": envelope.get("finish_reason"),
            "output_tokens": envelope.get("output_tokens"),
            "rosters": None,
            "shots": None,
            "mentions": None,
            "roster_names": [],
            "mention_counts": {},
        }

    rosters = payload.get("rosters", [])
    shots = payload.get("shot_metadata", [])
    mention_names = [
        str(entity.get("canonical_name") or "")
        for shot in shots
        if isinstance(shot, dict)
        for entity in shot.get("entities", [])
        if isinstance(entity, dict) and entity.get("canonical_name")
    ]
    return {
        "valid_json": True,
        "finish_reason": envelope.get("finish_reason"),
        "output_tokens": envelope.get("output_tokens"),
        "rosters": len(rosters),
        "shots": len(shots),
        "mentions": len(mention_names),
        "roster_names": [
            str(entity.get("canonical_name") or "")
            for entity in rosters
            if isinstance(entity, dict) and entity.get("canonical_name")
        ],
        "mention_counts": dict(Counter(mention_names)),
    }


def model_structure(
    samples: list[dict[str, Any]], predictions: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    valid_samples = [
        sample for sample in samples if predictions[sample["sample_id"]]["valid_json"]
    ]
    valid_predictions = [predictions[sample["sample_id"]] for sample in valid_samples]
    return {
        "valid_predictions": len(valid_predictions),
        "parse_failures": len(samples) - len(valid_predictions),
        "length_finishes": sum(
            predictions[sample["sample_id"]]["finish_reason"] == "length"
            for sample in samples
        ),
        "mean_rosters": mean(
            [prediction["rosters"] for prediction in valid_predictions]
        ),
        "mean_shots": mean([prediction["shots"] for prediction in valid_predictions]),
        "mean_mentions": mean(
            [prediction["mentions"] for prediction in valid_predictions]
        ),
        "mean_predicted_entities": mean(
            [sample["counts"]["predicted_entities"] for sample in valid_samples]
        ),
        "mean_predicted_spans": mean(
            [sample["counts"]["predicted_spans"] for sample in valid_samples]
        ),
        "mean_output_tokens": mean(
            [prediction["output_tokens"] for prediction in valid_predictions]
        ),
    }


def calculate_analysis() -> dict[str, Any]:
    v02_runs = {
        run["name"]: run
        for run in json.loads(COLLECTED_RUNS_PATH.read_text())["runs"]
        if run["name"] in MODELS
    }
    v0_runs = {
        run["name"]: run for run in json.loads(V0_PAYLOADS_PATH.read_text())["runs"]
    }
    candidates = {
        row["v02_name"]: row
        for row in json.loads(V0_CANDIDATES_PATH.read_text())["checkpoints"]
        if row["v02_name"] in MODELS
    }
    metadata = json.loads(INFERENCE_METADATA_PATH.read_text())["records"]
    raw_path_by_key = {
        (row["name"], row["sample_id"]): Path(row["raw_file"])
        for row in metadata
        if row["name"] in MODELS and row["shape"] == "half"
    }

    v02_samples: dict[str, list[dict[str, Any]]] = {}
    predictions: dict[str, dict[str, dict[str, Any]]] = {}
    for model in MODELS:
        v02_samples[model] = [
            sample
            for sample in v02_runs[model]["benchmark"]["entity_coverage"]["samples"]
            if "__half__" in sample["sample_id"]
        ]
        predictions[model] = {
            sample["sample_id"]: parse_prediction(
                raw_path_by_key[(model, sample["sample_id"])]
            )
            for sample in v02_samples[model]
        }

    failed_sample_ids = {
        sample["sample_id"]
        for model in MODELS
        for sample in v02_samples[model]
        if sample.get("error")
    }
    common_valid_samples = {
        model: [
            sample
            for sample in v02_samples[model]
            if sample["sample_id"] not in failed_sample_ids
        ]
        for model in MODELS
    }

    model_summaries: dict[str, Any] = {}
    for model in MODELS:
        own_clean = [sample for sample in v02_samples[model] if not sample.get("error")]
        model_summaries[model] = {
            "label": MODEL_LABELS[model],
            "v0": {
                "naming": candidates[model]["naming"],
                "appearance": candidates[model]["appearance"],
                "scored": candidates[model]["scored"],
                "failed": candidates[model]["failed"],
                "slices": pooled_slices(
                    v0_runs[model]["benchmark"]["entity_coverage"]["samples"]
                ),
            },
            "v02_half": {
                "official": pooled_slices(v02_samples[model]),
                "own_clean": pooled_slices(own_clean),
                "common_valid": pooled_slices(common_valid_samples[model]),
                "failed_samples": sorted(
                    sample["sample_id"]
                    for sample in v02_samples[model]
                    if sample.get("error")
                ),
            },
            "structure": model_structure(v02_samples[model], predictions[model]),
        }

    v02_sources_by_model = {
        model: source_slices(v02_samples[model]) for model in MODELS
    }
    v0_sources_by_model = {
        model: source_slices(v0_runs[model]["benchmark"]["entity_coverage"]["samples"])
        for model in MODELS
    }
    v02_source_rows = []
    v0_source_rows = []
    for source in sorted(v02_sources_by_model[A1740]):
        v02_source_rows.append(
            {
                "source": source,
                "characters": v02_sources_by_model[A1740][source]["characters"],
                "a1740": v02_sources_by_model[A1740][source],
                "consol": v02_sources_by_model[CONSOL][source],
                "appearance_gap": v02_sources_by_model[A1740][source]["appearance"]
                - v02_sources_by_model[CONSOL][source]["appearance"],
            }
        )
        v0_source_rows.append(
            {
                "source": source,
                "characters": v0_sources_by_model[A1740][source]["characters"],
                "a1740": v0_sources_by_model[A1740][source],
                "consol": v0_sources_by_model[CONSOL][source],
                "appearance_gap": v0_sources_by_model[A1740][source]["appearance"]
                - v0_sources_by_model[CONSOL][source]["appearance"],
            }
        )

    samples_by_model = {
        model: {sample["sample_id"]: sample for sample in v02_samples[model]}
        for model in MODELS
    }
    sample_rows = []
    for sample_id in sorted(samples_by_model[A1740]):
        row: dict[str, Any] = {
            "sample_id": sample_id,
            "source": sample_source(sample_id),
        }
        for model, key in ((A1740, "a1740"), (CONSOL, "consol")):
            sample = samples_by_model[model][sample_id]
            prediction = predictions[model][sample_id]
            row[key] = {
                "error": sample.get("error"),
                "naming": sample["metrics"][NAMING_KEY],
                "appearance": sample["metrics"][APPEARANCE_KEY],
                "predicted_entities": sample["counts"]["predicted_entities"],
                "predicted_spans": sample["counts"]["predicted_spans"],
                "rosters": prediction["rosters"],
                "shots": prediction["shots"],
                "mentions": prediction["mentions"],
                "output_tokens": prediction["output_tokens"],
            }
        row["appearance_gap"] = row["a1740"]["appearance"] - row["consol"]["appearance"]
        sample_rows.append(row)

    character_by_model: dict[str, dict[tuple[str, str], dict[str, Any]]] = {}
    for model in MODELS:
        character_by_model[model] = {
            (sample["sample_id"], character["label_id"]): character
            for sample in v02_samples[model]
            for character in sample["character_scores"]
            if character.get("scored")
        }
    character_rows = []
    for key, a1740_character in character_by_model[A1740].items():
        consol_character = character_by_model[CONSOL][key]
        character_rows.append(
            {
                "sample_id": key[0],
                "source": sample_source(key[0]),
                "label_id": key[1],
                "name_known": a1740_character["name_known"],
                "ground_truth_spans": a1740_character["ground_truth_span_count"],
                "a1740_naming": a1740_character.get(NAMING_KEY),
                "consol_naming": consol_character.get(NAMING_KEY),
                "a1740_appearance": a1740_character[APPEARANCE_KEY],
                "consol_appearance": consol_character[APPEARANCE_KEY],
                "a1740_predicted_spans": a1740_character[
                    "name_appearance_predicted_span_count"
                ],
                "consol_predicted_spans": consol_character[
                    "name_appearance_predicted_span_count"
                ],
                "appearance_gap": a1740_character[APPEARANCE_KEY]
                - consol_character[APPEARANCE_KEY],
            }
        )
    character_rows.sort(key=lambda row: abs(row["appearance_gap"]), reverse=True)

    v0_without_sport = {}
    for model in MODELS:
        non_sport_samples = [
            sample
            for sample in v0_runs[model]["benchmark"]["entity_coverage"]["samples"]
            if sample_source(sample["sample_id"]) != "sport-01"
        ]
        v0_without_sport[model] = pooled_slices(non_sport_samples)

    film05_id = "entity_coverage_v0__film-05__half__001"
    film02_id = "entity_coverage_v0__film-02__half__001"
    sport_v02_id = "entity_coverage_v0__sport-01__half__000"
    sport_v0_samples = {
        model: next(
            sample
            for sample in v0_runs[model]["benchmark"]["entity_coverage"]["samples"]
            if sample_source(sample["sample_id"]) == "sport-01"
        )
        for model in MODELS
    }
    examples = {
        "film05_husband_mentions": {
            model: predictions[model][film05_id]["mention_counts"].get("Husband", 0)
            for model in MODELS
        },
        "film02_roster_names": {
            model: predictions[model][film02_id]["roster_names"] for model in MODELS
        },
        "sport": {
            model: {
                "v0_appearance": sport_v0_samples[model]["metrics"][APPEARANCE_KEY],
                "v0_predicted_spans": sport_v0_samples[model]["counts"][
                    "predicted_spans"
                ],
                "v02_appearance": samples_by_model[model][sport_v02_id]["metrics"][
                    APPEARANCE_KEY
                ],
                "v02_predicted_spans": samples_by_model[model][sport_v02_id]["counts"][
                    "predicted_spans"
                ],
            }
            for model in MODELS
        },
    }

    official_gap = (
        model_summaries[A1740]["v02_half"]["official"]["appearance"]
        - model_summaries[CONSOL]["v02_half"]["official"]["appearance"]
    )
    common_valid_gap = (
        model_summaries[A1740]["v02_half"]["common_valid"]["appearance"]
        - model_summaries[CONSOL]["v02_half"]["common_valid"]["appearance"]
    )
    return {
        "models": model_summaries,
        "failed_sample_ids": sorted(failed_sample_ids),
        "gaps": {
            "official_v02_half_appearance": official_gap,
            "common_valid_v02_half_appearance": common_valid_gap,
            "failure_placement_share": (official_gap - common_valid_gap) / official_gap,
            "v0_appearance": candidates[A1740]["appearance"]
            - candidates[CONSOL]["appearance"],
            "v0_without_sport_appearance": v0_without_sport[A1740]["appearance"]
            - v0_without_sport[CONSOL]["appearance"],
        },
        "v0_without_sport": v0_without_sport,
        "v02_source_rows": v02_source_rows,
        "v0_source_rows": v0_source_rows,
        "sample_rows": sample_rows,
        "character_rows": character_rows,
        "examples": examples,
        "scorer_differences": {
            "v0_dataset": "https://huggingface.co/datasets/twelvelabs/entity_cov_v0_tdf",
            "v0_config": "chunk_10m",
            "v0_rows": 20,
            "v0_character_observations": model_summaries[A1740]["v0"]["slices"][
                "characters"
            ],
            "v0_schema": "flat roster + direct spans",
            "v0_mapper": "gpt-5.2",
            "v02_dataset": "https://huggingface.co/datasets/twelvelabs/entity_cov_v02_tdf",
            "v02_shape": "half",
            "v02_rows": 13,
            "v02_character_observations": model_summaries[A1740]["v02_half"][
                "official"
            ]["characters"],
            "v02_schema": "nested rosters + shot_metadata.entities",
            "v02_mapper": "gpt-5.4-mini",
        },
    }


def score(value: float | None, digits: int = 3) -> str:
    return "&mdash;" if value is None else f"{value:.{digits}f}"


def render_table(headers: list[str], rows: list[list[str]]) -> str:
    return (
        '<div class="table-wrap"><table><thead><tr>'
        + "".join(f"<th>{html.escape(header)}</th>" for header in headers)
        + "</tr></thead><tbody>"
        + "".join(
            "<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>"
            for row in rows
        )
        + "</tbody></table></div>"
    )


def render_report(analysis: dict[str, Any]) -> str:
    models = analysis["models"]
    gaps = analysis["gaps"]
    structure_table = render_table(
        [
            "Model",
            "Valid JSON",
            "Mean top-level rosters",
            "Mean scored person entities",
            "Mean shots",
            "Mean person mentions",
            "Mean scored spans",
        ],
        [
            [
                models[model]["label"],
                f"{models[model]['structure']['valid_predictions']}/13",
                score(models[model]["structure"]["mean_rosters"], 2),
                score(models[model]["structure"]["mean_predicted_entities"], 2),
                score(models[model]["structure"]["mean_shots"], 2),
                score(models[model]["structure"]["mean_mentions"], 2),
                score(models[model]["structure"]["mean_predicted_spans"], 2),
            ]
            for model in MODELS
        ],
    )
    slice_table = render_table(
        [
            "Slice",
            "A1740 naming",
            "Consol naming",
            "A1740 known appearance",
            "Consol known appearance",
            "A1740 unknown appearance",
            "Consol unknown appearance",
            "A1740 overall appearance",
            "Consol overall appearance",
        ],
        [
            [
                label,
                score(models[A1740]["v02_half"][key]["naming"]),
                score(models[CONSOL]["v02_half"][key]["naming"]),
                score(models[A1740]["v02_half"][key]["known_appearance"]),
                score(models[CONSOL]["v02_half"][key]["known_appearance"]),
                score(models[A1740]["v02_half"][key]["unknown_appearance"]),
                score(models[CONSOL]["v02_half"][key]["unknown_appearance"]),
                score(models[A1740]["v02_half"][key]["appearance"]),
                score(models[CONSOL]["v02_half"][key]["appearance"]),
            ]
            for label, key in (
                ("Official: all 13 rows", "official"),
                ("Own parse-success rows", "own_clean"),
                ("Common valid: exclude both failed rows", "common_valid"),
            )
        ],
    )
    source_table = render_table(
        [
            "Source",
            "Characters",
            "v0 A1740",
            "v0 Consol",
            "v0 A-C gap",
            "v0.2 half A1740",
            "v0.2 half Consol",
            "v0.2 A-C gap",
        ],
        [
            [
                v02_row["source"],
                f"{v0_row['characters']} / {v02_row['characters']}",
                score(v0_row["a1740"]["appearance"]),
                score(v0_row["consol"]["appearance"]),
                score(v0_row["appearance_gap"]),
                score(v02_row["a1740"]["appearance"]),
                score(v02_row["consol"]["appearance"]),
                score(v02_row["appearance_gap"]),
            ]
            for v0_row, v02_row in zip(
                analysis["v0_source_rows"],
                analysis["v02_source_rows"],
                strict=True,
            )
        ],
    )
    sample_table = render_table(
        [
            "Half sample",
            "A1740 appearance",
            "Consol appearance",
            "A-C gap",
            "A / C entities",
            "A / C spans",
            "A / C rosters",
            "A / C shots",
            "Failure",
        ],
        [
            [
                f'<span class="code">{row["sample_id"].replace("entity_coverage_v0__", "")}</span>',
                score(row["a1740"]["appearance"]),
                score(row["consol"]["appearance"]),
                score(row["appearance_gap"]),
                f"{row['a1740']['predicted_entities']} / {row['consol']['predicted_entities']}",
                f"{row['a1740']['predicted_spans']} / {row['consol']['predicted_spans']}",
                f"{row['a1740']['rosters'] if row['a1740']['rosters'] is not None else 'fail'} / {row['consol']['rosters'] if row['consol']['rosters'] is not None else 'fail'}",
                f"{row['a1740']['shots'] if row['a1740']['shots'] is not None else 'fail'} / {row['consol']['shots'] if row['consol']['shots'] is not None else 'fail'}",
                "A1740"
                if row["a1740"]["error"]
                else "Consol"
                if row["consol"]["error"]
                else "",
            ]
            for row in analysis["sample_rows"]
        ],
    )
    unknown_rows = [row for row in analysis["character_rows"] if not row["name_known"]]
    unknown_table = render_table(
        [
            "Sample",
            "GT label",
            "GT spans",
            "A1740 IoU",
            "Consol IoU",
            "A-C gap",
            "A / C predicted spans",
        ],
        [
            [
                f'<span class="code">{row["sample_id"].replace("entity_coverage_v0__", "")}</span>',
                html.escape(str(row["label_id"])),
                str(row["ground_truth_spans"]),
                score(row["a1740_appearance"]),
                score(row["consol_appearance"]),
                score(row["appearance_gap"]),
                f"{row['a1740_predicted_spans']} / {row['consol_predicted_spans']}",
            ]
            for row in unknown_rows[:15]
        ],
    )
    scorer = analysis["scorer_differences"]
    scorer_table = render_table(
        [
            "Benchmark",
            "Rows / character obs.",
            "Output representation",
            "Entity mapper",
        ],
        [
            [
                f'<a href="{scorer["v0_dataset"]}">v0 chunk_10m</a>',
                f"{scorer['v0_rows']} / {scorer['v0_character_observations']}",
                scorer["v0_schema"],
                scorer["v0_mapper"],
            ],
            [
                f'<a href="{scorer["v02_dataset"]}">v0.2 half</a>',
                f"{scorer['v02_rows']} / {scorer['v02_character_observations']}",
                scorer["v02_schema"],
                scorer["v02_mapper"],
            ],
        ],
    )
    chart_data = json.dumps(
        {
            "composition": {
                model: models[model]["v02_half"]["official"] for model in MODELS
            },
            "v0Sources": analysis["v0_source_rows"],
            "v02Sources": analysis["v02_source_rows"],
        }
    ).replace("</", "<\\/")
    husband = analysis["examples"]["film05_husband_mentions"]
    sport = analysis["examples"]["sport"]
    a1740_rosters = ", ".join(analysis["examples"]["film02_roster_names"][A1740])
    consol_rosters = ", ".join(analysis["examples"]["film02_roster_names"][CONSOL])

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Entity v0.2 half rank inversion audit</title>
  <style>
    :root {{ --bg:#f6f7f9; --panel:#fff; --ink:#1f2933; --muted:#687385; --line:#d9dee7; --blue:#0b62d6; --green:#137333; --amber:#8a5a00; --red:#b42318; }}
    * {{ box-sizing:border-box; }}
    body {{ margin:0; background:var(--bg); color:var(--ink); font:14px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; }}
    header {{ padding:25px 30px 18px; background:#101820; color:#fff; }}
    h1 {{ margin:0 0 7px; font-size:23px; letter-spacing:0; }}
    header p {{ margin:0; color:#cbd5e1; }}
    main {{ max-width:1600px; margin:auto; padding:20px 30px 55px; }}
    h2 {{ margin:29px 0 8px; font-size:18px; letter-spacing:0; }}
    h3 {{ margin:20px 0 7px; font-size:15px; letter-spacing:0; }}
    p {{ max-width:1120px; }}
    a {{ color:var(--blue); }}
    code,.code {{ font-family:ui-monospace,SFMono-Regular,Menlo,monospace; font-size:11px; }}
    .cards {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(190px,1fr)); gap:9px; margin:13px 0; }}
    .card {{ padding:11px 14px; border:1px solid var(--line); border-radius:8px; background:var(--panel); }}
    .card span {{ display:block; color:var(--muted); font-size:11px; font-weight:650; }}
    .card b {{ font-size:21px; }}
    .note {{ margin:10px 0; padding:10px 13px; border:1px solid #cfe0ff; border-radius:8px; background:#f0f6ff; max-width:1150px; }}
    .warn {{ border-color:#efd18a; background:#fff7e6; }}
    .danger {{ border-color:#f1b4ae; background:#fff1f0; }}
    .muted {{ color:var(--muted); }}
    .table-wrap {{ overflow:auto; border:1px solid var(--line); border-radius:8px; background:var(--panel); }}
    table {{ width:100%; border-collapse:collapse; font-size:12px; }}
    th,td {{ padding:7px 9px; border-bottom:1px solid var(--line); text-align:right; white-space:nowrap; }}
    th {{ position:sticky; top:0; background:#eef2f7; }}
    th:first-child,td:first-child {{ text-align:left; }}
    tr:last-child td {{ border-bottom:0; }}
    tr:hover td {{ background:#f7fbff; }}
    .chart-grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(480px,1fr)); gap:14px; margin-top:13px; }}
    .chart-wrap {{ min-width:0; min-height:390px; padding:14px; border:1px solid var(--line); border-radius:8px; background:var(--panel); }}
    .chart-wrap h3 {{ margin:0 0 8px; }}
    .chart-wrap canvas {{ width:100% !important; height:330px !important; }}
    ol {{ max-width:1100px; padding-left:24px; }}
    li {{ margin:6px 0; }}
    @media(max-width:700px) {{ main {{ padding:16px; }} header {{ padding:20px 16px; }} .chart-grid {{ grid-template-columns:1fr; }} }}
  </style>
</head>
<body>
<header>
  <h1>Why A1740 s400 outranks Consol s2000 on v0.2 half</h1>
  <p>Prediction, scorer, roster, and source-level audit &middot; 2026-07-21</p>
</header>
<main>
  <div class="cards">
    <div class="card"><span>OFFICIAL V0.2 HALF A-C GAP</span><b>{gaps["official_v02_half_appearance"]:+.1%}</b></div>
    <div class="card"><span>COMMON-VALID A-C GAP</span><b>{gaps["common_valid_v02_half_appearance"]:+.1%}</b></div>
    <div class="card"><span>FAILURE-PLACEMENT SHARE</span><b>{gaps["failure_placement_share"]:.0%}</b></div>
    <div class="card"><span>V0 GAP WITHOUT SPORT</span><b>{gaps["v0_without_sport_appearance"]:+.1%}</b></div>
  </div>
  <div class="note danger"><b>Verdict.</b> This is not a simple roster-count bug. Each model has one 64K parse failure, and the failed half differs; that placement explains about {gaps["failure_placement_share"]:.0%} of A1740's official lead. The remaining lead comes from name-only mapping and especially unnamed-character mapping/tracking. The original-v0 Consol lead is itself dominated by one <code>sport-01</code> result and disappears when sport is removed.</div>

  <h2>1. Parse-failure sensitivity</h2>
  <p>A1740 fails <code>film-04 half 001</code>; Consol fails <code>film-04 half 000</code>. "Own parse-success" removes a different row for each model and is not a fair pair. "Common valid" removes both rows from both models, leaving 61 shared character observations.</p>
  {slice_table}
  <div class="chart-grid"><div class="chart-wrap"><h3>Official half score composition</h3><canvas id="composition-chart"></canvas></div></div>

  <h2>2. Is the roster or output volume abnormal?</h2>
  <p>No. On the 12 parseable half outputs, roster count, person-entity count, shot count, and person mentions are similar. Top-level rosters also include objects and are not the scorer's identity unit.</p>
  {structure_table}
  <div class="note"><b>What the scorer actually uses.</b> It reads person/character entries from <code>shot_metadata.entities</code>, groups occurrences by <code>canonical_name</code>, and maps each name/description to ground truth with an LLM. Top-level <code>rosters</code> only provide a fallback description when a shot entity has none.</div>
  {sample_table}

  <h2>3. Where the v0.2 half gap comes from</h2>
  <p>For known characters, name + appearance IoU is nearly identical in the official pool: A1740 {score(models[A1740]["v02_half"]["official"]["known_appearance"])} versus Consol {score(models[CONSOL]["v02_half"]["official"]["known_appearance"])}. For unnamed characters it is A1740 {score(models[A1740]["v02_half"]["official"]["unknown_appearance"])} versus Consol {score(models[CONSOL]["v02_half"]["official"]["unknown_appearance"])}. Almost the entire aggregate gap is therefore in unnamed identity matching.</p>
  {unknown_table}
  <div class="note"><b>Concrete tracking example.</b> On <code>film-05 half 001</code>, both outputs contain a 12-entry roster with <code>Husband</code>. A1740 places Husband in {husband[A1740]} shots; Consol places him in only {husband[CONSOL]}. The scorer gives the GT Husband {score(next(row for row in unknown_rows if row["sample_id"].endswith("film-05__half__001") and row["label_id"] == "Husband")["a1740_appearance"])} versus {score(next(row for row in unknown_rows if row["sample_id"].endswith("film-05__half__001") and row["label_id"] == "Husband")["consol_appearance"])} IoU.</div>
  <div class="note warn"><b>Naming example.</b> On <code>film-02 half 001</code>, A1740 uses names and relationships: <span class="code">{html.escape(a1740_rosters)}</span>. Consol uses descriptive identities: <span class="code">{html.escape(consol_rosters)}</span>. This helps explain A1740's higher name-only score even when appearance mapping can rescue Consol.</div>

  <h2>4. Source composition and the original-v0 rank</h2>
  <p>The table reports name + appearance IoU pooled within each source. Positive gap means A1740 is better. The character column is <code>v0 / v0.2-half</code>.</p>
  {source_table}
  <div class="chart-grid"><div class="chart-wrap"><h3>A1740 minus Consol by source</h3><canvas id="source-gap-chart"></canvas></div></div>
  <div class="note"><b>Sport explains the original-v0 ordering.</b> On original v0 sport, A1740 scores {score(sport[A1740]["v0_appearance"])} from {sport[A1740]["v0_predicted_spans"]} direct spans; Consol scores {score(sport[CONSOL]["v0_appearance"])} from {sport[CONSOL]["v0_predicted_spans"]} spans. Without sport, original-v0 appearance becomes A1740 {score(analysis["v0_without_sport"][A1740]["appearance"])} versus Consol {score(analysis["v0_without_sport"][CONSOL]["appearance"])}. On nested v0.2 sport, the outputs have nearly equal {sport[A1740]["v02_predicted_spans"]} versus {sport[CONSOL]["v02_predicted_spans"]} shot spans, and A1740 leads {score(sport[A1740]["v02_appearance"])} to {score(sport[CONSOL]["v02_appearance"])}.</div>

  <h2>5. Why the two leaderboards are not interchangeable</h2>
  {scorer_table}
  <p>They reuse temporal IoU reduction, but they do not hold the prediction representation, windows, character weighting, or LLM mapper constant. A checkpoint can therefore improve direct 10-minute span extraction while getting worse at nested shot-level identity consistency.</p>

  <h2>6. Recommended follow-up</h2>
  <ol>
    <li>Report v0.2 <b>common-valid</b>, <b>known appearance</b>, and <b>unknown appearance</b> beside the official zero-penalized score.</li>
    <li>Persist the mapper's predicted-name to GT-label decisions. Current scorer payloads preserve final spans but not the LLM mapping rationale, which makes identity errors harder to audit.</li>
    <li>Add a deterministic consistency diagnostic: percentage of shot entity names found in the top-level roster, mentions per roster identity, and identities appearing in only one shot.</li>
    <li>Do not use original-v0 rank as a regression oracle for v0.2. At minimum, source-balance both and report the sport slice separately.</li>
  </ol>
  <p class="muted">Reproduce with <code>collect_v0_comparison_payloads.py</code>, <code>download_inference_metadata.py</code>, and <code>analyze_half_rank_inversion.py</code>. Scorer implementation: <a href="https://github.com/twelvelabs-io/pegasus/blob/main/eval/backend/src/eval_backend/scoring/benchmarks/coverage_with_nested_output/entity_coverage/evaluator.py">nested entity evaluator</a>.</p>
</main>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
<script>
const DATA={chart_data};
new Chart(document.getElementById("composition-chart"),{{type:"bar",data:{{labels:["Naming (known)","Appearance (known)","Appearance (unknown)","Appearance (overall)"],datasets:[{{label:"A1740 s400",data:[DATA.composition["{A1740}"].naming,DATA.composition["{A1740}"].known_appearance,DATA.composition["{A1740}"].unknown_appearance,DATA.composition["{A1740}"].appearance],backgroundColor:"#0b62d6"}},{{label:"Consol s2000",data:[DATA.composition["{CONSOL}"].naming,DATA.composition["{CONSOL}"].known_appearance,DATA.composition["{CONSOL}"].unknown_appearance,DATA.composition["{CONSOL}"].appearance],backgroundColor:"#137333"}}]}},options:{{maintainAspectRatio:false,scales:{{y:{{beginAtZero:true,max:0.45}}}}}}}});
new Chart(document.getElementById("source-gap-chart"),{{type:"bar",data:{{labels:DATA.v0Sources.map(row=>row.source),datasets:[{{label:"Original v0",data:DATA.v0Sources.map(row=>row.appearance_gap),backgroundColor:"#8a5a00"}},{{label:"v0.2 half",data:DATA.v02Sources.map(row=>row.appearance_gap),backgroundColor:"#0b62d6"}}]}},options:{{maintainAspectRatio:false,scales:{{y:{{title:{{display:true,text:"A1740 minus Consol IoU"}}}}}}}}}});
</script>
</body>
</html>
"""


def main() -> None:
    analysis = calculate_analysis()
    ANALYSIS_PATH.write_text(json.dumps(analysis, indent=2, sort_keys=True) + "\n")
    REPORT_PATH.write_text(render_report(analysis))
    print(f"Wrote {ANALYSIS_PATH}")
    print(f"Wrote {REPORT_PATH}")


if __name__ == "__main__":
    main()

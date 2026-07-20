#!/usr/bin/env python3
"""Build a per-sample entity-coverage prediction viewer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


NAMING_KEY = "entity_coverage::naming_iou"
APPEARANCE_KEY = "entity_coverage::name_appearance_iou"


def load_json_lines(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def parse_json_payload(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if not isinstance(value, str):
        raise TypeError(
            f"prediction payload must be an object or string, got {type(value).__name__}"
        )
    stripped = value.strip()
    if stripped.startswith("```"):
        stripped = (
            stripped.removeprefix("```json")
            .removeprefix("```")
            .removesuffix("```")
            .strip()
        )
    parsed = json.loads(stripped)
    if not isinstance(parsed, dict):
        raise TypeError("prediction JSON must be an object")
    return parsed


def parse_metadata(raw_row: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    metadata = raw_row.get("metadata") or {}
    if isinstance(metadata, str):
        metadata = json.loads(metadata)
    sample_metadata = metadata.get("sample_metadata") or [{}]
    media_metadata = metadata.get("media_metadata") or [{}]
    if isinstance(sample_metadata, list):
        sample_metadata = sample_metadata[0] if sample_metadata else {}
    if isinstance(media_metadata, list):
        media_metadata = media_metadata[0] if media_metadata else {}
    return sample_metadata, media_metadata


def normalize_entity_payload(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "domain": payload.get("domain"),
        "roster": payload.get("roster") or [],
        "spans": payload.get("spans") or [],
    }


def build_samples(arguments: argparse.Namespace) -> list[dict[str, Any]]:
    pegasus_records = {
        record["sample_id"]: record
        for record in load_json_lines(arguments.pegasus_predictions)
    }
    gemini_records = {
        record["sample_id"]: record
        for record in load_json_lines(arguments.gemini_predictions)
    }
    pegasus_scores = json.loads(arguments.pegasus_persample.read_text())
    gemini_audit = json.loads(arguments.gemini_audit.read_text())
    gemini_scores = {sample["sample_id"]: sample for sample in gemini_audit["samples"]}

    sample_ids = set(pegasus_records)
    for label, records in (
        ("Gemini predictions", gemini_records),
        ("Pegasus scores", pegasus_scores),
        ("Gemini scores", gemini_scores),
    ):
        if set(records) != sample_ids:
            missing = sorted(sample_ids - set(records))
            extra = sorted(set(records) - sample_ids)
            raise ValueError(
                f"{label} sample mismatch: missing={missing}, extra={extra}"
            )

    samples = []
    for sample_id in sorted(sample_ids):
        pegasus_record = pegasus_records[sample_id]
        gemini_record = gemini_records[sample_id]
        raw_row = gemini_record["raw_row"]
        sample_metadata, media_metadata = parse_metadata(raw_row)
        ground_truth = parse_json_payload(sample_metadata["ground_truth"])
        pegasus_response = pegasus_record.get("parsed_response") or {}
        pegasus_payload = parse_json_payload(pegasus_response.get("text"))
        gemini_payload = parse_json_payload(gemini_record["output"]["text"])
        duration_seconds = float(
            sample_metadata.get("chunk_duration_seconds")
            or float(sample_metadata.get("chunk_end_seconds") or 0)
            - float(sample_metadata.get("chunk_start_seconds") or 0)
        )
        if duration_seconds <= 0:
            all_spans = [
                *(ground_truth.get("spans") or []),
                *(pegasus_payload.get("spans") or []),
                *(gemini_payload.get("spans") or []),
            ]
            duration_seconds = max(
                (float(span.get("end") or 0) for span in all_spans), default=1.0
            )

        pegasus_sample_score = pegasus_scores[sample_id]
        gemini_sample_score = gemini_scores[sample_id]
        samples.append(
            {
                "sample_id": sample_id,
                "duration_seconds": duration_seconds,
                "media_path": media_metadata.get("media_path")
                or (raw_row.get("media") or [{}])[0].get("media_path"),
                "source_url": sample_metadata.get("source_video_url")
                or sample_metadata.get("video_url"),
                "ground_truth": normalize_entity_payload(ground_truth),
                "pegasus": {
                    **normalize_entity_payload(pegasus_payload),
                    "metrics": pegasus_sample_score.get("metrics") or {},
                    "counts": pegasus_sample_score.get("counts") or {},
                    "character_scores": pegasus_sample_score.get("character_scores")
                    or [],
                    "finish_reason": pegasus_response.get("finish_reason"),
                    "output_tokens": pegasus_response.get("output_tokens"),
                },
                "gemini": {
                    **normalize_entity_payload(gemini_payload),
                    "metrics": {
                        NAMING_KEY: gemini_sample_score["asr_naming_iou"],
                        APPEARANCE_KEY: gemini_sample_score["asr_appearance_iou"],
                    },
                    "counts": {
                        "predicted_entities": gemini_sample_score["asr_roster_count"],
                        "predicted_spans": gemini_sample_score["asr_span_count"],
                        "ground_truth_entities": gemini_sample_score[
                            "ground_truth_roster_count"
                        ],
                        "ground_truth_spans": gemini_sample_score[
                            "ground_truth_span_count"
                        ],
                    },
                    "asr_segment_count": gemini_sample_score["asr_segment_count"],
                },
            }
        )
    return samples


HTML_TEMPLATE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Entity coverage prediction viewer: Pegasus-15 vs Gemini 3 Flash + ASR</title>
<style>
:root{font-family:ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif;color:#172126;background:#f4f6f7;--gt:#2878b5;--pegasus:#087f5b;--gemini:#d15c3f;--border:#ccd5d9}
*{box-sizing:border-box}body{margin:0}main{width:min(1500px,calc(100% - 32px));margin:28px auto 60px}h1{font-size:26px;letter-spacing:0;margin:0 0 6px}h2,h3{letter-spacing:0}.subhead{color:#526168;margin:0 0 18px}.controls{display:flex;gap:10px;align-items:center;flex-wrap:wrap;padding:12px 0;border-top:1px solid var(--border);border-bottom:1px solid var(--border);position:sticky;top:0;background:#f4f6f7;z-index:10}.controls input,.controls select{height:36px;border:1px solid #aeb9be;background:#fff;padding:0 10px;font:inherit}.controls input{width:min(360px,100%)}button{height:36px;border:1px solid #8f9da4;background:#fff;padding:0 12px;font-weight:650;cursor:pointer}.count{margin-left:auto;color:#526168;font-size:13px}.aggregate{display:flex;gap:20px;flex-wrap:wrap;margin:14px 0 18px;font-size:13px}.aggregate strong{font-size:17px;display:block}.sample{border-top:1px solid var(--border);background:#fff}.sample:last-child{border-bottom:1px solid var(--border)}.sample>summary{cursor:pointer;padding:12px 14px;display:flex;gap:10px;align-items:center;flex-wrap:wrap;list-style:none}.sample>summary::-webkit-details-marker{display:none}.sample>summary::before{content:"+";width:20px;font-weight:800;color:#63727a}.sample[open]>summary::before{content:"−"}.sample-id{font:650 12px ui-monospace,SFMono-Regular,Menlo,monospace;margin-right:auto}.badge{font-size:12px;border:1px solid var(--border);padding:3px 7px;background:#f7f9fa;font-variant-numeric:tabular-nums}.winner{background:#e5f4ed;border-color:#88bea9}.sample-body{padding:0 14px 18px}.sample-meta{font-size:12px;color:#5c6970;margin:0 0 14px;overflow-wrap:anywhere}.sample-meta a{color:#0563c1}.comparison{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:16px}.model-panel{min-width:0}.model-panel h3{font-size:15px;margin:0 0 8px;padding-bottom:7px;border-bottom:3px solid}.model-panel.gt h3{border-color:var(--gt)}.model-panel.pegasus h3{border-color:var(--pegasus)}.model-panel.gemini h3{border-color:var(--gemini)}.metrics{display:flex;gap:6px;flex-wrap:wrap;min-height:27px;margin-bottom:8px}.metrics span{font-size:11px;background:#edf1f2;padding:4px 6px}.timeline{margin:10px 0 14px}.axis,.track-row{display:grid;grid-template-columns:112px minmax(0,1fr);gap:7px;align-items:center}.axis{font-size:10px;color:#68767d}.ticks{display:flex;justify-content:space-between;border-bottom:1px solid #aeb9be;padding-bottom:3px}.track-row{margin:4px 0}.track-label{font-size:11px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;text-align:right}.track{height:17px;position:relative;background:#edf1f2;overflow:hidden;border-left:1px solid #aeb9be;border-right:1px solid #aeb9be}.segment{position:absolute;top:3px;height:11px;min-width:2px;opacity:.88}.gt .segment{background:var(--gt)}.pegasus .segment{background:var(--pegasus)}.gemini .segment{background:var(--gemini)}table{width:100%;border-collapse:collapse;font-size:11px;table-layout:fixed}th,td{padding:5px 6px;border-bottom:1px solid #dce2e5;text-align:left;vertical-align:top;overflow-wrap:anywhere}th{background:#edf1f2}th:nth-child(1){width:28%}th:nth-child(2){width:17%}th:nth-child(3){width:12%}.appearance{color:#536168}.raw{margin-top:10px}.raw summary{cursor:pointer;font-size:12px;font-weight:650;color:#35444b}.raw pre{margin:7px 0 0;background:#101820;color:#eaf0f2;padding:10px;max-height:420px;overflow:auto;font:11px/1.45 ui-monospace,SFMono-Regular,Menlo,monospace;white-space:pre-wrap}.empty{font-size:12px;color:#7a878d;padding:10px 0}.hidden{display:none!important}
@media(max-width:1050px){.comparison{grid-template-columns:1fr}.model-panel{padding-top:10px}.count{width:100%;margin-left:0}}
@media(max-width:560px){main{width:358px;max-width:calc(100% - 20px);margin:18px 10px 40px}h1{font-size:21px;overflow-wrap:anywhere}.controls{position:static}.controls input,.controls select,button{width:100%}.sample>summary{align-items:flex-start}.sample-id{width:100%;overflow-wrap:anywhere}.axis,.track-row{grid-template-columns:88px minmax(0,1fr)}th:nth-child(4),td:nth-child(4){display:none}}
</style></head><body><main>
<h1>Pegasus-15 vs Gemini 3 Flash + ASR</h1>
<p class="subhead">Actual entity-coverage inference outputs, aligned with ground truth for every chunk_10m/test sample.</p>
<div class="controls"><input id="search" type="search" placeholder="Filter sample ID or predicted name"><select id="sort"><option value="id">Sort: sample ID</option><option value="pegasus">Sort: Pegasus naming advantage</option><option value="gemini">Sort: Gemini naming advantage</option><option value="low">Sort: lowest combined naming score</option></select><button id="expand">Expand all</button><button id="collapse">Collapse all</button><span class="count" id="count"></span></div>
<div class="aggregate"><span>Pegasus-15 naming<strong>0.2728</strong></span><span>Gemini naming<strong>0.2396</strong></span><span>Pegasus-15 naming + appearance<strong>0.4556</strong></span><span>Gemini naming + appearance<strong>0.3383</strong></span></div>
<div id="samples"></div>
</main><script>
window.addEventListener("error",event=>{const target=document.querySelector("#samples");if(target)target.innerHTML=`<p class="empty">Viewer error: ${String(event.message)}</p>`});
const DATA=__VIEWER_DATA__;
const NAMING="entity_coverage::naming_iou",APPEARANCE="entity_coverage::name_appearance_iou";
const escapeHtml=value=>String(value??"").replace(/[&<>"']/g,character=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"})[character]);
const score=value=>Number(value??0).toFixed(4);
const formatTime=seconds=>{const value=Math.max(0,Number(seconds)||0),minutes=Math.floor(value/60);return `${minutes}:${String(Math.floor(value%60)).padStart(2,"0")}`};
const appearanceText=value=>{if(!value)return "";if(typeof value==="string")return value;return Object.entries(value).map(([key,item])=>`${key}: ${item}`).join(" · ")};
function rosterTable(model){if(!model.roster.length)return '<div class="empty">No roster entries</div>';return `<table><thead><tr><th>Name</th><th>Label</th><th>Known</th><th>Appearance</th></tr></thead><tbody>${model.roster.map(entity=>`<tr><td><strong>${escapeHtml(entity.name)}</strong>${entity.name_evidence?`<br><span class="appearance">${escapeHtml(entity.name_evidence)}</span>`:""}</td><td>${escapeHtml(entity.label_id)}</td><td>${entity.name_known?"yes":"no"}</td><td class="appearance">${escapeHtml(appearanceText(entity.appearance))}</td></tr>`).join("")}</tbody></table>`}
function timeline(model,duration){if(!model.roster.length)return '<div class="empty">No timeline</div>';const spansByLabel={};for(const span of model.spans){(spansByLabel[String(span.label_id)]??=[]).push(span)}const ticks=[0,.25,.5,.75,1].map(part=>`<span>${formatTime(duration*part)}</span>`).join("");const rows=model.roster.map(entity=>{const segments=(spansByLabel[String(entity.label_id)]||[]).map(span=>{const start=Math.max(0,Math.min(duration,Number(span.start)||0)),end=Math.max(start,Math.min(duration,Number(span.end)||0));return `<i class="segment" style="left:${(start/duration*100).toFixed(3)}%;width:${Math.max(.15,(end-start)/duration*100).toFixed(3)}%" title="${escapeHtml(entity.name)} ${formatTime(span.start)}–${formatTime(span.end)}"></i>`}).join("");return `<div class="track-row"><span class="track-label" title="${escapeHtml(entity.name)}">${escapeHtml(entity.name||entity.label_id)}</span><div class="track">${segments}</div></div>`}).join("");return `<div class="timeline"><div class="axis"><span></span><div class="ticks">${ticks}</div></div>${rows}</div>`}
function metricLine(model,kind){if(kind==="gt")return `<div class="metrics"><span>roster ${model.roster.length}</span><span>spans ${model.spans.length}</span></div>`;return `<div class="metrics"><span>Naming ${score(model.metrics[NAMING])}</span><span>Name + appearance ${score(model.metrics[APPEARANCE])}</span><span>roster ${model.roster.length}</span><span>spans ${model.spans.length}</span>${model.asr_segment_count!==undefined?`<span>ASR segments ${model.asr_segment_count}</span>`:""}${model.output_tokens?`<span>output tokens ${model.output_tokens}</span>`:""}</div>`}
function modelPanel(title,kind,model,duration){const rawPayload={domain:model.domain,roster:model.roster,spans:model.spans},rawLabel=kind==="gt"?"Raw ground-truth JSON":"Raw inference output JSON";return `<section class="model-panel ${kind}"><h3>${escapeHtml(title)}</h3>${metricLine(model,kind)}${timeline(model,duration)}${rosterTable(model)}<details class="raw"><summary>${rawLabel}</summary><pre>${escapeHtml(JSON.stringify(rawPayload,null,2))}</pre></details></section>`}
function sampleSearchText(sample){return [sample.sample_id,...sample.ground_truth.roster.map(item=>item.name),...sample.pegasus.roster.map(item=>item.name),...sample.gemini.roster.map(item=>item.name)].join(" ").toLowerCase()}
function renderSample(sample,index){const pegasusNaming=sample.pegasus.metrics[NAMING]??0,geminiNaming=sample.gemini.metrics[NAMING]??0;return `<details class="sample" data-search="${escapeHtml(sampleSearchText(sample))}" ${index===0?"open":""}><summary><span class="sample-id">${escapeHtml(sample.sample_id)}</span><span class="badge ${pegasusNaming>geminiNaming?"winner":""}">Pegasus naming ${score(pegasusNaming)}</span><span class="badge ${geminiNaming>pegasusNaming?"winner":""}">Gemini naming ${score(geminiNaming)}</span><span class="badge">duration ${formatTime(sample.duration_seconds)}</span></summary><div class="sample-body"><p class="sample-meta">${escapeHtml(sample.media_path)}${sample.source_url?` · <a href="${escapeHtml(sample.source_url)}" target="_blank" rel="noreferrer">source video</a>`:""}</p><div class="comparison">${modelPanel("Ground truth","gt",sample.ground_truth,sample.duration_seconds)}${modelPanel("Pegasus-15","pegasus",sample.pegasus,sample.duration_seconds)}${modelPanel("Gemini 3 Flash + ASR","gemini",sample.gemini,sample.duration_seconds)}</div></div></details>`}
function sortedData(){const mode=document.querySelector("#sort").value;return [...DATA].sort((left,right)=>{const leftDelta=(left.pegasus.metrics[NAMING]??0)-(left.gemini.metrics[NAMING]??0),rightDelta=(right.pegasus.metrics[NAMING]??0)-(right.gemini.metrics[NAMING]??0);if(mode==="pegasus")return rightDelta-leftDelta;if(mode==="gemini")return leftDelta-rightDelta;if(mode==="low")return ((left.pegasus.metrics[NAMING]??0)+(left.gemini.metrics[NAMING]??0))-((right.pegasus.metrics[NAMING]??0)+(right.gemini.metrics[NAMING]??0));return left.sample_id.localeCompare(right.sample_id)})}
function render(){const query=document.querySelector("#search").value.trim().toLowerCase(),filtered=sortedData().filter(sample=>!query||sampleSearchText(sample).includes(query));document.querySelector("#samples").innerHTML=filtered.map(renderSample).join("");document.querySelector("#count").textContent=`${filtered.length} / ${DATA.length} samples`}
document.querySelector("#search").addEventListener("input",render);document.querySelector("#sort").addEventListener("change",render);document.querySelector("#expand").addEventListener("click",()=>document.querySelectorAll(".sample").forEach(item=>item.open=true));document.querySelector("#collapse").addEventListener("click",()=>document.querySelectorAll(".sample").forEach(item=>item.open=false));render();
</script></body></html>"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pegasus-predictions", type=Path, required=True)
    parser.add_argument("--pegasus-persample", type=Path, required=True)
    parser.add_argument("--gemini-predictions", type=Path, required=True)
    parser.add_argument("--gemini-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()

    samples = build_samples(arguments)
    embedded_data = json.dumps(
        samples, ensure_ascii=False, separators=(",", ":")
    ).replace("</", "<\\/")
    arguments.output.write_text(
        HTML_TEMPLATE.replace("__VIEWER_DATA__", embedded_data), encoding="utf-8"
    )
    print(f"wrote {arguments.output} with {len(samples)} samples")


if __name__ == "__main__":
    main()

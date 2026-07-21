#!/usr/bin/env python3
"""Build a searchable HTML viewer for raw Pegasus metadata outputs."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any


def escaped(value: Any) -> str:
    return html.escape(str(value if value is not None else ""))


def family(source_url: str) -> str:
    if "/editstock/" in source_url:
        return "EditStock"
    if "/tv-series/arcane/" in source_url:
        return "Arcane"
    if "/tv-series/game_of_thrones/" in source_url:
        return "Game of Thrones"
    if "/soccer-replay-1988/" in source_url:
        return "Soccer"
    return "Other"


def raw_output(output_url: str, raw_output_dir: Path) -> dict[str, Any]:
    with (raw_output_dir / Path(output_url).name).open(encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, dict):
        raise ValueError("raw output must be a JSON object")
    return payload


def entity_list(entities: Any) -> str:
    if not isinstance(entities, list) or not entities:
        return "<span class=\"empty\">None</span>"
    rows = []
    for entity in entities:
        if not isinstance(entity, dict):
            continue
        tags = ", ".join(str(tag) for tag in entity.get("tags", []) if tag)
        rows.append(
            "<li><b>" + escaped(entity.get("canonical_name")) + "</b>"
            + (f" <span class=\"type\">{escaped(entity.get('entity_type'))}</span>" if entity.get("entity_type") else "")
            + (f" <span class=\"tag\">{escaped(tags)}</span>" if tags else "")
            + (f"<br><span>{escaped(entity.get('appearance'))}</span>" if entity.get("appearance") else "")
            + "</li>"
        )
    return "<ul class=\"entities\">" + "".join(rows) + "</ul>"


def relationship_list(relationships: Any) -> str:
    if not isinstance(relationships, list) or not relationships:
        return "<span class=\"empty\">None</span>"
    return "<pre>" + escaped(json.dumps(relationships, ensure_ascii=False, indent=2)) + "</pre>"


def shot_table(shots: Any) -> str:
    if not isinstance(shots, list) or not shots:
        return "<span class=\"empty\">No shot metadata</span>"
    rows = []
    for shot in shots:
        if not isinstance(shot, dict):
            continue
        rows.append(
            "<tr>"
            f"<td>{escaped(shot.get('start_time'))}–{escaped(shot.get('end_time'))}</td>"
            f"<td>{escaped(shot.get('shot_type'))}</td>"
            f"<td>{escaped(shot.get('camera_motion'))}<br>{escaped(shot.get('camera_angle'))}</td>"
            f"<td>{escaped(shot.get('shot_summary'))}</td>"
            f"<td>{entity_list(shot.get('entities'))}</td>"
            "</tr>"
        )
    return """<table class=\"shots\"><thead><tr><th>Time (s)</th><th>Shot type</th><th>Camera</th><th>Model summary</th><th>Entities</th></tr></thead><tbody>""" + "".join(rows) + "</tbody></table>"


def sample_html(record: dict[str, Any], payload: dict[str, Any]) -> str:
    source_url = str(record.get("source_url") or "")
    output_url = str(record.get("output_url") or "")
    segments = payload.get("segments", {})
    video_segments = segments.get("video", []) if isinstance(segments, dict) else []
    video_segments = video_segments if isinstance(video_segments, list) else []
    metrics = payload.get("metrics", {}) if isinstance(payload.get("metrics"), dict) else {}
    errors = payload.get("definition_errors", [])
    errors = errors if isinstance(errors, list) else [errors]
    segment_sections = []
    for index, segment in enumerate(video_segments, start=1):
        if not isinstance(segment, dict):
            continue
        metadata = segment.get("metadata", {}) if isinstance(segment.get("metadata"), dict) else {}
        segment_sections.append(
            f"""<section class=\"segment\"><h3>Output segment {index}: {escaped(segment.get('start_time'))}–{escaped(segment.get('end_time'))} s</h3>
<p class=\"summary\">{escaped(metadata.get('video_summary'))}</p>
<div class=\"metadata-grid\"><section><h4>Entities ({len(metadata.get('rosters', [])) if isinstance(metadata.get('rosters'), list) else 0})</h4>{entity_list(metadata.get('rosters'))}</section>
<section><h4>Relationships</h4>{relationship_list(metadata.get('entity_relationships'))}</section></div>
<h4>Shot-level model output ({len(metadata.get('shot_metadata', [])) if isinstance(metadata.get('shot_metadata'), list) else 0})</h4>{shot_table(metadata.get('shot_metadata'))}</section>"""
        )
    error_html = "" if not errors else "<div class=\"error\"><b>Definition error:</b><br>" + "<br>".join(escaped(error) for error in errors) + "</div>"
    duration = " / ".join(f"{escaped(segment.get('start_time'))}–{escaped(segment.get('end_time'))}s" for segment in video_segments if isinstance(segment, dict)) or "no parsed segment"
    search_text = " ".join([source_url, str(record.get("job_id") or ""), family(source_url), json.dumps(payload.get("segments", {}), ensure_ascii=False)])
    return f"""<details class=\"sample\" data-search=\"{escaped(search_text).lower()}\">
<summary><span class=\"sample-title\">{escaped(Path(source_url).name)}</span><span class=\"pill\">{escaped(family(source_url))}</span><span class=\"pill\">{escaped(duration)}</span><span class=\"pill\">{len(video_segments)} segment(s)</span><span class=\"pill\">{escaped(metrics.get('finish_reason'))}</span></summary>
<div class=\"body\"><div class=\"source\"><b>Source:</b> <code>{escaped(source_url)}</code><br><b>Job:</b> <code>{escaped(record.get('job_id'))}</code><br><b>Raw output:</b> <code>{escaped(output_url)}</code></div>
<div class=\"metrics\"><span>Input tokens <b>{escaped(metrics.get('input_token_count'))}</b></span><span>Output tokens <b>{escaped(metrics.get('token_count'))}</b></span><span>Output characters <b>{escaped(metrics.get('total_output_chars'))}</b></span></div>{error_html}{''.join(segment_sections) or '<p class="empty">No parsed model metadata was produced.</p>'}</div></details>"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--raw-output-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    if not isinstance(manifest, list):
        raise ValueError("manifest must be an array")
    samples = []
    for record in manifest:
        if not isinstance(record, dict):
            continue
        samples.append(sample_html(record, raw_output(str(record.get("output_url") or ""), args.raw_output_dir)))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        """<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\"><title>Pegasus raw output viewer</title>
<style>
body{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;color:#24292f;max-width:1600px;margin:28px} h1{font-size:24px;margin-bottom:6px} h2{font-size:16px} .note{color:#57606a;line-height:1.55}.controls{position:sticky;top:0;background:#fff;padding:12px 0;border-bottom:1px solid #d0d7de;z-index:2}.controls input{width:min(760px,90%);padding:9px 11px;border:1px solid #8c959f;border-radius:6px;font-size:14px}.sample{border:1px solid #d0d7de;border-radius:8px;margin:12px 0;background:#fff}.sample summary{cursor:pointer;padding:11px 12px;background:#f6f8fa;border-radius:8px;font-size:13px}.sample[open] summary{border-bottom:1px solid #d0d7de;border-radius:8px 8px 0 0}.sample-title{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-weight:600}.pill{display:inline-block;border:1px solid #d0d7de;border-radius:999px;padding:2px 7px;margin-left:5px;background:#fff;font-size:11px;color:#57606a}.body{padding:12px}.source{font-size:12px;line-height:1.55;overflow-wrap:anywhere}.metrics{display:flex;gap:9px;flex-wrap:wrap;margin:12px 0}.metrics span{background:#ddf4ff;border-radius:6px;padding:5px 8px;font-size:12px}.segment{border-top:1px solid #d8dee4;margin-top:16px;padding-top:10px}.segment h3{font-size:14px}.segment h4{font-size:13px;margin-bottom:6px}.summary{background:#f6f8fa;padding:10px 12px;border-radius:6px;line-height:1.55;font-size:13px;white-space:pre-wrap}.metadata-grid{display:grid;grid-template-columns:1fr 1fr;gap:14px}.entities{padding-left:20px;font-size:12px;line-height:1.45}.entities li{margin-bottom:7px}.type,.tag{font-size:10px;background:#eaeef2;border-radius:999px;padding:2px 5px;margin-left:4px}.tag{background:#ddf4ff}.shots{border-collapse:collapse;width:100%;font-size:11px}.shots th,.shots td{border:1px solid #d0d7de;padding:5px 7px;text-align:left;vertical-align:top}.shots th{background:#f6f8fa}.shots td:first-child{white-space:nowrap}.shots .entities{margin:0;padding-left:14px}code{font-size:11px}pre{font-size:11px;white-space:pre-wrap;overflow-wrap:anywhere;background:#f6f8fa;padding:8px;border-radius:6px}.empty{color:#57606a;font-style:italic}.error{background:#ffebe9;border:1px solid #cf222e;border-radius:6px;padding:10px;margin:12px 0;font-size:12px}@media(max-width:900px){.metadata-grid{grid-template-columns:1fr}.shots{display:block;overflow-x:auto}}
</style></head><body><h1>Pegasus raw model-output viewer — assembly-v0, chunk duration 1,200s</h1>
<p class=\"note\">Each expandable row is one indexing job from <code>raw_output_manifest.json</code>. It shows the final metadata produced by the model—not the 16 final benchmark answers. Use search for a video filename, family, job ID, entity, or words in the model output.</p>
<div class=\"controls\"><input id=\"search\" type=\"search\" placeholder=\"Filter 233 raw outputs (e.g. Arcane, knife, S01E05, job ID)\"><span id=\"count\"></span></div><main>"""
        + "\n".join(samples)
        + """</main><script>const samples=[...document.querySelectorAll('.sample')];const count=document.querySelector('#count');function filter(){const query=document.querySelector('#search').value.toLowerCase().trim();let visible=0;for(const sample of samples){const show=!query||sample.dataset.search.includes(query);sample.hidden=!show;if(show)visible++;}count.textContent=` ${visible} / ${samples.length} shown`;};document.querySelector('#search').addEventListener('input',filter);filter();</script></body></html>""",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Build a side-by-side visualization of chunk_05m predictions:
  GT vs gemini-2.5-pro vs pegasus-15 (rep. Pegasus RL model).

Purpose: understand why gemini-2.5-pro has the highest naming but the
lowest name_appearance_iou compared to Pegasus. HTML shows per-sample
timeline of GT spans / Gemini pred spans / Pegasus pred spans per
character, plus summary counts (# roster, # spans, total span duration).

Output: 260713_gemini_vs_pegasus_chunk05m.html in the same directory.
"""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

TDF_PATH = Path(
    "/Users/long8v/pegasus-vcs-2339-entity-eval/data/entity-eval/output/chunked/entity_coverage_v0_chunk_05m.tdf.jsonl"
)
RAW_ROOT = Path(
    "/Users/long8v/pegasus-vcs-2339-entity-eval/eval/eval-service/eval_output/"
    "ckpt-comparison-20260709-chunk10_20_45-recovered/raw_outputs/chunk05m"
)
SCORED_ROOT = Path(
    "/Users/long8v/pegasus-vcs-2339-entity-eval/eval/eval-service/eval_output/"
    "ckpt-comparison-20260709-chunk10_20_45-recovered/scored/chunk05m"
)
MODELS = [
    ("gemini-2.5-pro", "#c02a2a"),
    ("pegasus-15", "#f28e2b"),
    ("pegasus-15-sft", "#2ca02c"),
    ("pegasus-15-rl", "#9467bd"),
    ("entity-h0-sme-2200", "#17becf"),
]
GT_COLOR = "#1f77b4"

OUT_HTML = Path(__file__).resolve().parent / "260713_gemini_vs_pegasus_chunk05m.html"


def load_tdf() -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for line in TDF_PATH.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        rows[row["id"]] = row
    return rows


def parse_gt(row: dict[str, Any]) -> dict[str, Any]:
    """Ground truth from TDF row.

    entity_cov_v0_tdf stores GT as a fenced JSON in the assistant message's
    text (messages[1].content[*].text), not in metadata.
    """
    for msg in row.get("messages", []):
        if msg.get("role") != "assistant":
            continue
        content = msg.get("content") or []
        if isinstance(content, list):
            for c in content:
                text = c.get("text") if isinstance(c, dict) else None
                if not text:
                    continue
                return _parse_text_payload(text)
        elif isinstance(content, str):
            return _parse_text_payload(content)
    return {"roster": [], "spans": [], "domain": "?"}


def _parse_text_payload(text: str) -> dict[str, Any]:
    stripped = text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        try:
            return json.loads(stripped[stripped.index("{") : stripped.rindex("}") + 1])
        except Exception:  # noqa: BLE001
            return {"roster": [], "spans": [], "_parse_error": True, "_raw": stripped[:400]}


def load_model_predictions(model_name: str) -> dict[str, dict[str, Any]]:
    """Load predictions for one model.

    Prefer `scored/chunk05m/<model>/predictions.jsonl` (Pegasus recovery
    pipeline emits this and it already carries sample_id ↔ raw output).
    Fall back to `raw_outputs/chunk05m/<model>/*.json` (Gemini caller emits
    this with `_sample_id` embedded).
    """
    preds: dict[str, dict[str, Any]] = {}

    scored_path = SCORED_ROOT / model_name / "predictions.jsonl"
    if scored_path.exists():
        for line in scored_path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            sid = rec.get("sample_id") or rec.get("id")
            output = rec.get("output") or {}
            text = output.get("text") if isinstance(output, dict) else ""
            if not sid or not text:
                continue
            preds[sid] = _parse_text_payload(text)
        return preds

    raw_dir = RAW_ROOT / model_name
    if not raw_dir.exists():
        return preds
    for path in sorted(raw_dir.glob("*.json")):
        try:
            blob = json.loads(path.read_text())
        except Exception:  # noqa: S112, BLE001
            continue
        sid = blob.get("_sample_id")
        text = blob.get("text", "")
        if not sid or not text:
            continue
        preds[sid] = _parse_text_payload(text)
    return preds


def span_stats(payload: dict[str, Any]) -> dict[str, Any]:
    roster = payload.get("roster") or []
    spans = payload.get("spans") or []
    total_seconds = 0.0
    per_char: dict[str, list[tuple[float, float]]] = {}
    for s in spans:
        try:
            start = float(s.get("start", 0.0))
            end = float(s.get("end", 0.0))
        except (TypeError, ValueError):
            continue
        if end <= start:
            continue
        total_seconds += end - start
        per_char.setdefault(str(s.get("label_id", "?")), []).append((start, end))
    return {
        "n_roster": len(roster),
        "n_spans": len(spans),
        "total_seconds": total_seconds,
        "per_char": per_char,
        "roster": roster,
    }


def roster_char_labels(roster: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for r in roster:
        label_id = str(r.get("label_id", "?"))
        out[label_id] = {
            "name": r.get("name", "?"),
            "name_known": r.get("name_known"),
            "appearance": r.get("appearance", {}),
            "domain": r.get("domain", ""),
        }
    return out


def render_timeline_bars(max_seconds: float, spans_per_role: list[tuple[str, str, list[tuple[float, float]]]]) -> str:
    """spans_per_role = [(label, color, [(start,end),...]), ...]"""
    parts = [
        f"<div class='timeline'><div class='timeline-axis'>"
        f"<span class='tick' style='left:0%'>0s</span>"
        f"<span class='tick' style='left:25%'>{max_seconds * 0.25:.0f}s</span>"
        f"<span class='tick' style='left:50%'>{max_seconds * 0.50:.0f}s</span>"
        f"<span class='tick' style='left:75%'>{max_seconds * 0.75:.0f}s</span>"
        f"<span class='tick' style='left:100%'>{max_seconds:.0f}s</span>"
        f"</div>"
    ]
    for label, color, intervals in spans_per_role:
        bar_html = "".join(
            f"<span class='seg' style='left:{max(0, start) / max_seconds * 100:.2f}%;"
            f"width:{max(0, end - start) / max_seconds * 100:.2f}%;background:{color}'"
            f" title='{start:.1f}s → {end:.1f}s ({end - start:.1f}s)'></span>"
            for start, end in intervals
        )
        parts.append(
            f"<div class='timeline-row'>"
            f"<span class='timeline-label'>{html.escape(label)}</span>"
            f"<span class='timeline-bar'>{bar_html}</span>"
            f"</div>"
        )
    parts.append("</div>")
    return "".join(parts)


def build_sample_card(sample_id: str, row: dict[str, Any], model_preds: dict[str, dict[str, Any]]) -> str:
    gt = parse_gt(row)
    gt_stats = span_stats(gt)
    media = (row.get("media") or [{}])[0]
    media_path = media.get("media_path", "")
    duration_hint = 300.0  # chunk_05m == 300s nominal
    # collect all seconds to derive max
    max_seconds = duration_hint
    for label, _ in [("gt", "")] + [(m, c) for m, c in MODELS]:
        stats = gt_stats if label == "gt" else span_stats(model_preds.get(label, {}))
        for _, intervals in stats["per_char"].items():
            for _, e in intervals:
                max_seconds = max(max_seconds, e)

    # Summary counts table for the sample
    summary_rows = [
        (
            "GT",
            gt_stats["n_roster"],
            gt_stats["n_spans"],
            gt_stats["total_seconds"],
            "-",
        )
    ]
    per_model_stats: dict[str, dict[str, Any]] = {}
    for model_name, _ in MODELS:
        payload = model_preds.get(model_name, {})
        if "_parse_error" in payload:
            summary_rows.append((model_name, "-", "-", "-", "parse_error"))
        else:
            stats = span_stats(payload)
            per_model_stats[model_name] = stats
            summary_rows.append((model_name, stats["n_roster"], stats["n_spans"], stats["total_seconds"], ""))

    summary_html = (
        "<table class='sum'><thead><tr><th>source</th><th># roster</th><th># spans</th>"
        "<th>total span (s)</th><th>note</th></tr></thead><tbody>"
        + "".join(
            f"<tr class='{'gt' if r[0] == 'GT' else ''}'><td>{html.escape(str(r[0]))}</td>"
            f"<td>{r[1]}</td><td>{r[2]}</td>"
            f"<td>{(r[3] if isinstance(r[3], str) else f'{r[3]:.1f}')}</td>"
            f"<td class='note'>{html.escape(str(r[4]))}</td></tr>"
            for r in summary_rows
        )
        + "</tbody></table>"
    )

    # Timeline: one row per (source × character). Group by source for readability.
    timeline_sections: list[str] = []
    gt_roster_by_id = roster_char_labels(gt.get("roster", []))
    # GT section
    gt_rows = []
    for label_id, intervals in gt_stats["per_char"].items():
        char_meta = gt_roster_by_id.get(label_id, {})
        display = f"{char_meta.get('name', label_id)} ({label_id})"
        gt_rows.append((display, GT_COLOR, intervals))
    if gt_rows:
        timeline_sections.append("<h4>GT</h4>" + render_timeline_bars(max_seconds, gt_rows))
    # Each model
    for model_name, color in MODELS:
        stats = per_model_stats.get(model_name)
        if stats is None:
            continue
        model_roster_by_id = roster_char_labels(stats["roster"])
        rows_ = []
        for label_id, intervals in stats["per_char"].items():
            char_meta = model_roster_by_id.get(label_id, {})
            display = f"{char_meta.get('name', label_id)} ({label_id})"
            rows_.append((display, color, intervals))
        if rows_:
            timeline_sections.append(
                f"<h4 style='color:{color}'>{html.escape(model_name)}</h4>" + render_timeline_bars(max_seconds, rows_)
            )

    # Roster comparison table
    def roster_json_block(title: str, roster: list[dict[str, Any]], color: str) -> str:
        rows = []
        for r in roster:
            rows.append(
                f"<tr><td style='background:{color}22'>{html.escape(str(r.get('name', '?')))}</td>"
                f"<td>{'✓' if r.get('name_known') else ''}</td>"
                f"<td>{html.escape(str(r.get('label_id', '?')))}</td>"
                f"<td class='mono'>{html.escape(json.dumps(r.get('appearance', {}), ensure_ascii=False))[:180]}</td>"
                f"<td class='mono'>{html.escape(str(r.get('name_evidence', '') or ''))[:180]}</td></tr>"
            )
        body = "".join(rows) if rows else "<tr><td colspan=5><i>(empty)</i></td></tr>"
        return (
            f"<div class='roster-block'><h4 style='color:{color}'>{html.escape(title)}</h4>"
            f"<table class='roster'><thead><tr><th>name</th><th>known?</th><th>label_id</th>"
            f"<th>appearance</th><th>name_evidence</th></tr></thead><tbody>{body}</tbody></table></div>"
        )

    roster_html = roster_json_block("GT", gt.get("roster", []), GT_COLOR)
    for model_name, color in MODELS:
        payload = model_preds.get(model_name, {})
        if "_parse_error" in payload:
            roster_html += (
                f"<div class='roster-block'><h4 style='color:{color}'>{html.escape(model_name)} — parse error</h4>"
                f"<pre>{html.escape(str(payload.get('_raw', ''))[:400])}</pre></div>"
            )
        else:
            roster_html += roster_json_block(model_name, payload.get("roster", []), color)

    return (
        f"<details class='sample'><summary><b>{html.escape(sample_id)}</b> "
        f"<span class='meta'>media: {html.escape(media_path)}</span></summary>"
        f"<div class='body'>"
        f"<h3>Summary counts</h3>{summary_html}"
        f"<h3>Timelines (per character)</h3>{''.join(timeline_sections)}"
        f"<h3>Roster comparison</h3><div class='rosters'>{roster_html}</div>"
        f"</div></details>"
    )


def build_global_summary(all_stats: dict[str, list[dict[str, Any]]]) -> str:
    """all_stats: {model or 'GT': [ {n_roster, n_spans, total_seconds}, ... per sample]}"""
    rows = []
    order = ["GT"] + [m for m, _ in MODELS]
    for src in order:
        stats_list = all_stats.get(src, [])
        if not stats_list:
            continue
        n_samples = len(stats_list)
        avg_roster = sum(s["n_roster"] for s in stats_list) / n_samples
        avg_spans = sum(s["n_spans"] for s in stats_list) / n_samples
        avg_total = sum(s["total_seconds"] for s in stats_list) / n_samples
        rows.append(
            f"<tr><td>{html.escape(src)}</td><td>{n_samples}</td>"
            f"<td>{avg_roster:.1f}</td><td>{avg_spans:.1f}</td>"
            f"<td>{avg_total:.1f}</td></tr>"
        )
    return (
        "<table class='sum global'><thead><tr>"
        "<th>source</th><th>n samples</th><th>avg # roster</th>"
        "<th>avg # spans</th><th>avg total span (s)</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )


def main() -> int:
    tdf = load_tdf()
    print(f"loaded {len(tdf)} TDF rows")

    all_model_preds: dict[str, dict[str, dict[str, Any]]] = {}
    for model_name, _ in MODELS:
        preds = load_model_predictions(model_name)
        all_model_preds[model_name] = preds
        print(f"  {model_name}: {len(preds)} preds")

    per_sample_cards: list[str] = []
    all_stats: dict[str, list[dict[str, Any]]] = {"GT": []}
    for sid in sorted(tdf.keys()):
        row = tdf[sid]
        model_preds = {m: all_model_preds.get(m, {}).get(sid, {}) for m, _ in MODELS}
        per_sample_cards.append(build_sample_card(sid, row, model_preds))
        # Collect global stats
        all_stats["GT"].append(span_stats(parse_gt(row)))
        for m, _ in MODELS:
            payload = model_preds.get(m, {})
            if "_parse_error" not in payload:
                all_stats.setdefault(m, []).append(span_stats(payload))

    # Metric table from A-1741 for reference
    metric_table = """
    <table class='metric'>
    <thead><tr><th>model</th><th>naming_iou</th><th>name_appearance_iou</th><th>delta</th></tr></thead>
    <tbody>
      <tr><td>pegasus-15-sft</td><td>0.123</td><td>0.339</td><td>0.293</td></tr>
      <tr><td>pegasus-15-rl</td><td>0.146</td><td>0.371</td><td>0.286</td></tr>
      <tr><td>pegasus-15</td><td>0.145</td><td>0.349</td><td>0.271</td></tr>
      <tr><td>entity-h0-sme-1300</td><td>0.107</td><td>0.303</td><td>0.265</td></tr>
      <tr><td>entity-h0-sme-2200</td><td>0.097</td><td>0.308</td><td>0.282</td></tr>
      <tr class='hi'><td><b>gemini-2.5-pro</b></td><td><b>0.217</b></td><td><b>0.238</b></td><td><b>0.046</b></td></tr>
    </tbody></table>
    """

    html_out = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>Gemini-2.5-pro vs Pegasus — chunk_05m entity-cov predictions</title>
<style>
body{{font-family:-apple-system,Segoe UI,Roboto,sans-serif;margin:24px;color:#222;max-width:1600px}}
h1{{font-size:22px;margin-bottom:6px}}
h2{{font-size:16px;margin:20px 0 8px}}
h3{{font-size:14px;margin:16px 0 6px}}
h4{{font-size:13px;margin:10px 0 4px}}
table{{border-collapse:collapse;font-size:12px;margin:8px 0}}
th,td{{border:1px solid #ccc;padding:4px 8px;vertical-align:top}}
th{{background:#f3f3f3}}
.sum td:first-child{{font-weight:600}}
.sum tr.gt td{{background:#eef5ff}}
.metric tr.hi td{{background:#fff3d6}}
.mono{{font-family:ui-monospace,Menlo,Monaco,monospace;font-size:11px}}
.note{{color:#666;font-size:11px}}
.timeline{{margin:6px 0 10px;font-size:11px}}
.timeline-axis{{position:relative;margin-left:170px;height:16px;border-bottom:1px solid #9aa4b2;background:linear-gradient(to right,#d0d7de 1px,transparent 1px);background-size:25% 100%}}
.timeline-axis .tick{{position:absolute;top:0;transform:translateX(-50%);font-size:10px;color:#555;white-space:nowrap}}
.timeline-row{{display:grid;grid-template-columns:170px minmax(0,1fr);gap:6px;align-items:center;margin:2px 0}}
.timeline-label{{text-align:right;font-size:11px;color:#444;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}}
.timeline-bar{{position:relative;height:14px;background:#f6f8fa;border:1px solid #d0d7de;border-radius:3px;overflow:hidden}}
.timeline-bar .seg{{position:absolute;top:2px;height:10px;border-radius:2px;min-width:2px;opacity:0.85}}
.sample{{border:1px solid #d0d7de;border-radius:8px;margin:8px 0;background:#fff}}
.sample summary{{cursor:pointer;padding:8px 12px;background:#f6f8fa;border-radius:8px;font-size:13px}}
.sample summary .meta{{color:#666;font-size:11px;margin-left:8px}}
.sample .body{{padding:10px 14px}}
.rosters .roster-block{{margin:8px 0}}
.roster{{width:100%}}
.roster td:first-child{{font-weight:600}}
.legend{{display:flex;gap:16px;margin:6px 0;font-size:12px;align-items:center}}
.legend .swatch{{display:inline-block;width:14px;height:8px;border-radius:2px;margin-right:4px;vertical-align:-1px}}
</style></head><body>
<h1>Gemini-2.5-pro vs Pegasus — chunk_05m entity-cov predictions</h1>
<p class='note'>Dataset: twelvelabs/entity_cov_v0_tdf, chunk_05m (34 rows).
Purpose: figure out why gemini-2.5-pro has the highest naming_iou but the lowest name_appearance_iou compared to Pegasus.</p>

<h2>Metrics (A-1741 chunk_05m)</h2>
{metric_table}

<h2>Aggregate span statistics (mean per sample)</h2>
{build_global_summary(all_stats)}

<h2>Legend</h2>
<div class='legend'>
  <span><span class='swatch' style='background:{GT_COLOR}'></span>GT</span>
  {' '.join(f"<span><span class='swatch' style='background:{c}'></span>{html.escape(m)}</span>" for m, c in MODELS)}
</div>

<h2>Per-sample cards ({len(per_sample_cards)})</h2>
{''.join(per_sample_cards)}
</body></html>
"""

    OUT_HTML.write_text(html_out, encoding="utf-8")
    print(f"wrote {OUT_HTML} ({OUT_HTML.stat().st_size / 1024:.1f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

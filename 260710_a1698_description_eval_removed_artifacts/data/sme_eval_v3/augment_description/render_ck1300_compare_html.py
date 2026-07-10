"""Render ck1300-vs-previous description judge comparison HTML."""

from __future__ import annotations

import argparse
import html
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
OUTPUT_DIR = ROOT / "data/sme_eval_v3/augment_description/output"


@dataclass(frozen=True)
class RunPaths:
    s3: str
    eval_json: Path
    exploded_jsonl: Path
    label: str


@dataclass
class Record:
    config: str
    sample_id: str
    sample_short: str
    key: tuple[str, int]
    gt_idx: int
    pred_idx: int | None
    iou: float
    start_time: float | None
    end_time: float | None
    media_path: str
    reference: str
    prediction: str
    score: float
    method: str
    positive_tags: list[str]
    negative_tags: list[str]
    positive_tag_rationales: dict[str, str]
    negative_tag_rationales: dict[str, str]
    tags: list[str]
    rationale: str
    rubric: dict[str, Any]


RUNS = {
    "H13": {
        "previous": RunPaths(
            s3="s3://tl-data-training-pegasus-us-west-2/eval_results/outputs/lia-h13-others-full-2607031334/sme_eval_v3.1_fast/predictions.jsonl",
            eval_json=Path(
                "/private/tmp/sme_description_eval_full_2607031334/h13_exploded_desc_v2/metadata_evaluation.json"
            ),
            exploded_jsonl=Path(
                "/private/tmp/sme_description_eval_full_2607031334/h13/predictions.description_only.exploded.jsonl"
            ),
            label="lia-h13-others-full-2607031334",
        ),
        "ck1300": RunPaths(
            s3="s3://tl-data-training-pegasus-us-west-2/eval_results/outputs/lia-consol-ck1300-h13-others-full-2607031534/sme_eval_v3.1_fast/predictions.jsonl",
            eval_json=Path(
                "/private/tmp/sme_description_eval_lia_consol_ck1300_2607031534/h13_exploded_desc/metadata_evaluation.json"
            ),
            exploded_jsonl=Path(
                "/private/tmp/sme_description_eval_lia_consol_ck1300_2607031534/h13/predictions.description_only.exploded.jsonl"
            ),
            label="lia-consol-ck1300-h13-others-full-2607031534",
        ),
    },
    "H14": {
        "previous": RunPaths(
            s3="s3://tl-data-training-pegasus-us-west-2/eval_results/outputs/lia-h14-others-full-2607031334/sme_eval_v3.1_fast/predictions.jsonl",
            eval_json=Path(
                "/private/tmp/sme_description_eval_full_2607031334/h14_exploded_desc_v2/metadata_evaluation.json"
            ),
            exploded_jsonl=Path(
                "/private/tmp/sme_description_eval_full_2607031334/h14/predictions.description_only.exploded.jsonl"
            ),
            label="lia-h14-others-full-2607031334",
        ),
        "ck1300": RunPaths(
            s3="s3://tl-data-training-pegasus-us-west-2/eval_results/outputs/lia-consol-ck1300-h14-others-full-2607031534/sme_eval_v3.1_fast/predictions.jsonl",
            eval_json=Path(
                "/private/tmp/sme_description_eval_lia_consol_ck1300_2607031534/h14_exploded_desc/metadata_evaluation.json"
            ),
            exploded_jsonl=Path(
                "/private/tmp/sme_description_eval_lia_consol_ck1300_2607031534/h14/predictions.description_only.exploded.jsonl"
            ),
            label="lia-consol-ck1300-h14-others-full-2607031534",
        ),
    },
}


def e(value: Any) -> str:
    return html.escape("" if value is None else str(value))


def as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def as_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def load_jsonl(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        rows[str(row["sample_id"])] = row
    return rows


def first_description(value: Any) -> str:
    if isinstance(value, list) and value and isinstance(value[0], dict):
        return str(value[0].get("description", ""))
    if isinstance(value, dict):
        return str(value.get("description", ""))
    return ""


def first_time(value: Any, name: str) -> float | None:
    if isinstance(value, list) and value and isinstance(value[0], dict):
        raw = value[0].get(name)
        return as_float(raw) if raw is not None else None
    return None


def pick_rubric(row: dict[str, Any], gt_idx: int) -> dict[str, Any]:
    rubrics = row.get("description_eval", {}).get("rubrics", [])
    if not isinstance(rubrics, list):
        return {}
    for rubric in rubrics:
        if isinstance(rubric, dict) and as_int(rubric.get("chapter_index")) == gt_idx:
            return rubric
    if rubrics and isinstance(rubrics[0], dict):
        return rubrics[0]
    return {}


def load_records(config: str, paths: RunPaths) -> dict[tuple[str, int], Record]:
    rows = load_jsonl(paths.exploded_jsonl)
    eval_data = json.loads(paths.eval_json.read_text())
    records: dict[tuple[str, int], Record] = {}
    for sample_id, sample_eval in eval_data.get("by_sample", {}).items():
        row = rows.get(sample_id)
        if not row:
            continue
        original_sample_id = str(row.get("original_sample_id", sample_id))
        gt_idx = as_int(row.get("original_gt_idx"))
        if gt_idx is None:
            continue
        pred_idx = as_int(row.get("original_pred_idx"))
        pair = (sample_eval.get("matched_pairs") or [{}])[0]
        field_judge = pair.get("field_judges", {}).get("description", {})
        field_scores = pair.get("field_scores", {})
        score = as_float(field_scores.get("description", field_judge.get("score", 0.0)))
        method = pair.get("methods", {}).get("description", "")
        chapters = row.get("chapters")
        record = Record(
            config=config,
            sample_id=original_sample_id,
            sample_short=original_sample_id[:8],
            key=(original_sample_id, gt_idx),
            gt_idx=gt_idx,
            pred_idx=pred_idx,
            iou=as_float(row.get("original_iou", pair.get("iou", 0.0))),
            start_time=first_time(chapters, "start_time"),
            end_time=first_time(chapters, "end_time"),
            media_path=str(row.get("media_path", "")),
            reference=first_description(chapters),
            prediction=first_description(row.get("response")),
            score=score,
            method=str(method),
            positive_tags=[str(tag) for tag in field_judge.get("positive_tags", []) if tag],
            negative_tags=[str(tag) for tag in field_judge.get("negative_tags", []) if tag],
            positive_tag_rationales={
                str(tag): str(reason)
                for tag, reason in field_judge.get("positive_tag_rationales", {}).items()
                if tag and reason
            },
            negative_tag_rationales={
                str(tag): str(reason)
                for tag, reason in field_judge.get("negative_tag_rationales", {}).items()
                if tag and reason
            },
            tags=[str(tag) for tag in field_judge.get("tags", []) if tag],
            rationale=str(field_judge.get("rationale", "")),
            rubric=pick_rubric(row, gt_idx),
        )
        records[record.key] = record
    return records


def mean(records: list[Record]) -> float:
    if not records:
        return 0.0
    return sum(r.score for r in records) / len(records)


def iou_weighted(records: list[Record]) -> float:
    total = sum(r.iou for r in records)
    if not total:
        return 0.0
    return sum(r.score * r.iou for r in records) / total


def score_dist(records: list[Record]) -> Counter[int]:
    return Counter(round(r.score) for r in records)


def tag_counts(records: list[Record], attr: str = "tags") -> Counter[str]:
    counts: Counter[str] = Counter()
    for record in records:
        counts.update(getattr(record, attr))
    return counts


def tag_diff(before: list[str], after: list[str]) -> tuple[list[str], list[str]]:
    before_set = set(before)
    after_set = set(after)
    return sorted(after_set - before_set), sorted(before_set - after_set)


def delta_class(delta: float) -> str:
    if delta > 0:
        return "pos"
    if delta < 0:
        return "neg"
    return "zero"


def score_badge(score: float | None) -> str:
    if score is None:
        return '<span class="score none">-</span>'
    rounded = round(score)
    return f'<span class="score s{rounded}">{rounded}</span>'


def render_metric(label: str, value: str, detail: str = "") -> str:
    return f'<div class="metric"><span>{e(label)}</span><strong>{value}</strong><small>{e(detail)}</small></div>'


def render_score_bars(prev: list[Record], new: list[Record], prev_display: str, new_display: str) -> str:
    prev_counts = score_dist(prev)
    new_counts = score_dist(new)
    max_count = max([1, *prev_counts.values(), *new_counts.values()])
    bars = []
    for score in range(6):
        prev_count = prev_counts.get(score, 0)
        new_count = new_counts.get(score, 0)
        bars.append(
            "<div>"
            f'<small>{prev_count}/{new_count}</small>'
            f'<span class="old" style="height:{max(2, prev_count / max_count * 100):.1f}%"></span>'
            f'<span class="new" style="height:{max(2, new_count / max_count * 100):.1f}%"></span>'
            f"<label>{score}</label>"
            "</div>"
        )
    return (
        '<div class="score-bars">'
        + "".join(bars)
        + f'</div><div class="legend"><span class="old-dot"></span> {e(prev_display)} <span class="new-dot"></span> {e(new_display)}</div>'
    )


def render_count_bars(counts: Counter[int], total: int) -> str:
    if not counts:
        return '<p class="muted">No common high-IoU segments.</p>'
    max_count = max(counts.values())
    rows = []
    for delta in sorted(counts):
        count = counts[delta]
        pct = count / total * 100 if total else 0.0
        cls = delta_class(delta)
        label = f"{delta:+d}"
        rows.append(
            f'<div class="bar-row"><span>{label}</span><div class="bar"><i class="{cls}" style="width:{count / max_count * 100:.1f}%"></i></div><b>{count}</b><em>{pct:.1f}%</em></div>'
        )
    return "".join(rows)


def render_tag_table(title: str, counts: Counter[str]) -> str:
    if not counts:
        body = '<tr><td class="muted">none</td><td></td></tr>'
    else:
        body = "".join(
            f'<tr><td><span class="tag">{e(tag)}</span></td><td>{count}</td></tr>'
            for tag, count in counts.most_common(10)
        )
    return f'<section class="mini"><h4>{e(title)}</h4><table><tbody>{body}</tbody></table></section>'


def render_tag_rationale_list(tags: list[str], rationales: dict[str, str], tag_class: str) -> str:
    if not tags:
        return '<p class="muted">none</p>'
    items = []
    for tag in tags:
        reason = rationales.get(tag)
        reason_html = f'<p>{e(reason)}</p>' if reason else '<p class="muted">No tag-level rationale.</p>'
        items.append(f'<div class="tag-reason"><span class="tag {tag_class}">{e(tag)}</span>{reason_html}</div>')
    return '<div class="tag-reasons">' + "".join(items) + "</div>"


def render_sample_delta_table(pairs: list[tuple[Record, Record]]) -> str:
    grouped: dict[str, list[float]] = defaultdict(list)
    for prev, new in pairs:
        grouped[prev.sample_id].append(new.score - prev.score)
    rows = sorted(
        ((sample_id, len(values), sum(values) / len(values)) for sample_id, values in grouped.items()),
        key=lambda item: item[2],
        reverse=True,
    )
    body = "".join(
        f'<tr><td>{e(sample_id[:8])}</td><td>{count}</td><td class="num {delta_class(delta)}">{delta:+.3f}</td></tr>'
        for sample_id, count, delta in rows
    )
    return f"<table><thead><tr><th>Sample</th><th>Common Segments</th><th>Mean Δ</th></tr></thead><tbody>{body}</tbody></table>"


def render_list(items: Any) -> str:
    if not isinstance(items, list) or not items:
        return '<p class="muted">not provided</p>'
    return '<ul class="rubric-list">' + "".join(f"<li>{e(item)}</li>" for item in items) + "</ul>"


def render_anchor_panel(rubric: dict[str, Any]) -> str:
    if not rubric:
        return (
            '<section class="anchor-details"><h4>Rubric + Score Anchors</h4><p class="muted">not provided</p></section>'
        )
    anchors = rubric.get("score_anchors")
    if not isinstance(anchors, dict):
        anchors = {str(score): rubric.get(f"score_{score}_description", "not provided") for score in range(1, 6)}
    anchor_cards = []
    for score in range(1, 6):
        text = anchors.get(str(score), anchors.get(score, "not provided"))
        anchor_cards.append(f'<div class="anchor"><b>{score}</b><p>{e(text)}</p></div>')
    return (
        '<section class="anchor-details">'
        "<h4>Rubric + Score Anchors</h4>"
        '<div class="rubric-meta">'
        f'<section><h5>Criteria</h5><p>{e(rubric.get("criteria", "not provided"))}</p></section>'
        f'<section><h5>Must Include</h5>{render_list(rubric.get("must_include"))}</section>'
        f'<section><h5>Common Failure Modes</h5>{render_list(rubric.get("common_failure_modes"))}</section>'
        "</div>"
        '<div class="anchor-grid">' + "".join(anchor_cards) + "</div></section>"
    )


def render_prediction_panel(title: str, record: Record | None) -> str:
    if record is None:
        return f'<section><h4>{e(title)}</h4><p class="muted">No matched segment in this run.</p></section>'
    return (
        f"<section><h4>{e(title)} Prediction</h4><p>{e(record.prediction)}</p>"
        f"<h4>{e(title)} Rationale</h4><p>{e(record.rationale or record.method)}</p>"
        f'<h4>{e(title)} Positive Tags</h4>'
        f"{render_tag_rationale_list(record.positive_tags, record.positive_tag_rationales, 'positive')}"
        f'<h4>{e(title)} Negative Tags</h4>'
        f"{render_tag_rationale_list(record.negative_tags, record.negative_tag_rationales, 'negative')}</section>"
    )


def render_card(status: str, prev: Record | None, new: Record | None, prev_display: str, new_display: str) -> str:
    base = new or prev
    assert base is not None
    delta = None if prev is None or new is None else new.score - prev.score
    gained_pos: list[str] = []
    lost_pos: list[str] = []
    gained_neg: list[str] = []
    lost_neg: list[str] = []
    if prev and new:
        gained_pos, lost_pos = tag_diff(prev.positive_tags, new.positive_tags)
        gained_neg, lost_neg = tag_diff(prev.negative_tags, new.negative_tags)
    search_terms = " ".join(
        [
            base.sample_id,
            str(base.gt_idx),
            *(prev.positive_tags if prev else []),
            *(prev.negative_tags if prev else []),
            *(new.positive_tags if new else []),
            *(new.negative_tags if new else []),
        ]
    )
    scoreline = f"{score_badge(prev.score if prev else None)}<span class=\"arrow\">→</span>{score_badge(new.score if new else None)}"
    if delta is not None:
        scoreline += f'<span class="delta {delta_class(delta)}">{delta:+.0f}</span>'
    else:
        scoreline += (
            f'<span class="delta zero">{e(new_display)} only</span>'
            if new
            else f'<span class="delta zero">{e(prev_display)} only</span>'
        )
    time_bits = []
    if base.start_time is not None and base.end_time is not None:
        time_bits.append(f"{base.start_time:.1f}s - {base.end_time:.1f}s")
    if prev and new:
        time_bits.append(f"IoU {prev_display} {prev.iou:.3f} / {new_display} {new.iou:.3f}")
    else:
        time_bits.append(f"IoU {base.iou:.3f}")
    gained_pos_html = (
        "".join(f'<span class="tag gain positive">{e(tag)}</span>' for tag in gained_pos)
        or '<span class="muted">none</span>'
    )
    lost_pos_html = (
        "".join(f'<span class="tag loss positive">{e(tag)}</span>' for tag in lost_pos)
        or '<span class="muted">none</span>'
    )
    gained_neg_html = (
        "".join(f'<span class="tag loss negative">{e(tag)}</span>' for tag in gained_neg)
        or '<span class="muted">none</span>'
    )
    lost_neg_html = (
        "".join(f'<span class="tag gain negative">{e(tag)}</span>' for tag in lost_neg)
        or '<span class="muted">none</span>'
    )
    return (
        f'<article class="segment-card {status}" data-status="{status}" data-search="{e(search_terms)}">'
        "<header><div>"
        f"<strong>{e(base.sample_short)}</strong><span>GT {base.gt_idx} · {' · '.join(e(bit) for bit in time_bits)}</span>"
        f'</div><div class="scoreline">{scoreline}</div></header>'
        f'<p class="s3"><code>{e(base.media_path)}</code></p>'
        '<div class="tag-diff four-col">'
        f'<div><b>Gained Positive</b>{gained_pos_html}</div>'
        f'<div><b>Lost Positive</b>{lost_pos_html}</div>'
        f'<div><b>Gained Negative</b>{gained_neg_html}</div>'
        f'<div><b>Lost Negative</b>{lost_neg_html}</div>'
        '</div>'
        '<div class="cols">'
        f"{render_prediction_panel(prev_display, prev)}"
        f"{render_prediction_panel(new_display, new)}"
        "</div>"
        f'<details><summary>Reference Description</summary><p>{e(base.reference)}</p></details>'
        f"{render_anchor_panel(base.rubric)}"
        "</article>"
    )


def status_for(prev: Record | None, new: Record | None) -> str:
    if prev is None:
        return "added"
    if new is None:
        return "removed"
    if new.score > prev.score:
        return "improved"
    if new.score < prev.score:
        return "worsened"
    return "same"


def render_cards(
    title: str, cards: list[tuple[str, Record | None, Record | None]], prev_display: str, new_display: str
) -> str:
    if not cards:
        return f'<h3>{e(title)}</h3><div class="cards"><p class="muted">No segments.</p></div>'
    body = "".join(render_card(status, prev, new, prev_display, new_display) for status, prev, new in cards)
    return f'<h3>{e(title)}</h3><div class="cards">{body}</div>'


def render_config(config: str, threshold: float, prev_display: str, new_display: str) -> str:
    paths = RUNS[config]
    prev_all = load_records(config, paths["previous"])
    new_all = load_records(config, paths["ck1300"])
    prev_high = {k: v for k, v in prev_all.items() if v.iou > threshold}
    new_high = {k: v for k, v in new_all.items() if v.iou > threshold}
    common_keys = sorted(set(prev_high) & set(new_high))
    common_pairs = [(prev_high[k], new_high[k]) for k in common_keys]
    movement = Counter(round(new.score - prev.score) for prev, new in common_pairs)
    improved_pairs = [(prev, new) for prev, new in common_pairs if new.score > prev.score]
    worsened_pairs = [(prev, new) for prev, new in common_pairs if new.score < prev.score]
    added_keys = sorted(set(new_high) - set(prev_high))
    removed_keys = sorted(set(prev_high) - set(new_high))
    added = [new_high[k] for k in added_keys]
    removed = [prev_high[k] for k in removed_keys]
    delta = mean(list(new_high.values())) - mean(list(prev_high.values()))
    common_delta = mean([new for _, new in common_pairs]) - mean([prev for prev, _ in common_pairs])
    improved_new = [new for _, new in improved_pairs]
    worsened_new = [new for _, new in worsened_pairs]
    improved_gained_positive: Counter[str] = Counter()
    improved_lost_negative: Counter[str] = Counter()
    worsened_gained_negative: Counter[str] = Counter()
    worsened_lost_positive: Counter[str] = Counter()
    for prev, new in improved_pairs:
        gained_positive, _ = tag_diff(prev.positive_tags, new.positive_tags)
        _, lost_negative = tag_diff(prev.negative_tags, new.negative_tags)
        improved_gained_positive.update(gained_positive)
        improved_lost_negative.update(lost_negative)
    for prev, new in worsened_pairs:
        gained_negative, _ = tag_diff(prev.negative_tags, new.negative_tags)
        _, lost_positive = tag_diff(prev.positive_tags, new.positive_tags)
        worsened_gained_negative.update(gained_negative)
        worsened_lost_positive.update(lost_positive)

    detail_cards: list[tuple[str, Record | None, Record | None]] = []
    detail_cards.extend(
        ("improved", prev, new)
        for prev, new in sorted(improved_pairs, key=lambda p: p[1].score - p[0].score, reverse=True)
    )
    detail_cards.extend(
        ("worsened", prev, new) for prev, new in sorted(worsened_pairs, key=lambda p: p[1].score - p[0].score)
    )
    detail_cards.extend(("added", None, rec) for rec in sorted(added, key=lambda r: (r.score, r.sample_id, r.gt_idx)))
    detail_cards.extend(
        ("removed", rec, None) for rec in sorted(removed, key=lambda r: (r.score, r.sample_id, r.gt_idx))
    )

    return f"""
    <section id="{config}" class="config-block">
      <div class="section-title"><h2>{config}</h2><p>{e(paths['previous'].label)} → {e(paths['ck1300'].label)}</p></div>
      <div class="s3-grid"><div><b>{e(prev_display)} S3</b><code>{e(paths['previous'].s3)}</code></div><div><b>{e(new_display)} S3</b><code>{e(paths['ck1300'].s3)}</code></div></div>
      <div class="metrics">
        {render_metric(f'{prev_display} Score', f'{mean(list(prev_high.values())):.3f}', f'n={len(prev_high)} / all={len(prev_all)}, IoU-w={iou_weighted(list(prev_high.values())):.3f}')}
        {render_metric(f'{new_display} Score', f'{mean(list(new_high.values())):.3f}', f'n={len(new_high)} / all={len(new_all)}, IoU-w={iou_weighted(list(new_high.values())):.3f}')}
        {render_metric('High-IoU Δ', f'<span class="delta {delta_class(delta)}">{delta:+.3f}</span>', f'{new_display} - {prev_display} over independently filtered runs')}
        {render_metric('Common High-IoU Δ', f'<span class="delta {delta_class(common_delta)}">{common_delta:+.3f}</span>', f'common={len(common_pairs)}, improved={len(improved_pairs)}, worsened={len(worsened_pairs)}')}
      </div>
      <div class="panel two">
        <section><h4>Score Distribution</h4>{render_score_bars(list(prev_high.values()), list(new_high.values()), prev_display, new_display)}</section>
        <section><h4>Common Segment Movement</h4>{render_count_bars(movement, len(common_pairs))}</section>
      </div>
      <div class="panel tag-panels">
        {render_tag_table(f'Improved: {new_display} Positive Tags', tag_counts(improved_new, 'positive_tags'))}
        {render_tag_table('Improved: Lost Negative Tags', improved_lost_negative)}
        {render_tag_table('Improved: Gained Positive Tags', improved_gained_positive)}
        {render_tag_table(f'Worsened: {new_display} Negative Tags', tag_counts(worsened_new, 'negative_tags'))}
        {render_tag_table('Worsened: Gained Negative Tags', worsened_gained_negative)}
        {render_tag_table('Worsened: Lost Positive Tags', worsened_lost_positive)}
      </div>
      <div class="panel"><h4>Sample-Level Mean Delta, Common High-IoU Segments</h4>{render_sample_delta_table(common_pairs)}</div>
      <div class="controls"><button data-filter="all">All</button><button data-filter="improved">Improved</button><button data-filter="worsened">Worsened</button><button data-filter="added">Added in {e(new_display)}</button><button data-filter="removed">Removed</button><input placeholder="search sample/tag" /></div>
      {render_cards('High-IoU Segment Details', detail_cards, prev_display, new_display)}
    </section>
    """


STYLE = """
:root { --bg:#f6f7f9; --ink:#1f2933; --muted:#687385; --line:#d9dee7; --panel:#fff; --old:#94a3b8; --new:#2563eb; --pos:#0f8a5f; --neg:#c43e3e; --warn:#a16207; }
* { box-sizing:border-box; }
body { margin:0; background:var(--bg); color:var(--ink); font:14px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; }
header.hero { padding:28px 32px 20px; background:#101820; color:white; }
header.hero h1 { margin:0 0 8px; font-size:28px; letter-spacing:0; }
header.hero p { margin:0; color:#cbd5e1; max-width:1120px; }
nav { display:flex; gap:10px; padding:12px 32px; background:#fff; border-bottom:1px solid var(--line); position:sticky; top:0; z-index:10; }
nav a { color:#143d73; text-decoration:none; font-weight:700; padding:6px 10px; border:1px solid var(--line); border-radius:6px; }
main { padding:24px 32px 48px; max-width:1680px; margin:0 auto; }
.section-title { display:flex; justify-content:space-between; gap:24px; align-items:end; margin:16px 0; }
.section-title h2 { margin:0; font-size:24px; } .section-title p { margin:0; color:var(--muted); }
.s3-grid { display:grid; grid-template-columns:1fr 1fr; gap:12px; margin-bottom:14px; }
.s3-grid div, .metric, .panel, .segment-card { background:var(--panel); border:1px solid var(--line); border-radius:8px; }
.s3-grid div { padding:12px; min-width:0; } code { white-space:normal; overflow-wrap:anywhere; font-family:ui-monospace,SFMono-Regular,Menlo,monospace; font-size:12px; }
.metrics { display:grid; grid-template-columns:repeat(4, minmax(0,1fr)); gap:12px; margin:12px 0; }
.metric { padding:14px; } .metric span { display:block; color:var(--muted); font-size:12px; } .metric strong { display:block; font-size:22px; margin:4px 0; } .metric small { color:var(--muted); }
.delta { display:inline-block; padding:2px 8px; border-radius:999px; font-weight:700; background:#e5e7eb; margin-left:6px; } .delta.pos { color:var(--pos); background:#dcfce7; } .delta.neg { color:var(--neg); background:#fee2e2; } .delta.zero { color:#57606f; }
.panel { padding:16px; margin:12px 0; } .panel.two { display:grid; grid-template-columns:1fr 1fr; gap:18px; } .panel.four { display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:14px; } .panel.tag-panels { display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:14px; }
h3 { margin:22px 0 10px; font-size:18px; } h4 { margin:0 0 8px; font-size:13px; color:#344054; }
.score-bars { height:220px; display:flex; gap:18px; align-items:end; border-left:1px solid var(--line); border-bottom:1px solid var(--line); padding:10px 8px 24px; }
.score-bars div { flex:1; height:100%; display:flex; align-items:end; justify-content:center; gap:4px; position:relative; } .score-bars label { position:absolute; bottom:-22px; } .score-bars small { position:absolute; top:-12px; color:var(--muted); font-size:11px; } .score-bars span { width:18px; min-height:2px; border-radius:4px 4px 0 0; } .score-bars .old { background:var(--old); } .score-bars .new { background:var(--new); }
.legend { color:var(--muted); margin-top:26px; } .old-dot,.new-dot { display:inline-block; width:10px; height:10px; border-radius:50%; margin-left:8px; } .old-dot { background:var(--old); } .new-dot { background:var(--new); }
.bar-row { display:grid; grid-template-columns:54px 1fr 38px 54px; gap:8px; align-items:center; margin:8px 0; } .bar { height:9px; background:#edf0f5; border-radius:99px; overflow:hidden; } .bar i { display:block; height:100%; background:#94a3b8; } .bar i.pos { background:var(--pos); } .bar i.neg { background:var(--neg); } .bar i.zero { background:#64748b; } .bar-row em { color:var(--muted); font-style:normal; font-size:12px; }
table { width:100%; border-collapse:collapse; } th,td { border-bottom:1px solid var(--line); padding:8px; vertical-align:top; text-align:left; } th { font-size:12px; color:var(--muted); } .num.pos { color:var(--pos); font-weight:700; } .num.neg { color:var(--neg); font-weight:700; }
.controls { display:flex; gap:8px; align-items:center; margin:22px 0 14px; } button { border:1px solid var(--line); background:#fff; border-radius:6px; padding:7px 10px; cursor:pointer; } button.active { background:#143d73; color:white; } input { margin-left:auto; min-width:260px; padding:8px 10px; border:1px solid var(--line); border-radius:6px; }
.cards { display:grid; gap:12px; } .segment-card { padding:14px; } .segment-card.improved { border-left:5px solid var(--pos); } .segment-card.worsened { border-left:5px solid var(--neg); } .segment-card.added { border-left:5px solid var(--new); } .segment-card.removed { border-left:5px solid var(--old); }
.segment-card header { display:flex; justify-content:space-between; gap:16px; align-items:center; } .segment-card header span { display:block; color:var(--muted); font-size:12px; } .scoreline { display:flex; gap:8px; align-items:center; flex-wrap:wrap; } .arrow { color:var(--muted); }
.score { display:inline-flex; align-items:center; justify-content:center; flex:0 0 32px; width:32px; min-width:32px; height:32px; line-height:32px; border-radius:50%; color:#fff !important; font-weight:800; font-size:15px; text-align:center; font-variant-numeric:tabular-nums; overflow:hidden; } .score.s0 { background:#111827; } .score.s1 { background:#b91c1c; } .score.s2 { background:#c76c12; } .score.s3 { background:#2563eb; } .score.s4 { background:#0f8a5f; } .score.s5 { background:#064e3b; } .score.none { background:#9ca3af; }
.s3 { margin:8px 0 10px; color:var(--muted); } .tag-diff { display:grid; grid-template-columns:1fr 1fr; gap:12px; margin:10px 0; } .tag-diff.four-col { grid-template-columns:repeat(4,minmax(0,1fr)); } .tag-diff b { display:block; margin:0 0 4px; }
.tag { display:inline-block; margin:2px 4px 2px 0; padding:2px 7px; border-radius:999px; background:#eef2ff; color:#1e3a8a; font-size:12px; font-weight:700; } .tag.positive { background:#dcfce7; color:#166534; } .tag.negative { background:#fee2e2; color:#991b1b; } .tag.gain { box-shadow:inset 0 0 0 1px #16a34a; } .tag.loss { box-shadow:inset 0 0 0 1px #dc2626; }
.cols { display:grid; grid-template-columns:1fr 1fr; gap:12px; margin-top:10px; } .cols section { background:#f8fafc; border:1px solid var(--line); border-radius:6px; padding:12px; } .cols p, details p { white-space:pre-wrap; margin:0 0 10px; } details { margin-top:10px; } summary { cursor:pointer; font-weight:700; } .muted { color:var(--muted); }
.tag-reasons { display:grid; gap:8px; margin:0 0 10px; } .tag-reason { padding:8px; background:#fff; border:1px solid var(--line); border-radius:6px; } .tag-reason p { margin:4px 0 0; color:#344054; }
.anchor-details { margin-top:10px; padding:12px; background:#fffaf0; border:1px solid #e6d8ad; border-radius:6px; }
.anchor-details h4 { color:#65440d; margin:0 0 10px; }
.anchor-details h5 { margin:0 0 6px; color:#475467; font-size:12px; text-transform:uppercase; letter-spacing:0; }
.rubric-meta { display:grid; grid-template-columns:1.2fr 1fr 1fr; gap:10px; margin-bottom:10px; }
.rubric-meta section, .anchor { background:#fff; border:1px solid var(--line); border-radius:6px; padding:10px; }
.rubric-meta p, .anchor p { margin:0; white-space:pre-wrap; }
.rubric-list { margin:0; padding-left:18px; }
.anchor-grid { display:grid; grid-template-columns:repeat(5,minmax(0,1fr)); gap:8px; }
.anchor b { display:inline-flex; align-items:center; justify-content:center; width:24px; height:24px; border-radius:50%; background:#143d73; color:#fff; font-weight:800; }
.anchor p { margin-top:8px; }
@media (max-width: 1100px) { .rubric-meta, .anchor-grid { grid-template-columns:1fr; } }
@media (max-width: 900px) { .metrics,.panel.two,.panel.four,.panel.tag-panels,.s3-grid,.cols,.tag-diff,.tag-diff.four-col { grid-template-columns:1fr; } nav { position:static; } input { min-width:0; width:100%; } .controls { flex-wrap:wrap; } }
"""


SCRIPT = """
document.querySelectorAll('.config-block').forEach(block => {
  const buttons = block.querySelectorAll('button[data-filter]');
  const input = block.querySelector('input');
  const cards = block.querySelectorAll('.segment-card');
  let filter = 'all';
  function apply() {
    const q = (input?.value || '').toLowerCase();
    cards.forEach(card => {
      const statusOk = filter === 'all' || card.dataset.status === filter;
      const searchOk = !q || (card.dataset.search || '').toLowerCase().includes(q);
      card.style.display = statusOk && searchOk ? '' : 'none';
    });
    buttons.forEach(btn => btn.classList.toggle('active', btn.dataset.filter === filter));
  }
  buttons.forEach(btn => btn.addEventListener('click', () => { filter = btn.dataset.filter; apply(); }));
  input?.addEventListener('input', apply);
  apply();
});
"""


def configure_eval_base(eval_base_dir: Path) -> None:
    mapping = {
        ("H13", "previous"): eval_base_dir / "prev_h13/metadata_evaluation.json",
        ("H14", "previous"): eval_base_dir / "prev_h14/metadata_evaluation.json",
        ("H13", "ck1300"): eval_base_dir / "ck1300_h13/metadata_evaluation.json",
        ("H14", "ck1300"): eval_base_dir / "ck1300_h14/metadata_evaluation.json",
    }
    for (config, run_name), eval_json in mapping.items():
        current = RUNS[config][run_name]
        RUNS[config][run_name] = RunPaths(
            s3=current.s3,
            eval_json=eval_json,
            exploded_jsonl=current.exploded_jsonl,
            label=current.label,
        )


def render_page(threshold: float, model_label: str, prev_display: str, new_display: str) -> str:
    body = "\n".join(render_config(config, threshold, prev_display, new_display) for config in ["H13", "H14"])
    page_title = f"{new_display} vs {prev_display} Description Judge Comparison"
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>{e(page_title)}, IoU &gt; {threshold:.2f}</title>
<style>{STYLE}</style>
</head>
<body>
<header class="hero"><h1>{e(page_title)}</h1><p>Judge: <code>{e(model_label)}</code>. Filtered to high temporal-alignment records only: <code>original_iou &gt; {threshold:.2f}</code>. Common segment deltas require both {e(prev_display)} and {e(new_display)} records to pass the threshold; added/removed segments are filtered by their own run's IoU. Positive and negative rationale tags are shown separately.</p></header>
<nav><a href="#H13">H13</a><a href="#H14">H14</a></nav>
<main>{body}</main>
<script>{SCRIPT}</script>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iou-threshold", type=float, default=0.99)
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_DIR / "260703_lia_consol_ck1300_vs_prev_description_judge_compare_iou_gt099.html",
    )
    parser.add_argument("--eval-base-dir", type=Path, default=None)
    parser.add_argument("--model-label", default="gemini-3.1-pro-preview")
    parser.add_argument("--previous-display-label", default="previous")
    parser.add_argument("--new-display-label", default="ck1300")
    args = parser.parse_args()
    if args.eval_base_dir is not None:
        configure_eval_base(args.eval_base_dir)
    html_text = render_page(args.iou_threshold, args.model_label, args.previous_display_label, args.new_display_label)
    args.output.write_text(html_text)
    print(args.output)


if __name__ == "__main__":
    main()

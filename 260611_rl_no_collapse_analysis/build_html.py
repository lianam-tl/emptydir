"""[cc-generated] Reproduce 260508_rl_eval_research_questions rollout analyses for the
hgmkw8sg RL run where response length did NOT collapse.

Reads jsonl files under ROLLOUT_DIR/{step}.jsonl. Generates 4 standalone HTMLs +
1 wrapper:
  - 260611_hgmkw8sg_early_rollouts.html        (steps 1-7, per-step cards)
  - 260611_hgmkw8sg_step400_rollouts.html      (step 400, 20 sample cards)
  - 260611_hgmkw8sg_significant_think.html     (steps 1-7, >=100 think words)
  - 260611_hgmkw8sg_cognitive_patterns.html    (per-metric tables + monotonic trends)
  - 260611_hgmkw8sg_rl_no_collapse.html        (wrapper with tabs)

Pattern detection mirrors the names from 260503_think_pattern_v2.html using
regex heuristics on the pre-</think> text.
"""

from __future__ import annotations

import html
import json
import random
import re
from collections import defaultdict
from pathlib import Path

ROLLOUT_DIR = Path("/tmp/rollout_logs_hgmkw8sg")
OUT_DIR = Path("/Users/long8v/emptydir/260611_rl_no_collapse_analysis/out")
OUT_DIR.mkdir(parents=True, exist_ok=True)

WANDB_URL = "https://wandb.ai/twelvelabs/pegasus-rl/runs/hgmkw8sg"
CKPT_NAME = "rl_rl_consol_0516_mtp_alpha1_bs_224_8k_lr5e_7_no_detach_encoder_mtp_loss_scale_0p5_think"

EARLY_STEPS = list(range(1, 8))
INSPECT_STEP = 400
TREND_STEPS = sorted(set(range(10, 411, 20)) | set(EARLY_STEPS))  # 1..7 + every-20

SAMPLES_PER_BUCKET_HAS = 5
SAMPLES_PER_BUCKET_NO = 15
SIG_THINK_WORDS = 100
SIG_SAMPLES_PER_STEP = 5

random.seed(0)


# ---------------------------- pattern detection ----------------------------

PATTERN_REGEX = {
    # numbered list "1. text" / "1) text"
    "numbered_list": re.compile(r"(?m)^\s*\d+[.)]\s+\S"),
    # bulleted list "- text" / "* text" / "• text"
    "bulleted_list": re.compile(r"(?m)^\s*[-*•]\s+\S"),
    # enumeration of segments: "Segment 1", "scene 1", "clip 1", "first segment"
    "enumerates_segments": re.compile(
        r"\b(?:segment|scene|clip|chapter|section)\s*#?\s*\d+\b|"
        r"\bfirst\s+(?:segment|scene|clip|chapter|section)\b",
        re.IGNORECASE,
    ),
    # timestamps mm:ss, hh:mm:ss, or "12.3 seconds"
    "timestamp_mention": re.compile(
        r"\b\d{1,2}:\d{2}(?::\d{2})?\b|\b\d+(?:\.\d+)?\s*(?:s|sec|seconds?)\b",
        re.IGNORECASE,
    ),
    # quoted ASR / spoken content
    "asr_quote": re.compile(r"(?:[\"“”][^\"“”\n]{3,}[\"“”])|"
                              r"\b(?:says?|said|says,|saying|asks?|asked|tells?|told|"
                              r"shouts?|yells?|narrates?|announces?|comments?|reads?)\b",
                              re.IGNORECASE),
    # words drawn directly from json schema/keys
    "json_schema_words": re.compile(
        r"\b(?:results|start_time|end_time|player_name|emotion|play_action|play_result|"
        r"summary|down|game_clock|utility_tags|json|schema|field|key|value|results?\":?)\b",
        re.IGNORECASE,
    ),
    # "I see / I observe / video shows / clip shows / watching"
    "video_watch_lang": re.compile(
        r"\bI\s+(?:see|observe|notice|watch|spot|recognize|identify)\b|"
        r"\b(?:the\s+)?(?:video|clip|footage|scene|frames?|broadcast)\s+(?:shows?|depicts?|displays?|features?|appears?|seems?|begins?|cuts?)\b|"
        r"\bvisually\b|\bin the (?:video|clip|footage)\b",
        re.IGNORECASE,
    ),
    # uncertainty markers
    "uncertainty": re.compile(
        r"\b(?:maybe|perhaps|might|likely|possibly|probably|unclear|unsure|"
        r"not\s+sure|hmm+|i\s+think|i\s+believe|seems\s+(?:like|to)|appears\s+to|"
        r"can'?t\s+tell|hard\s+to\s+tell|difficult\s+to|confusing|i'?m\s+confused|"
        r"i'?m\s+not\s+(?:sure|certain))\b",
        re.IGNORECASE,
    ),
    # self-correction
    "self_correct": re.compile(
        r"\b(?:wait|actually|let me\s+re|reconsider|on\s+second\s+thought|"
        r"i\s+was\s+wrong|correction|scratch\s+that|never\s+mind|"
        r"let'?s\s+(?:re|redo)|re-?examine|re-?check|re-?look|hold\s+on)\b",
        re.IGNORECASE,
    ),
    # planning language
    "plan_words": re.compile(
        r"\b(?:plan|approach|strategy|step\s*1|step\s*one|"
        r"first,?\s+|then,?\s+|next,?\s+|finally,?\s+|"
        r"i'?ll\s+|i\s+will\s+|i\s+need\s+to|i\s+should|my\s+task)\b",
        re.IGNORECASE,
    ),
    # verification language
    "verify_words": re.compile(
        r"\b(?:verify|verifies|verified|verifying|"
        r"check|checks|checked|checking|"
        r"confirm|confirms|confirmed|confirming|"
        r"validate|validated|"
        r"double[-\s]?check|re-?check)\b",
        re.IGNORECASE,
    ),
    # mentioning word counts (lexical 'count words' / numeric '<N> words')
    "gt_word_count_check": re.compile(
        r"\b\d{1,4}\s+words?\b|\bword\s+count\b|\bcount(?:ing)?\s+(?:the\s+)?words?\b",
        re.IGNORECASE,
    ),
    # summary line at the end
    "summary_at_end": re.compile(
        r"\b(?:in\s+summary|to\s+summarize|in\s+conclusion|final\s+answer|"
        r"summary:|conclusion:|so,?\s+the\s+answer\s+is|therefore,?\s+)",
        re.IGNORECASE,
    ),
}

PATTERN_ORDER = list(PATTERN_REGEX.keys())


def pre_think(text: str) -> str:
    """Return text before </think>; if no closer, return full text."""
    idx = text.find("</think>")
    return text[:idx] if idx >= 0 else text


def detect_patterns(think_text: str) -> dict[str, bool]:
    return {name: bool(rx.search(think_text)) for name, rx in PATTERN_REGEX.items()}


# ---------------------------- IO ----------------------------

def load_step(step: int) -> list[dict]:
    path = ROLLOUT_DIR / f"{step}.jsonl"
    if not path.exists():
        return []
    rows = []
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


# ---------------------------- shared HTML ----------------------------

ROLLOUT_CSS = """
  body { font-family: -apple-system, BlinkMacSystemFont, 'SF Pro Text', sans-serif; max-width: 1200px; margin: 24px auto; padding: 0 20px; color: #222; background: #fafafa; }
  h1 { font-size: 22px; margin-bottom: 4px; }
  .sub { color: #666; margin-bottom: 24px; font-size: 13px; }
  .summary { background: #fff; border: 1px solid #ddd; border-radius: 8px; padding: 16px; margin-bottom: 24px; }
  .summary h2 { margin: 0 0 12px 0; font-size: 15px; }
  .summary-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 8px; }
  .summary-grid div { background: #f4f4f4; padding: 8px; border-radius: 4px; font-size: 13px; }
  .summary-grid .label { color: #666; font-size: 11px; text-transform: uppercase; }
  .summary-grid .val { font-weight: 600; font-size: 16px; }
  .card { background: #fff; border: 1px solid #ddd; border-radius: 8px; padding: 14px 18px; margin-bottom: 16px; }
  .hdr { display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px; }
  .title { font-weight: 600; font-size: 14px; }
  .meta { font-size: 12px; color: #666; margin-bottom: 8px; }
  .meta span { margin-right: 16px; }
  .metrics { display: grid; grid-template-columns: repeat(6, 1fr); gap: 4px; margin: 8px 0 12px; font-size: 11px; }
  .m { background: #f4f4f4; padding: 4px 6px; border-radius: 3px; display: flex; flex-direction: column; }
  .m .k { color: #888; font-size: 10px; }
  .m .v { font-weight: 600; }
  pre { white-space: pre-wrap; word-wrap: break-word; padding: 10px; border-radius: 4px; font-size: 12px; max-height: 500px; overflow: auto; font-family: ui-monospace, 'SF Mono', Menlo, monospace; }
  pre.output { background: #fffbe6; border: 1px solid #eee; }
  pre.gt { background: #e8f5e9; border: 1px solid #c5e1c8; }
  pre.input { background: #f0f4f8; border: 1px solid #eee; }
  pre.think { background: #f4ebff; border: 1px solid #d9c8f5; }
  pre.ans { background: #fffbe6; border: 1px solid #f5e8a0; }
  .badge { display: inline-block; padding: 2px 8px; border-radius: 10px; font-size: 11px; font-weight: 600; }
  .badge.yes { background: #e0f7e9; color: #1b6f3a; }
  .badge.no { background: #fde8e8; color: #9b1c1c; }
  .badge.t { background:#d9c8f5;color:#5e2d99 }
  .think-tag { background: #ffd3a5; padding: 1px 4px; border-radius: 2px; font-weight: 600; }
  details summary { cursor: pointer; font-weight: 600; font-size: 12px; color: #444; padding: 4px 0; }
  code { background: #eee; padding: 1px 4px; border-radius: 3px; font-size: 11px; }
  .step-anchor { background: #1a4d8c; color: #fff; padding: 6px 12px; margin: 24px 0 12px; border-radius: 4px; font-weight: 600; font-size: 14px; }
"""


def fmt_metric(row: dict, key: str) -> str:
    val = row.get(key)
    if val is None:
        return "—"
    if isinstance(val, float):
        return f"{val:.4f}"
    return str(val)


METRIC_KEYS = [
    "unified_score",
    "score",
    "f1_temporal",
    "f1_segment",
    "format_score",
    "json_parse_ok",
    "schema_valid",
    "instruction_ok",
    "stop_reason_is_length",
    "num_pred_segments",
    "num_gt_segments",
]


def actual_think_words(output: str) -> int:
    """Count words before </think>. `think_word_len` field in logs is broken (always 0)."""
    if "</think>" not in output:
        return 0
    return len(output[:output.find("</think>")].split())


def render_sample_card(row: dict, idx: int | None = None) -> str:
    sample_id = row.get("sample_id", "?")
    short_id = sample_id.replace("-", "")[:32]
    data_source = row.get("data_source", "?")
    step = row.get("step", "?")
    output = row.get("output") or ""
    gts = row.get("gts") or ""
    input_text = row.get("input") or ""
    has_close = "</think>" in output
    has_open = "<think>" in output

    metric_html = "".join(
        f"<div class='m'><span class='k'>{html.escape(k)}</span>"
        f"<span class='v'>{fmt_metric(row, k)}</span></div>"
        for k in METRIC_KEYS
    )

    badge = (
        "<span class='badge yes'>has &lt;/think&gt;</span>"
        if has_close
        else "<span class='badge no'>no &lt;/think&gt;</span>"
    )
    if has_open and not has_close:
        badge += " <span class='badge t'>has &lt;think&gt; only</span>"

    title = (
        f"#{idx} · sample_id <code>{html.escape(short_id)}</code>"
        if idx is not None
        else f"sample <code>{html.escape(short_id)}</code>"
    )

    safe_output = html.escape(output).replace("&lt;/think&gt;",
                                              "<span class='think-tag'>&lt;/think&gt;</span>")
    safe_output = safe_output.replace("&lt;think&gt;",
                                      "<span class='think-tag'>&lt;think&gt;</span>")
    return f"""
<div class='card'>
  <div class='hdr'>
    <div class='title'>{title}</div>
    <div>{badge}</div>
  </div>
  <div class='meta'>
    <span>data_source: <code>{html.escape(str(data_source))}</code></span>
    <span>step: {step}</span>
  </div>
  <div class='metrics'>{metric_html}</div>
  <details open><summary>output (model response)</summary>
    <pre class='output'>{safe_output}</pre>
  </details>
  <details><summary>ground truth</summary>
    <pre class='gt'>{html.escape(gts)}</pre>
  </details>
  <details><summary>input prompt ({len(input_text)} chars)</summary>
    <pre class='input'>{html.escape(input_text[:4000])}</pre>
  </details>
</div>"""


# ---------------------------- 1) step-N rollouts (20 cards) ----------------------------

def build_step_rollouts(step: int, total_samples_expected: int | None = None) -> tuple[str, dict]:
    rows = load_step(step)
    n = len(rows)
    has_close = [r for r in rows if "</think>" in (r.get("output") or "")]
    has_open_only = [r for r in rows
                     if "<think>" in (r.get("output") or "")
                     and "</think>" not in (r.get("output") or "")]
    avg = lambda k: sum(float(r.get(k, 0) or 0) for r in rows) / max(n, 1)
    avg_think_w = sum(actual_think_words(r.get("output") or "") for r in rows) / max(n, 1)

    pick_has = random.sample(has_close, min(SAMPLES_PER_BUCKET_HAS, len(has_close)))
    pool_no = [r for r in rows if r not in has_close]
    pick_no = random.sample(pool_no, min(SAMPLES_PER_BUCKET_NO, len(pool_no)))
    picks = pick_has + pick_no
    random.shuffle(picks)

    cards = "".join(render_sample_card(r, i + 1) for i, r in enumerate(picks))

    summary = f"""
<div class='summary'>
  <h2>Aggregate over all {n} samples in step {step}</h2>
  <div class='summary-grid'>
    <div><div class='label'>has &lt;/think&gt;</div><div class='val'>{len(has_close)} / {n} ({100*len(has_close)/max(n,1):.1f}%)</div></div>
    <div><div class='label'>has &lt;think&gt;</div><div class='val'>{len(has_open_only)} / {n} ({100*len(has_open_only)/max(n,1):.1f}%)</div></div>
    <div><div class='label'>avg unified_score</div><div class='val'>{avg('unified_score'):.3f}</div></div>
    <div><div class='label'>avg format_score</div><div class='val'>{avg('format_score'):.3f}</div></div>
    <div><div class='label'>avg f1_temporal</div><div class='val'>{avg('f1_temporal'):.3f}</div></div>
    <div><div class='label'>avg f1_segment</div><div class='val'>{avg('f1_segment'):.3f}</div></div>
    <div><div class='label'>avg think words (pre-/think)</div><div class='val'>{avg_think_w:.0f}</div></div>
    <div><div class='label'>avg stop_reason_is_length</div><div class='val'>{avg('stop_reason_is_length'):.3f}</div></div>
  </div>
  <p style='font-size: 12px; color: #666; margin-top: 12px; margin-bottom: 0;'>
    The system prompt instructs the model to put reasoning between <code>&lt;think&gt;</code> and <code>&lt;/think&gt;</code>.
    Below: up to {SAMPLES_PER_BUCKET_HAS} random samples that DID emit <code>&lt;/think&gt;</code> + up to {SAMPLES_PER_BUCKET_NO} that did NOT.
  </p>
</div>"""

    stats = {
        "step": step,
        "n": n,
        "has_close": len(has_close),
        "has_open_only": len(has_open_only),
        "avg_unified": avg("unified_score"),
        "avg_format": avg("format_score"),
        "avg_f1_seg": avg("f1_segment"),
        "avg_f1_temp": avg("f1_temporal"),
        "avg_think_words": avg_think_w,
        "avg_stop_len": avg("stop_reason_is_length"),
    }

    html_doc = f"""<!doctype html>
<html><head><meta charset='utf-8'>
<title>hgmkw8sg · step {step} rollouts</title>
<style>{ROLLOUT_CSS}</style>
</head><body>
<h1>Run hgmkw8sg · step {step} rollouts</h1>
<div class='sub'>
  <a href='{WANDB_URL}'>wandb run</a>
  · checkpoint: <code>{CKPT_NAME}</code>
  · enable_thinking=True · response length did NOT collapse
</div>
{summary}
{cards}
</body></html>"""
    return html_doc, stats


# ---------------------------- 2) early-step rollouts (1..7) ----------------------------

def build_early_rollouts() -> tuple[str, list[dict]]:
    stats_list = []
    per_step_html = {}
    for step in EARLY_STEPS:
        rows = load_step(step)
        n = len(rows)
        if n == 0:
            continue
        has_close = [r for r in rows if "</think>" in (r.get("output") or "")]
        avg = lambda k: sum(float(r.get(k, 0) or 0) for r in rows) / max(n, 1)
        avg_think_w = sum(actual_think_words(r.get("output") or "") for r in rows) / max(n, 1)
        truncated = sum(1 for r in rows if float(r.get("stop_reason_is_length", 0) or 0) > 0.5)

        # Sort rows by unified_score descending and pick 20 evenly spaced
        rows_sorted = sorted(rows, key=lambda r: float(r.get("unified_score", 0) or 0), reverse=True)
        step_size = max(1, len(rows_sorted) // 20)
        picks = rows_sorted[::step_size][:20]

        cards = "".join(render_sample_card(r, i + 1) for i, r in enumerate(picks))
        per_step_html[step] = f"""
<div class='step-anchor' id='step-{step}'>step {step}</div>
<div class='summary'>
  <h2>Aggregate over all {n} samples in step {step}</h2>
  <div class='summary-grid'>
    <div><div class='label'>has &lt;/think&gt;</div><div class='val'>{len(has_close)} / {n} ({100*len(has_close)/max(n,1):.1f}%)</div></div>
    <div><div class='label'>truncated (len stop)</div><div class='val'>{truncated} / {n} ({100*truncated/max(n,1):.1f}%)</div></div>
    <div><div class='label'>avg unified_score</div><div class='val'>{avg('unified_score'):.3f}</div></div>
    <div><div class='label'>avg format_score</div><div class='val'>{avg('format_score'):.3f}</div></div>
    <div><div class='label'>avg f1_temporal</div><div class='val'>{avg('f1_temporal'):.3f}</div></div>
    <div><div class='label'>avg f1_segment</div><div class='val'>{avg('f1_segment'):.3f}</div></div>
    <div><div class='label'>avg think words (pre-/think)</div><div class='val'>{avg_think_w:.0f}</div></div>
    <div><div class='label'>avg stop_reason_is_length</div><div class='val'>{avg('stop_reason_is_length'):.3f}</div></div>
  </div>
  <p style='font-size: 12px; color: #666; margin-top: 8px; margin-bottom: 0;'>
    Showing 20 samples evenly spaced across the unified_score distribution (best at top).
  </p>
</div>
{cards}"""
        stats_list.append({
            "step": step, "n": n,
            "has_close": len(has_close),
            "trunc": truncated,
            "avg_unified": avg("unified_score"),
            "avg_format": avg("format_score"),
            "avg_f1_seg": avg("f1_segment"),
            "avg_f1_temp": avg("f1_temporal"),
            "avg_think_words": avg_think_w,
        })

    nav_links = " ".join(
        f"<a href='#step-{s['step']}'>step {s['step']}</a>" for s in stats_list
    )
    summary_table_rows = "".join(
        f"<tr><td>{s['step']}</td><td>{s['n']}</td>"
        f"<td>{s['has_close']} ({100*s['has_close']/max(s['n'],1):.1f}%)</td>"
        f"<td>{s['trunc']} ({100*s['trunc']/max(s['n'],1):.1f}%)</td>"
        f"<td>{s['avg_unified']:.3f}</td>"
        f"<td>{s['avg_format']:.3f}</td>"
        f"<td>{s['avg_f1_seg']:.3f}</td>"
        f"<td>{s['avg_f1_temp']:.3f}</td>"
        f"<td>{s['avg_think_words']:.0f}</td></tr>"
        for s in stats_list
    )

    html_doc = f"""<!doctype html>
<html><head><meta charset='utf-8'>
<title>hgmkw8sg · early-step rollouts (1-7)</title>
<style>{ROLLOUT_CSS}
.nav a {{ margin-right: 12px; }}
.summary-table {{ width:100%; border-collapse:collapse; font-size:13px; }}
.summary-table th, .summary-table td {{ padding:5px 9px; border:1px solid #ddd; text-align:right; }}
.summary-table th {{ background:#eee; }}
.summary-table td:first-child, .summary-table th:first-child {{ text-align:left; }}
</style></head><body>
<h1>Run hgmkw8sg · early-step rollouts (1-7)</h1>
<div class='sub'>
  <a href='{WANDB_URL}'>wandb run</a> · checkpoint: <code>{CKPT_NAME}</code>
  · response length did NOT collapse
</div>
<div class='nav'>Jump to: {nav_links}</div>
<div class='summary'>
  <h2>Per-step summary</h2>
  <table class='summary-table'>
    <tr><th>step</th><th>n</th><th>has &lt;/think&gt;</th><th>truncated</th><th>avg unified</th><th>avg format</th><th>avg f1_seg</th><th>avg f1_temp</th><th>avg think words</th></tr>
    {summary_table_rows}
  </table>
</div>
{''.join(per_step_html.values())}
</body></html>"""
    return html_doc, stats_list


# ---------------------------- 3) significant-think (>=100 words) ----------------------------

def render_sig_card(row: dict, think_text: str) -> str:
    sample_id = row.get("sample_id", "?")
    short_id = sample_id.replace("-", "")[:32]
    data_source = row.get("data_source", "?")
    output = row.get("output") or ""
    gts = row.get("gts") or ""
    think_words = len(think_text.split())
    # Answer part (after </think>)
    if "</think>" in output:
        answer = output.split("</think>", 1)[1].strip()
    else:
        answer = ""

    metric_html = "".join(
        f"<div class='m'><span class='k'>{html.escape(k)}</span>"
        f"<span class='v'>{fmt_metric(row, k)}</span></div>"
        for k in METRIC_KEYS
    )
    return f"""
<div class='card'>
  <div class='hdr'>
    <div class='title'>sample <code>{html.escape(short_id)}</code></div>
    <div><span class='badge t'>{think_words} think words</span></div>
  </div>
  <div class='meta'>
    <span>data_source: <code>{html.escape(str(data_source))[:80]}</code></span>
  </div>
  <div class='metrics'>{metric_html}</div>
  <details open><summary>think (pre-&lt;/think&gt;, {think_words} words)</summary>
    <pre class='think'>{html.escape(think_text)}</pre>
  </details>
  <details><summary>answer (post-&lt;/think&gt;)</summary>
    <pre class='ans'>{html.escape(answer)}</pre>
  </details>
  <details><summary>ground truth</summary>
    <pre class='gt'>{html.escape(gts)}</pre>
  </details>
</div>"""


def build_significant_think() -> str:
    """Per-step counts table + samples with pre-</think> >= 100 words."""
    table_rows = []
    sections = []
    for step in EARLY_STEPS:
        rows = load_step(step)
        n_total = len(rows)
        sig_rows = []
        think_word_counts = []
        for r in rows:
            out = r.get("output") or ""
            if "</think>" not in out:
                continue
            pre = pre_think(out)
            wc = len(pre.split())
            if wc >= SIG_THINK_WORDS:
                sig_rows.append((r, pre, wc))
                think_word_counts.append(wc)
        n_sig = len(sig_rows)
        avg_words = sum(think_word_counts) / max(n_sig, 1)
        avg_unified = (
            sum(float(r.get("unified_score", 0) or 0) for r, _, _ in sig_rows)
            / max(n_sig, 1)
        )
        table_rows.append(
            f"<tr><td>{step}</td><td>{n_total}</td>"
            f"<td>{n_sig} ({100*n_sig/max(n_total,1):.1f}%)</td>"
            f"<td>{avg_words:.0f}</td>"
            f"<td>{avg_unified:.3f}</td></tr>"
        )
        # Show SIG_SAMPLES_PER_STEP samples ranked by think_word count desc
        sig_rows.sort(key=lambda x: x[2], reverse=True)
        picks = sig_rows[:SIG_SAMPLES_PER_STEP]
        if picks:
            cards = "".join(render_sig_card(r, pre) for r, pre, _ in picks)
            sections.append(
                f"<div class='step-anchor' id='sig-step-{step}'>step {step} — top {len(picks)} by think length</div>"
                + cards
            )

    html_doc = f"""<!doctype html>
<html><head><meta charset='utf-8'>
<title>hgmkw8sg · significant-think rollouts (steps 1-7)</title>
<style>{ROLLOUT_CSS}
.summary-table {{ width:100%; border-collapse:collapse; font-size:13px; }}
.summary-table th, .summary-table td {{ padding:5px 9px; border:1px solid #ddd; text-align:right; }}
.summary-table th {{ background:#eee; }}
.summary-table td:first-child, .summary-table th:first-child {{ text-align:left; }}
</style></head><body>
<h1>Significant-think rollouts (steps 1-7)</h1>
<div class='sub'>
  <a href='{WANDB_URL}'>hgmkw8sg</a>
  · checkpoint: <code>{CKPT_NAME}</code>
  · filter: pre-<code>&lt;/think&gt;</code> content has &ge; {SIG_THINK_WORDS} words
  · response length did NOT collapse
</div>
<div class='summary'><h2>Per-step counts</h2>
<table class='summary-table'>
<tr><th>step</th><th>total rollouts</th><th>with significant think (&ge;{SIG_THINK_WORDS}w)</th><th>avg think words (when present)</th><th>avg unified (significant)</th></tr>
{''.join(table_rows)}
</table>
<p style='font-size:12px;color:#666;margin-top:8px;margin-bottom:0'>
  Below: top {SIG_SAMPLES_PER_STEP} samples per step ranked by think word count.
</p>
</div>
{''.join(sections)}
</body></html>"""
    return html_doc


# ---------------------------- 4) cognitive patterns v2 ----------------------------

def build_cognitive_patterns() -> str:
    # 1. trend over steps: pattern hit rate among rollouts with non-empty think
    # 2. per-metric pattern win/lose differential

    # Collect: for each step, for each rollout with non-empty think, pattern hits + metrics
    step_pattern_rate: dict[int, dict[str, float]] = {}
    step_n: dict[int, int] = {}

    # For per-metric tables: aggregate over all trend steps a list of rollouts
    # with non-empty think, with their pattern hits and metric values
    all_rollouts: list[dict] = []

    for step in TREND_STEPS:
        rows = load_step(step)
        if not rows:
            continue
        bucket = []
        for r in rows:
            out = r.get("output") or ""
            pre = pre_think(out)
            if not pre.strip():
                continue
            # require at least some think content (>0 words)
            words = pre.split()
            if len(words) < 5:
                continue
            hits = detect_patterns(pre)
            bucket.append({
                "patterns": hits,
                "score": float(r.get("score", 0) or 0),
                "unified_score": float(r.get("unified_score", 0) or 0),
                "f1_segment": float(r.get("f1_segment", 0) or 0),
                "f1_temporal": float(r.get("f1_temporal", 0) or 0),
                "format_score": float(r.get("format_score", 0) or 0),
                "step": step,
            })
        if not bucket:
            continue
        step_n[step] = len(bucket)
        step_pattern_rate[step] = {
            name: sum(1 for b in bucket if b["patterns"][name]) / len(bucket)
            for name in PATTERN_ORDER
        }
        all_rollouts.extend(bucket)

    if not all_rollouts:
        return "<html><body>No rollouts found.</body></html>"

    # For each metric, bucket samples by per-sample value;
    # since we have a single run (no two model comparison), reinterpret:
    #   high vs low buckets per metric (above-median vs below-median).
    # That captures pattern association: "when the model gets HIGH unified, which patterns appear?"
    METRICS = ["score", "unified_score", "f1_segment", "f1_temporal", "format_score"]
    metric_tables_html = []
    for metric in METRICS:
        vals = sorted(b[metric] for b in all_rollouts)
        if not vals:
            continue
        # quartile split: top 25% (win) vs bottom 25% (lose)
        q25 = vals[len(vals) // 4]
        q75 = vals[3 * len(vals) // 4]
        if q25 == q75:
            # too many ties (e.g. all zero for some metric) — fall back to nonzero vs zero
            high = [b for b in all_rollouts if b[metric] > 0]
            low = [b for b in all_rollouts if b[metric] == 0]
            label = f"split=nonzero vs zero (n={len(all_rollouts)})"
        else:
            high = [b for b in all_rollouts if b[metric] >= q75]
            low = [b for b in all_rollouts if b[metric] <= q25]
            label = f"top-25% (≥{q75:.3f}) vs bottom-25% (≤{q25:.3f})"
        if not high or not low:
            continue
        rows_data = []
        for name in PATTERN_ORDER:
            high_pct = 100 * sum(1 for b in high if b["patterns"][name]) / len(high)
            low_pct = 100 * sum(1 for b in low if b["patterns"][name]) / len(low)
            diff = high_pct - low_pct
            rows_data.append((name, high_pct, low_pct, diff))
        rows_data.sort(key=lambda x: abs(x[3]), reverse=True)
        body = "".join(
            f"<tr class='{'pos' if d > 0 else 'neg'}'>"
            f"<td>{n}</td><td>{h:.1f}%</td><td>{l:.1f}%</td>"
            f"<td><b>{d:+.1f}</b></td></tr>"
            for n, h, l, d in rows_data
        )
        metric_tables_html.append(
            f"<h3>{metric} <span class='small'>(win n={len(high)}, "
            f"lose n={len(low)}, {label})</span></h3>"
            f"<table><tr><th>pattern</th><th>win %</th><th>lose %</th><th>Δ pp</th></tr>{body}</table>"
        )

    # Trend SVG: pattern rate vs step.
    # Split patterns into "win-correlated" (Δ > 0 on unified_score) and "lose-correlated" (Δ < 0)
    # using the same quartile split as the unified_score table above.
    win_patterns = []
    lose_patterns = []
    if all_rollouts:
        u_vals = sorted(b["unified_score"] for b in all_rollouts)
        u_q25 = u_vals[len(u_vals) // 4]
        u_q75 = u_vals[3 * len(u_vals) // 4]
        if u_q25 == u_q75:
            high_u = [b for b in all_rollouts if b["unified_score"] > 0]
            low_u = [b for b in all_rollouts if b["unified_score"] == 0]
        else:
            high_u = [b for b in all_rollouts if b["unified_score"] >= u_q75]
            low_u = [b for b in all_rollouts if b["unified_score"] <= u_q25]
        for name in PATTERN_ORDER:
            h_pct = 100 * sum(1 for b in high_u if b["patterns"][name]) / max(len(high_u), 1)
            l_pct = 100 * sum(1 for b in low_u if b["patterns"][name]) / max(len(low_u), 1)
            (win_patterns if h_pct > l_pct else lose_patterns).append(name)

    COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e",
              "#17becf", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22",
              "#aec7e8", "#ffbb78", "#98df8a"]

    def render_trend_svg(pattern_list: list[str], title: str) -> str:
        ordered_steps = sorted(step_pattern_rate.keys())
        if not ordered_steps or not pattern_list:
            return f"<h3>{title}</h3><p style='color:#999;font-size:12px'>(no patterns in this class)</p>"
        s_min, s_max = ordered_steps[0], ordered_steps[-1]
        W, H = 760, 220
        L, R, T, B = 38, 130, 18, 30
        plot_w = W - L - R
        plot_h = H - T - B
        # 5 horizontal gridlines (0%, 25%, 50%, 75%, 100%)
        gridlines = []
        for pct, y_frac in zip([0, 25, 50, 75, 100], [1.0, 0.75, 0.5, 0.25, 0.0]):
            y = T + plot_h * y_frac
            gridlines.append(
                f"<line x1='{L}' y1='{y:.1f}' x2='{L+plot_w}' y2='{y:.1f}' stroke='#eee'/>"
                f"<text x='{L-3}' y='{y+3:.1f}' font-size='10' text-anchor='end'>{pct}%</text>"
            )
        # x ticks
        x_ticks = []
        for s in ordered_steps[::max(1, len(ordered_steps)//7)]:
            x = L + plot_w * (s - s_min) / max(s_max - s_min, 1)
            x_ticks.append(f"<text x='{x:.1f}' y='{T+plot_h+14}' font-size='10' text-anchor='middle'>{s}</text>")
        paths = []
        legend = []
        for i, name in enumerate(pattern_list):
            color = COLORS[i % len(COLORS)]
            pts = []
            for s in ordered_steps:
                rate = step_pattern_rate[s].get(name, 0)
                x = L + plot_w * (s - s_min) / max(s_max - s_min, 1)
                y = T + plot_h * (1 - rate)
                pts.append(f"{x:.1f},{y:.1f}")
            path = "M" + " L".join(pts)
            paths.append(f"<path d='{path}' stroke='{color}' fill='none' stroke-width='1.6'/>")
            ly = T + i * 16
            legend.append(
                f"<rect x='{L+plot_w+8}' y='{ly}' width='10' height='10' fill='{color}'/>"
                f"<text x='{L+plot_w+22}' y='{ly+9}' font-size='11'>{name}</text>"
            )
        svg = (
            f"<svg viewBox='0 0 {W} {H}' style='width:100%;max-width:{W}px;border:1px solid #eee;background:#fafafa'>"
            + "".join(gridlines)
            + "".join(x_ticks)
            + "".join(paths)
            + "".join(legend)
            + f"<text x='{L+plot_w/2}' y='{H-4}' font-size='11' text-anchor='middle' fill='#666'>training step</text>"
            + "</svg>"
        )
        return f"<h3>{title}</h3>{svg}"

    trend_html = render_trend_svg(win_patterns, "Win-correlated patterns (% of rollouts with ≥1 hit)") + \
                 render_trend_svg(lose_patterns, "Lose-correlated patterns")

    css = """
body{font-family:-apple-system,system-ui,sans-serif;max-width:1500px;margin:18px auto;padding:0 14px;color:#222}
h1,h2{border-bottom:1px solid #ccc;padding-bottom:4px}
.summary{background:#fafafa;border:1px solid #ddd;padding:10px 14px;border-radius:6px;margin:10px 0}
table{border-collapse:collapse;margin:6px 0;font-size:13px}
td,th{border:1px solid #ddd;padding:3px 9px;text-align:left}
th{background:#eee}
tr.pos td:last-child{color:#1b5e20;font-weight:bold}
tr.neg td:last-child{color:#b71c1c;font-weight:bold}
.small{color:#666;font-size:11px}
.note{background:#fff3e0;border-left:4px solid #ff9800;padding:8px 12px;margin:8px 0;font-size:13.5px}
nav a{margin-right:14px;font-size:13px}
"""

    total_rollouts = sum(step_n.values())

    return f"""<!doctype html>
<html><head><meta charset='utf-8'>
<title>Cognitive patterns — hgmkw8sg (no-collapse run)</title>
<style>{css}</style></head><body>
<h1>Cognitive Patterns — hgmkw8sg (response length did NOT collapse)</h1>
<nav>
 <a href='#m'>Per-metric tables</a>
 <a href='#trend'>Monotonic trends</a>
</nav>
<div class='summary'>
 Run: <a href='{WANDB_URL}'>hgmkw8sg</a> · checkpoint: <code>{CKPT_NAME}</code><br>
 Steps analyzed (trends): {len(step_pattern_rate)} steps ({', '.join(str(s) for s in sorted(step_pattern_rate)[:12])}, ...)<br>
 Total rollouts with non-empty &lt;think&gt;: <b>{total_rollouts}</b><br>
 Bucket split: per-metric quartile (top-25% win vs bottom-25% lose)
</div>

<h2 id='m'>Per-metric pattern differentials (single-run interpretation)</h2>
<div class='note'>
For each metric, split rollouts into top-25% (win) and bottom-25% (lose) buckets,
then compute pattern hit rates per bucket. Δpp &gt; 0 = pattern more common in win rollouts.
The original 260503 analysis compared two models (think vs nothink); here we have one run,
so we compare the model's own win-vs-lose rollouts.
</div>
{''.join(metric_tables_html)}

<h2 id='trend'>Monotonic trends — pattern rate vs training step</h2>
<div class='note'>Does the model adopt win-patterns over training? Drop lose-patterns?
Patterns are classified as win/lose based on the unified_score median split above.
Since response length did not collapse here, sustained think content lets us track
pattern evolution across all {len(step_pattern_rate)} sampled steps.</div>
{trend_html}
</body></html>"""


# ---------------------------- main ----------------------------

def main():
    # 1) early-step rollouts
    early_html, early_stats = build_early_rollouts()
    (OUT_DIR / "260611_hgmkw8sg_early_rollouts.html").write_text(early_html)
    print(f"wrote early_rollouts ({len(early_html)} bytes)")

    # 2) significant-think
    sig_html = build_significant_think()
    (OUT_DIR / "260611_hgmkw8sg_significant_think.html").write_text(sig_html)
    print(f"wrote significant_think ({len(sig_html)} bytes)")

    # 3) step-400 rollouts
    step_html, step_stats = build_step_rollouts(INSPECT_STEP)
    (OUT_DIR / f"260611_hgmkw8sg_step{INSPECT_STEP}_rollouts.html").write_text(step_html)
    print(f"wrote step{INSPECT_STEP}_rollouts ({len(step_html)} bytes, stats={step_stats})")

    # 4) cognitive patterns
    pat_html = build_cognitive_patterns()
    (OUT_DIR / "260611_hgmkw8sg_cognitive_patterns.html").write_text(pat_html)
    print(f"wrote cognitive_patterns ({len(pat_html)} bytes)")

    # 5) wrapper
    early_opts = "\n".join(
        f"<option value='{s['step']}'>step {s['step']} (avg unified {s['avg_unified']:.3f}, "
        f"avg think {s['avg_think_words']:.0f}w)</option>"
        for s in early_stats
    )
    wrapper = f"""<!doctype html><html><head><meta charset='utf-8'>
<title>RL rollouts — hgmkw8sg (no-collapse)</title>
<style>
body{{font-family:-apple-system,Segoe UI,sans-serif;margin:14px;color:#222;background:#fafafa}}
h1{{margin:0 0 4px}}
.meta{{color:#666;font-size:13px;margin-bottom:14px}}
.toptabs{{display:flex;gap:4px;margin-bottom:0;border-bottom:2px solid #1a4d8c;padding-bottom:0}}
.toptabs button{{padding:8px 16px;border:1px solid #ccc;border-bottom:none;background:#f0f0f0;cursor:pointer;font-size:13px;border-radius:4px 4px 0 0}}
.toptabs button.active{{background:#1a4d8c;color:#fff;border-color:#1a4d8c}}
.tabpane{{display:none}}
.tabpane.active{{display:block}}
.tabpane iframe{{width:100%;height:calc(100vh - 100px);border:none;background:#fff}}
a{{color:#1a4d8c}}
code{{background:#f4f4f4;padding:1px 4px;border-radius:3px;font-size:11px}}
</style></head><body>
<h1>RL rollouts — hgmkw8sg (response length did NOT collapse)</h1>
<div class='meta'>
  Run: <a href='{WANDB_URL}'>hgmkw8sg</a> ·
  checkpoint: <code>{CKPT_NAME}</code> ·
  generated 2026-06-11 · replicates 260508_rl_eval_research_questions rollout-side analyses
</div>
<div class='toptabs'>
  <button class='active' data-tab='early'>1) early-step rollouts (1-7)</button>
  <button data-tab='sigthink'>2) significant-think (≥{SIG_THINK_WORDS}w)</button>
  <button data-tab='late'>3) step {INSPECT_STEP} rollouts</button>
  <button data-tab='pattern'>4) cognitive patterns</button>
</div>
<div id='tab-early' class='tabpane active'>
  <div style='margin:12px 0;font-size:13px'>
    Early-step rollouts. Step:
    <select id='early-step' style='padding:4px 10px;font-size:13px'>
{early_opts}
    </select>
    <a href='260611_hgmkw8sg_early_rollouts.html' target='_blank' style='margin-left:12px'>open in new tab</a>
  </div>
  <iframe id='early-iframe' src='260611_hgmkw8sg_early_rollouts.html#step-1' style='width:100%;height:calc(100vh-160px);border:none;background:#fff'></iframe>
</div>
<div id='tab-sigthink' class='tabpane'><iframe src='260611_hgmkw8sg_significant_think.html'></iframe></div>
<div id='tab-late' class='tabpane'><iframe src='260611_hgmkw8sg_step{INSPECT_STEP}_rollouts.html'></iframe></div>
<div id='tab-pattern' class='tabpane'><iframe src='260611_hgmkw8sg_cognitive_patterns.html'></iframe></div>
<script>
document.querySelectorAll('.toptabs button').forEach(btn => {{
  btn.addEventListener('click', () => {{
    document.querySelectorAll('.toptabs button').forEach(b => b.classList.remove('active'));
    document.querySelectorAll('.tabpane').forEach(p => p.classList.remove('active'));
    btn.classList.add('active');
    document.getElementById('tab-' + btn.dataset.tab).classList.add('active');
  }});
}});
document.getElementById('early-step').addEventListener('change', e => {{
  const f = document.getElementById('early-iframe');
  f.src = '260611_hgmkw8sg_early_rollouts.html#step-' + e.target.value;
}});
</script>
</body></html>"""
    (OUT_DIR / "260611_hgmkw8sg_rl_no_collapse.html").write_text(wrapper)
    print(f"wrote wrapper ({len(wrapper)} bytes)")


if __name__ == "__main__":
    main()

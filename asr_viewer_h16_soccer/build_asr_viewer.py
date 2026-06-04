"""
Build a self-contained HTML viewer for sme_eval_v3.1_fast / H16_SOCCER to
inspect whether ASR alone could solve the segment retrieval task.

Per sample card:
  - question (user_query_segment)
  - GT bar: all chapters on the video timeline
  - ASR bar: ASR segments within +/-60s of any GT chapter (click to expand text)

Output: HTML to ~/Desktop/html/asr_viewer_h16_soccer.html
"""

import json
import os
from pathlib import Path

from datasets import load_dataset

WINDOW_SECONDS = 60
DATASET = "twelvelabs/sme_eval_v3.1_fast"
CONFIG = "H16_SOCCER"
SPLIT = "test"
OUT_PATH = Path.home() / "Desktop" / "html" / "asr_viewer_h16_soccer.html"


def load_hf_token() -> str:
    env_text = (Path.home() / "pegasus" / ".env").read_text()
    for line in env_text.splitlines():
        if line.startswith("HF_TOKEN="):
            return line.split("=", 1)[1].strip()
    raise RuntimeError("HF_TOKEN not found in ~/pegasus/.env")


def in_window(asr_start: float, asr_end: float, chapters: list) -> bool:
    for ch in chapters:
        if asr_end >= ch["start_time"] - WINDOW_SECONDS and asr_start <= ch["end_time"] + WINDOW_SECONDS:
            return True
    return False


def build_payload():
    os.environ["HF_TOKEN"] = load_hf_token()
    ds = load_dataset(DATASET, CONFIG, split=SPLIT)
    samples = []
    for row in ds:
        md = json.loads(row["metadata"])
        sm = md["sample_metadata"][0]
        duration = float(md["media_metadata"][0]["duration"])
        question = (sm.get("user_query_segment") or ["(no question)"])[0]
        chapters = sm.get("chapters", [])
        gt = [
            {
                "start": float(ch["start_time"]),
                "end": float(ch["end_time"]),
                "label": " / ".join(
                    str(ch[k]) for k in ("player", "team", "header_result", "display_time") if ch.get(k)
                ),
            }
            for ch in chapters
        ]
        asr_obj = sm.get("asr") or {}
        asr_segs = asr_obj.get("segments", []) if isinstance(asr_obj, dict) else []
        asr = []
        for seg in asr_segs:
            s, e = float(seg["start"]), float(seg["end"])
            if not in_window(s, e, chapters):
                continue
            asr.append(
                {
                    "start": s,
                    "end": e,
                    "text": (seg.get("asr") or {}).get("text", ""),
                    "speaker": seg.get("speaker_id", ""),
                }
            )
        samples.append(
            {
                "id": row["id"],
                "media_path": row["media"][0]["media_path"] if row["media"] else "",
                "duration": duration,
                "question": question,
                "gt": gt,
                "asr": asr,
            }
        )
    return samples


HTML_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>ASR viewer - H16_SOCCER</title>
<style>
  :root { --gt: #2563eb; --asr: #f59e0b; --asr-hi: #ef4444; --bg: #fafafa; --card: #fff; --text: #111; --muted: #666; }
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: var(--bg); color: var(--text); margin: 0; padding: 24px; }
  h1 { margin: 0 0 4px; }
  .meta { color: var(--muted); font-size: 13px; margin-bottom: 24px; }
  .card { background: var(--card); border: 1px solid #e5e5e5; border-radius: 8px; padding: 16px; margin-bottom: 16px; box-shadow: 0 1px 2px rgba(0,0,0,0.04); }
  .question { font-size: 14px; line-height: 1.5; margin-bottom: 12px; }
  .question b { color: var(--muted); font-weight: 600; }
  .sample-meta { font-size: 12px; color: var(--muted); margin-bottom: 12px; font-family: ui-monospace, Menlo, monospace; }
  .track-label { font-size: 11px; color: var(--muted); margin: 8px 0 4px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; }
  .track { position: relative; height: 22px; background: #f0f0f0; border-radius: 3px; overflow: hidden; }
  .block { position: absolute; top: 0; bottom: 0; min-width: 2px; cursor: pointer; opacity: 0.85; }
  .block.gt { background: var(--gt); }
  .block.asr { background: var(--asr); }
  .block:hover { opacity: 1; outline: 1px solid #111; }
  .axis { position: relative; height: 14px; margin-top: 2px; font-size: 10px; color: var(--muted); }
  .axis span { position: absolute; transform: translateX(-50%); }
  .legend { font-size: 11px; color: var(--muted); margin-top: 6px; }
  .legend .dot { display: inline-block; width: 10px; height: 10px; border-radius: 2px; vertical-align: middle; margin: 0 4px 0 12px; }
  .asr-detail { margin-top: 8px; padding: 10px 12px; background: #fff7ed; border-left: 3px solid var(--asr); border-radius: 3px; font-size: 13px; display: none; }
  .asr-detail.show { display: block; }
  .asr-detail .t { color: var(--muted); font-family: ui-monospace, Menlo, monospace; font-size: 11px; margin-right: 8px; }
  .gt-detail { margin-top: 8px; padding: 10px 12px; background: #eff6ff; border-left: 3px solid var(--gt); border-radius: 3px; font-size: 13px; display: none; }
  .gt-detail.show { display: block; }
</style>
</head>
<body>
<h1>ASR viewer - H16_SOCCER</h1>
<div class="meta">Dataset: {DATASET} / {CONFIG} / {SPLIT} - {N} samples - ASR filtered to +/-{W}s of any GT chapter. Click a block to see its text.</div>
<div id="root"></div>
<script>
const DATA = {DATA};

function fmtTime(t) {
  const m = Math.floor(t / 60).toString().padStart(2, '0');
  const s = (t % 60).toFixed(1).padStart(4, '0');
  return `${m}:${s}`;
}

function renderSample(sample, idx) {
  const card = document.createElement('div');
  card.className = 'card';
  const dur = sample.duration;
  card.innerHTML = `
    <div class="question"><b>Q${idx + 1}.</b> ${escape(sample.question)}</div>
    <div class="sample-meta">id=${sample.id.slice(0, 12)}... | duration=${fmtTime(dur)} (${dur.toFixed(0)}s) | gt_chapters=${sample.gt.length} | asr_in_window=${sample.asr.length}</div>
    <div class="track-label">Ground truth chapters (${sample.gt.length})</div>
    <div class="track" data-track="gt"></div>
    <div class="axis"><span style="left:0%">0</span><span style="left:25%">${fmtTime(dur*0.25)}</span><span style="left:50%">${fmtTime(dur*0.5)}</span><span style="left:75%">${fmtTime(dur*0.75)}</span><span style="left:100%">${fmtTime(dur)}</span></div>
    <div class="gt-detail"></div>
    <div class="track-label">ASR within +/-60s of GT (${sample.asr.length} chunks)</div>
    <div class="track" data-track="asr"></div>
    <div class="legend"><span class="dot" style="background:var(--gt)"></span>GT chapter <span class="dot" style="background:var(--asr)"></span>ASR chunk</div>
    <div class="asr-detail"></div>
  `;
  const gtTrack = card.querySelector('[data-track="gt"]');
  const asrTrack = card.querySelector('[data-track="asr"]');
  const gtDetail = card.querySelector('.gt-detail');
  const asrDetail = card.querySelector('.asr-detail');

  sample.gt.forEach((g, i) => {
    const b = document.createElement('div');
    b.className = 'block gt';
    b.style.left = (g.start / dur * 100) + '%';
    b.style.width = Math.max(0.2, (g.end - g.start) / dur * 100) + '%';
    b.title = `${fmtTime(g.start)}-${fmtTime(g.end)}: ${g.label}`;
    b.onclick = () => {
      gtDetail.innerHTML = `<span class="t">${fmtTime(g.start)}-${fmtTime(g.end)}</span>${escape(g.label)}`;
      gtDetail.classList.add('show');
    };
    gtTrack.appendChild(b);
  });

  sample.asr.forEach((a) => {
    const b = document.createElement('div');
    b.className = 'block asr';
    b.style.left = (a.start / dur * 100) + '%';
    b.style.width = Math.max(0.2, (a.end - a.start) / dur * 100) + '%';
    b.title = `${fmtTime(a.start)}-${fmtTime(a.end)}: ${a.text.slice(0, 80)}`;
    b.onclick = () => {
      asrDetail.innerHTML = `<span class="t">${fmtTime(a.start)}-${fmtTime(a.end)} [${escape(a.speaker)}]</span>${escape(a.text)}`;
      asrDetail.classList.add('show');
    };
    asrTrack.appendChild(b);
  });
  return card;
}

function escape(s) {
  return String(s == null ? '' : s).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
}

const root = document.getElementById('root');
DATA.forEach((s, i) => root.appendChild(renderSample(s, i)));
</script>
</body>
</html>
"""


def main():
    samples = build_payload()
    html = (
        HTML_TEMPLATE.replace("{DATASET}", DATASET)
        .replace("{CONFIG}", CONFIG)
        .replace("{SPLIT}", SPLIT)
        .replace("{N}", str(len(samples)))
        .replace("{W}", str(WINDOW_SECONDS))
        .replace("{DATA}", json.dumps(samples))
    )
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(html)
    print(f"wrote {OUT_PATH} ({len(html):,} bytes, {len(samples)} samples)")


if __name__ == "__main__":
    main()

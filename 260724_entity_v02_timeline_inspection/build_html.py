#!/usr/bin/env python3
"""Build a small offline inspection page for the dashboard timeline data."""

import html
import json
from pathlib import Path

from timeline_data import timeline_records


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "260722_entity_v02_streamlit/gemini_timeline_data.json"
OUTPUT_PATH = Path(__file__).with_name("timeline_inspection.html")
SAMPLE_ID = "film-01:000"


def main() -> None:
    data = json.loads(DATA_PATH.read_text())
    sections = []
    for model_name, model in data["models"].items():
        sample = model["samples"][SAMPLE_ID]
        records, duration = timeline_records(
            sample["prediction"],
            data["ground_truth"][SAMPLE_ID],
            sample["character_scores"],
            mapping=sample["mapping"],
        )
        records_by_lane = {}
        for record in records:
            records_by_lane.setdefault(record["lane"], []).append(record)
        lanes = []
        for lane, lane_records in records_by_lane.items():
            source = lane_records[0]["source"]
            bars = "".join(
                '<span class="bar {}" style="left:{:.4f}%;width:{:.4f}%" '
                'title="{:.2f}–{:.2f}s"></span>'.format(
                    "gt" if source == "GT" else "prediction",
                    100 * record["start"] / duration,
                    max(0.12, 100 * (record["end"] - record["start"]) / duration),
                    record["start"],
                    record["end"],
                )
                for record in lane_records
            )
            lanes.append(
                f'<div class="lane"><div title="{html.escape(lane)}">'
                f"{html.escape(lane)}</div><div class=track>{bars}</div></div>"
            )
        statistics = model["statistics"]
        sections.append(
            f"<section><h2>{html.escape(model_name)}</h2>"
            f"<p>{statistics['parsed_media_count']}/18 parsed · "
            f"avg shots {statistics['average_shots']:.1f} · "
            f"pred/GT shots {statistics['average_predicted_to_ground_truth_shot_count_ratio']:.3f}</p>"
            f"{''.join(lanes)}<div class=axis>0s <span>{duration:.1f}s</span></div></section>"
        )
    OUTPUT_PATH.write_text(
        """<!doctype html><meta charset=utf-8><title>Entity v0.2 timeline inspection</title>
<style>
body{font:13px -apple-system,BlinkMacSystemFont,sans-serif;margin:24px;background:#f6f7f9;color:#1f2937}
h1{margin-bottom:4px} h2{font-size:17px;margin:0} p{color:#667085;margin:5px 0 12px}
section{background:#fff;border:1px solid #d9dee7;border-radius:8px;padding:16px;margin:16px 0}
.lane{display:grid;grid-template-columns:250px 1fr;gap:8px;align-items:center;margin:4px 0}
.lane>div:first-child{white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.track{height:12px;background:#eef1f5;position:relative;border-radius:3px;overflow:hidden}
.bar{position:absolute;top:1px;height:10px;min-width:1px;border-radius:2px}.gt{background:#1769aa}.prediction{background:#d95f02}
.axis{margin-left:258px;color:#667085;border-top:1px solid #98a2b3;padding-top:3px}.axis span{float:right}
</style><h1>Half matched-entity timelines</h1><p>film-01:000 · blue = GT, orange = prediction</p>"""
        + "".join(sections)
    )
    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

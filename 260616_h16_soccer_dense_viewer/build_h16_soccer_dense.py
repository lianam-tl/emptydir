#!/usr/bin/env python
"""
Build ONE row for twelvelabs/sme_eval_v3.1_lia, config H16_SOCCER_DENSE.

Source: "Pegasus 1.5 Goal-Detection Issues: Belarus Soccer Footage" Notion doc.
Placeholder ground-truth chapters = the Pegasus H16_soccer_events output
(the doc's result.json). lia will fix the chapters later.

Schema mirrors the live twelvelabs/sme_eval_v3.1_fast :: H16_SOCCER row exactly:
  columns: id, media, messages(JSON str? -> stored as list), metadata(JSON str)
  - messages: single user turn, video slot with video=null (query lives in metadata)
  - metadata: {media_metadata:[...], sample_metadata:[{...}]}

Usage:
  python build_h16_soccer_dense.py --video /path/to/video.mp4              # dry-run: transcode+probe+build, print row
  python build_h16_soccer_dense.py --video ... --upload-s3                 # also upload transcoded mp4 to S3
  python build_h16_soccer_dense.py --video ... --upload-s3 --push-hf       # also push the config to HF
"""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

REPO_ID = "twelvelabs/sme_eval_v3.1_lia"
CONFIG = "H16_SOCCER_DENSE"
SAMPLE_ID = "belarus_soccer_dense_6a22a2fd"
S3_BUCKET = "tl-data-training-pegasus-us-west-2"
S3_KEY = "raw_media/private/video_from_gtm/dean-pegasus15-soccer.mp4"
S3_URI = f"s3://{S3_BUCKET}/{S3_KEY}"
REGION = "us-west-2"

# --- Query text: the H16_soccer_events description from the doc's input prompt ---
USER_QUERY = (
    "Identify important soccer gameplay events shown in this soccer broadcast, whether "
    "live or replayed. Each segment represents exactly one discrete game event - if "
    "multiple distinct events occur in sequence, emit a separate segment for each. Do not "
    "group more than one event type into a single segment. A segment ends at the moment "
    "the primary action concludes: a save ends when the goalkeeper secures the ball, a "
    "foul ends when the referee whistle sounds, a shot ends at the moment of contact. Do "
    "not extend into the subsequent restart or reaction."
)

EVENT_ENUM = [
    "GOAL",
    "SHOT_ON_TARGET",
    "SHOT_OFF_TARGET",
    "HEADER",
    "SAVE",
    "FREE_KICK",
    "CORNER_KICK",
    "PENALTY_KICK",
    "COUNTER_ATTACK",
    "PASS",
    "DRIBBLE",
    "SUBSTITUTION",
    "FOUL",
    "YELLOW_CARD",
    "RED_CARD",
    "OFFSIDE",
    "VAR_REVIEW",
    "STOPPAGE_TIME",
    "HYDRATION_BREAK",
    "EXTRA_TIME",
    "PENALTY_SHOOT_OUT",
    "CELEBRATION",
]

METADATA_SCHEMA = {
    "name": "H16_DENSE",
    "strict": False,
    "schema": {
        "type": "object",
        "properties": {
            "start_time": {"type": "number", "description": "Segment start (seconds)"},
            "end_time": {"type": "number", "description": "Segment end (seconds)"},
            "event_type": {
                "type": "string",
                "enum": EVENT_ENUM,
                "description": "Primary gameplay event in this segment.",
            },
            "playback_status": {
                "type": "string",
                "enum": ["LIVE_GAMEPLAY", "REPLAY"],
                "description": "Whether the event is shown as live gameplay or replay footage.",
            },
            "match_start_time": {
                "type": "string",
                "description": "Match clock when the event begins, read from scoreboard overlay. e.g. '73:15'.",
            },
            "match_end_time": {"type": "string", "description": "Match clock when the event ends. e.g. '73:45'."},
            "description": {
                "type": "string",
                "description": "1-2 sentence narrative of what happened and who was involved.",
            },
            "player_names": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Players involved in the event.",
            },
        },
        "required": [
            "start_time",
            "end_time",
            "event_type",
            "playback_status",
            "match_start_time",
            "match_end_time",
            "description",
            "player_names",
        ],
        "additionalProperties": False,
    },
}

SEGMENT_DICT = {
    "Segment ID": "H16",
    "Segment Definition": "Dense soccer gameplay events: every discrete shot, save, foul, pass, dribble, goal, etc., live or replayed.",
    "Category": "Sports-specific",
}

# --- Placeholder chapters = Pegasus H16_soccer_events from the doc's result.json ---
# (lia will replace these with corrected ground truth, esp. the GOAL outcomes.)
PEGASUS_H16_EVENTS = [
    {
        "start_time": 0,
        "end_time": 7,
        "event_type": "SHOT_ON_TARGET",
        "playback_status": "LIVE_GAMEPLAY",
        "match_start_time": "00:00",
        "match_end_time": "00:07",
        "description": "A player in a black jersey receives a pass and shoots towards the goal. The goalkeeper makes a diving save to his left, preventing a goal.",
        "player_names": [],
    },
    {
        "start_time": 7,
        "end_time": 34,
        "event_type": "SAVE",
        "playback_status": "LIVE_GAMEPLAY",
        "match_start_time": "00:07",
        "match_end_time": "00:34",
        "description": "The goalkeeper restarts play with a kick. A player in a black jersey intercepts the ball and takes a shot, which the goalkeeper saves again. The referee then signals for a foul.",
        "player_names": [],
    },
    {
        "start_time": 34,
        "end_time": 43,
        "event_type": "PASS",
        "playback_status": "LIVE_GAMEPLAY",
        "match_start_time": "00:34",
        "match_end_time": "00:43",
        "description": "Following the foul, a player in a black jersey takes a free kick, passing the ball to a teammate who then kicks it out of bounds.",
        "player_names": [],
    },
    {
        "start_time": 43,
        "end_time": 54,
        "event_type": "DRIBBLE",
        "playback_status": "LIVE_GAMEPLAY",
        "match_start_time": "00:43",
        "match_end_time": "00:54",
        "description": "A player in a pink bib dribbles the ball down the field, maneuvering past a defender in a black jersey before passing to a teammate.",
        "player_names": [],
    },
    {
        "start_time": 54,
        "end_time": 64,
        "event_type": "FOUL",
        "playback_status": "LIVE_GAMEPLAY",
        "match_start_time": "00:54",
        "match_end_time": "01:04",
        "description": "A player in a pink bib is tackled by a player in a black jersey. The player in pink falls to the ground and appears to be injured.",
        "player_names": [],
    },
    {
        "start_time": 64,
        "end_time": 71,
        "event_type": "SHOT_OFF_TARGET",
        "playback_status": "LIVE_GAMEPLAY",
        "match_start_time": "01:04",
        "match_end_time": "01:11",
        "description": "A player in a black jersey dribbles towards the goal and takes a shot, but the ball goes wide of the target.",
        "player_names": [],
    },
    {
        "start_time": 71,
        "end_time": 81,
        "event_type": "DRIBBLE",
        "playback_status": "LIVE_GAMEPLAY",
        "match_start_time": "01:11",
        "match_end_time": "01:21",
        "description": "A player in a pink bib dribbles the ball down the sideline, evading a defender before passing to a teammate.",
        "player_names": [],
    },
    {
        "start_time": 81,
        "end_time": 87,
        "event_type": "SHOT_OFF_TARGET",
        "playback_status": "LIVE_GAMEPLAY",
        "match_start_time": "01:21",
        "match_end_time": "01:27",
        "description": "A player in a pink bib takes a long shot from outside the penalty area, but the ball goes over the crossbar.",
        "player_names": [],
    },
    {
        "start_time": 87,
        "end_time": 101,
        "event_type": "PASS",
        "playback_status": "LIVE_GAMEPLAY",
        "match_start_time": "01:27",
        "match_end_time": "01:41",
        "description": "Players in pink bibs pass the ball amongst themselves, trying to build an attack. A player in a black jersey intercepts a pass.",
        "player_names": [],
    },
    {
        "start_time": 101,
        "end_time": 108,
        "event_type": "FOUL",
        "playback_status": "LIVE_GAMEPLAY",
        "match_start_time": "01:41",
        "match_end_time": "01:48",
        "description": "A player in a pink bib dribbles towards the goal and is fouled by a player in a black jersey, who slides in and takes him down.",
        "player_names": [],
    },
    {
        "start_time": 108,
        "end_time": 118,
        "event_type": "PASS",
        "playback_status": "LIVE_GAMEPLAY",
        "match_start_time": "01:48",
        "match_end_time": "01:58",
        "description": "Players in black jerseys pass the ball around the midfield, looking for an opening in the defense.",
        "player_names": [],
    },
    {
        "start_time": 118,
        "end_time": 132,
        "event_type": "SHOT_OFF_TARGET",
        "playback_status": "LIVE_GAMEPLAY",
        "match_start_time": "01:58",
        "match_end_time": "02:12",
        "description": "A player in a black jersey dribbles past a defender and takes a shot, but the ball goes wide of the goal.",
        "player_names": [],
    },
    {
        "start_time": 132,
        "end_time": 148,
        "event_type": "PASS",
        "playback_status": "LIVE_GAMEPLAY",
        "match_start_time": "02:12",
        "match_end_time": "02:28",
        "description": "Players in pink bibs pass the ball around, trying to create a scoring opportunity. A player in a black jersey intercepts a pass and clears the ball.",
        "player_names": [],
    },
    {
        "start_time": 148,
        "end_time": 158,
        "event_type": "PASS",
        "playback_status": "LIVE_GAMEPLAY",
        "match_start_time": "02:28",
        "match_end_time": "02:38",
        "description": "Players in pink bibs pass the ball amongst themselves, trying to maintain possession.",
        "player_names": [],
    },
    {
        "start_time": 158,
        "end_time": 170,
        "event_type": "PASS",
        "playback_status": "LIVE_GAMEPLAY",
        "match_start_time": "02:38",
        "match_end_time": "02:50",
        "description": "The goalkeeper kicks the ball downfield. A player in a pink bib receives the ball and passes it to a teammate.",
        "player_names": [],
    },
    {
        "start_time": 170,
        "end_time": 180,
        "event_type": "PASS",
        "playback_status": "LIVE_GAMEPLAY",
        "match_start_time": "02:50",
        "match_end_time": "03:00",
        "description": "A player in a green jersey passes the ball to a teammate in a white jersey, who then dribbles down the field.",
        "player_names": [],
    },
]


def transcode(src: Path, dst: Path) -> None:
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(src),
        "-c:v",
        "libx264",
        "-profile:v",
        "high",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        "-c:a",
        "aac",
        str(dst),
    ]
    subprocess.run(cmd, check=True, capture_output=True)


def ffprobe_media_metadata(path: Path) -> dict:
    out = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,codec_name,avg_frame_rate,nb_frames:format=duration",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    info = json.loads(out)
    st = info["streams"][0]
    duration = float(info["format"]["duration"])
    num, den = st["avg_frame_rate"].split("/")
    fps = float(num) / float(den) if float(den) else 0.0
    nb = st.get("nb_frames")
    total_frames = int(nb) if nb and nb != "N/A" else int(round(duration * fps))
    return {
        "duration": round(duration, 2),
        "total_frames": total_frames,
        "height": int(st["height"]),
        "width": int(st["width"]),
        "codec_name": st["codec_name"],
        "video_fps": round(fps, 3),
    }


def build_row(media_metadata: dict, chapters: list | None = None) -> dict:
    chapters = chapters if chapters is not None else PEGASUS_H16_EVENTS
    media = [{"media_path": S3_URI, "region": REGION, "type": "video"}]
    messages = [{"content": [{"image": None, "text": None, "type": "video", "video": None}], "role": "user"}]
    sample_metadata = [
        {
            "domain": ["Sport"],
            "specific_domain": "Soccer",
            "chapters": chapters,
            "segment_dict": SEGMENT_DICT,
            "user_query_segment": [USER_QUERY],
            "metadata_schema": METADATA_SCHEMA,
            "segment_id": "H16_SOCCER_DENSE",
            "annotation_info": {
                "source": "Pegasus 1.5 output (placeholder, NOT ground truth)",
                "notion": "https://www.notion.so/twelvelabs/Pegasus-1-5-Goal-Detection-Issues-Belarus-Soccer-Footage-379cab56b71d80898ee9c7c6a83ffa17",
                "asset_id": "6a22a2fd9bbec24cdc8582c7",
                "todo": "Replace chapters with corrected GT; mark GOALs at ~00:04-07 and ~02:11.",
            },
        }
    ]
    metadata = {"media_metadata": [media_metadata], "sample_metadata": sample_metadata}
    return {"id": SAMPLE_ID, "media": media, "messages": messages, "metadata": json.dumps(metadata, ensure_ascii=False)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True, help="local path to source video.mp4")
    ap.add_argument(
        "--chapters", help="path to corrected chapters JSON (from viewer.html). Defaults to embedded Pegasus output."
    )
    ap.add_argument("--upload-s3", action="store_true")
    ap.add_argument("--push-hf", action="store_true")
    args = ap.parse_args()

    src = Path(args.video).expanduser()
    assert src.exists(), f"no such file: {src}"

    chapters = None
    if args.chapters:
        chapters = json.loads(Path(args.chapters).expanduser().read_text())
        print(f"[chapters] loaded {len(chapters)} corrected chapters from {args.chapters}")

    # No transcode: source is already H.264 / yuv420p and spec-compliant.
    print("[1/3] ffprobe (source, no transcode) ...")
    mm = ffprobe_media_metadata(src)
    print("      media_metadata:", json.dumps(mm))
    row = build_row(mm, chapters)
    n_ch = len(chapters) if chapters is not None else len(PEGASUS_H16_EVENTS)
    print("[2/3] row built. chapters:", n_ch, "| metadata bytes:", len(row["metadata"]))
    print(json.dumps({k: (v if k != "metadata" else "<json str>") for k, v in row.items()}, indent=2)[:600])

    if args.upload_s3:
        print(f"[s3] uploading -> {S3_URI}")
        subprocess.run(["s5cmd", "--profile", "training", "cp", str(src), S3_URI], check=True)
    else:
        print("[s3] SKIPPED (pass --upload-s3)")

    if args.push_hf:
        import os
        from datasets import Dataset

        print(f"[3/3] push_to_hub {REPO_ID} :: {CONFIG} (split=test)")
        ds = Dataset.from_list([row])
        ds.push_to_hub(REPO_ID, config_name=CONFIG, split="test", private=True, token=os.environ["HF_TOKEN"])
        print("done.")
    else:
        print("[3/3] HF push SKIPPED (pass --push-hf)")


if __name__ == "__main__":
    main()

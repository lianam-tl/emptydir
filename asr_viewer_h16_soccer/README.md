# ASR viewer - H16_SOCCER

Investigates whether `twelvelabs/sme_eval_v3.1_fast` / `H16_SOCCER` can be solved
from ASR alone.

Per sample card in the generated HTML:
- Question (`user_query_segment[0]`)
- GT timeline bar: all chapters on the full video timeline (click for player/team/time)
- ASR timeline bar: ASR segments within +/-60s of any GT chapter (click for transcript text)

## Run

```bash
~/.venv/bin/python build_asr_viewer.py
# -> ~/Desktop/html/asr_viewer_h16_soccer.html
```

Needs `HF_TOKEN` in `~/pegasus/.env`.

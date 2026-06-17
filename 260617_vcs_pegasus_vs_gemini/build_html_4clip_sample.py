import json

data = json.load(open('pegasus-vs-gemini.json'))

# The pegasus prompt is identical across items; pull the structured prompt from item 0.
prompt = data[0]['pegasus_prompt']

html = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>VCS eval — Pegasus vs Gemini</title>
<style>
  :root {
    --peg: #2563eb; --gem: #d97706;
    --bg: #0f1117; --panel: #171a23; --panel2: #1e222d;
    --border: #2a2f3a; --text: #e6e8ee; --muted: #9aa3b2;
  }
  * { box-sizing: border-box; }
  body { margin:0; font-family: -apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;
    background: var(--bg); color: var(--text); line-height:1.5; font-size:14px; }
  header { padding: 22px 28px; border-bottom:1px solid var(--border); background:var(--panel); }
  header h1 { margin:0 0 4px; font-size:20px; }
  header p { margin:0; color:var(--muted); font-size:13px; }
  .wrap { max-width:1500px; margin:0 auto; padding: 20px 28px 80px; }
  h2 { font-size:17px; border-left:3px solid #4b5563; padding-left:10px; margin:34px 0 14px; }
  .tag { display:inline-block; padding:1px 8px; border-radius:10px; font-size:11px; font-weight:600; }
  .peg { background:rgba(37,99,235,.18); color:#93b4ff; }
  .gem { background:rgba(217,119,6,.18); color:#f5c074; }
  .pill { background:var(--panel2); border:1px solid var(--border); border-radius:8px; padding:2px 9px; font-size:12px; color:var(--muted);}
  table { border-collapse:collapse; width:100%; margin:8px 0; font-size:13px; }
  th,td { border:1px solid var(--border); padding:7px 10px; text-align:left; vertical-align:top; }
  th { background:var(--panel2); font-weight:600; }
  td.num { text-align:right; font-variant-numeric:tabular-nums; }
  .grid2 { display:grid; grid-template-columns:1fr 1fr; gap:14px; }
  .card { background:var(--panel); border:1px solid var(--border); border-radius:10px; padding:14px 16px; }
  .card.pegc { border-top:3px solid var(--peg); }
  .card.gemc { border-top:3px solid var(--gem); }
  .card h3 { margin:0 0 10px; font-size:14px; }
  details { background:var(--panel); border:1px solid var(--border); border-radius:8px; margin:8px 0; }
  summary { cursor:pointer; padding:10px 14px; font-weight:600; }
  details > div { padding:0 14px 14px; }
  pre { background:#0b0d13; border:1px solid var(--border); border-radius:8px; padding:14px; overflow:auto;
    font-size:12px; color:#cbd3e1; white-space:pre-wrap; word-break:break-word; }
  .ent { border:1px solid var(--border); border-radius:8px; padding:8px 10px; margin:6px 0; background:var(--panel2);}
  .ent .nm { font-weight:600; }
  .ent .meta { color:var(--muted); font-size:12px; }
  .ent .ty { font-size:10px; padding:1px 6px; border-radius:6px; background:#33384a; color:#aab; margin-left:6px;}
  .timeline { position:relative; height:auto; margin:10px 0; }
  .trow { display:flex; align-items:center; gap:8px; margin:3px 0;}
  .trow .lab { width:60px; font-size:11px; color:var(--muted); text-align:right; flex:0 0 60px;}
  .bar { position:relative; height:16px; background:#0b0d13; border-radius:4px; flex:1; border:1px solid var(--border);}
  .seg { position:absolute; height:100%; border-radius:3px; opacity:.85;}
  .seg.p { background:var(--peg);} .seg.g{ background:var(--gem);}
  .legend { font-size:12px; color:var(--muted); margin:6px 0;}
  .nav { position:sticky; top:0; background:var(--panel); border-bottom:1px solid var(--border); z-index:5;
    padding:8px 28px; display:flex; gap:6px; flex-wrap:wrap;}
  .nav a { color:var(--text); text-decoration:none; font-size:12px; padding:4px 10px; border-radius:6px; background:var(--panel2); border:1px solid var(--border);}
  .nav a:hover { background:#262b38;}
  .kv { color:var(--muted);}
  .diffbox { background:var(--panel); border:1px solid var(--border); border-left:3px solid #10b981; border-radius:8px; padding:12px 16px; margin:10px 0;}
  .diffbox b { color:#6ee7b7; }
  small.mono { font-family:ui-monospace,Menlo,monospace; color:var(--muted); font-size:11px;}
</style>
</head>
<body>
<header>
  <h1>VCS Evaluation Set — <span class="peg tag">Pegasus 1.5 SME</span> vs <span class="gem tag">Gemini</span></h1>
  <p>4 chunks (10-min each) from Arcane S01E01/E02 and Game of Thrones S01E01/E02 ·
     <a href="https://linear.app/twelve-labs/issue/A-1663/absorb-vcs-evaluation-set" style="color:#93b4ff">A-1663</a></p>
</header>
<div class="nav">
  <a href="#prompt">Prompt</a>
  <a href="#diff">Key differences</a>
  <a href="#counts">Counts</a>
"""

for it in data:
    html += f'  <a href="#v-{it["label"]}">{it["label"]}</a>\n'
html += "</div>\n<div class='wrap'>\n"

# ---------- PROMPT SECTION ----------
clip = prompt['clip']
axes = prompt['params']['segment_definitions']
html += '<h2 id="prompt">1 · The prompt (what both models are asked to do)</h2>\n'
html += f"""<p>Both models receive the <b>same task</b>: densely structure a 10-minute video chunk
(<small class="mono">{clip['startSec']}–{clip['endSec']}s</small>, temperature {prompt['temperature']}, max_tokens {prompt['max_tokens']:,})
along <b>three axes</b>. Pegasus is driven by this exact structured schema; Gemini is given the equivalent task and returns its own JSON shape.</p>"""

for ax in axes:
    fields = ", ".join(f["name"] for f in ax["fields"])
    html += f"""<details><summary>Axis: <span class="pill">{ax['id']}</span> — fields: {fields}</summary>
    <div><pre>{json.dumps(ax['description'], ensure_ascii=False, indent=0)[1:-1]}</pre>"""
    html += "<table><tr><th>field</th><th>type</th><th>description</th></tr>"
    for f in ax["fields"]:
        enum = f.get("enum")
        desc = f["description"]
        if enum:
            desc += f"<br><small class='mono'>enum: {', '.join(enum)}</small>"
        html += f"<tr><td><b>{f['name']}</b></td><td>{f['type']}</td><td>{desc}</td></tr>"
    html += "</table></div></details>\n"

# ---------- KEY DIFFERENCES ----------
html += '<h2 id="diff">2 · Key differences at a glance</h2>\n'
diffs = [
    ("Output schema",
     "<b>Temporal segments.</b> Each axis is a list of <code>{start_time, end_time(float sec), metadata}</code>. Entities live on a timeline; shots reference entities as inline <code>name;type;desc</code> strings; one <code>video</code> segment carries title/description/work.",
     "<b>Flat document.</b> Top-level <code>title/description/work</code>, a typed <code>entities</code> bag (person/character/place/object/concept/thing), a separate <code>entity_relationships</code> list, and <code>shots</code> with <code>mm:ss</code> string times. No per-entity timeline."),
    ("Entity naming",
     "<b>Aggressive canonicalization.</b> Names characters via world knowledge from the first frame — calls the opening-flashback girls <i>Vi</i>, <i>Powder</i> and the man <i>Vander</i> before any name is spoken.",
     "<b>Conservative / grounded.</b> Uses descriptive labels until a name is evidenced: <i>young girl with pink hair</i>, <i>large man with beard</i>, <i>dead woman</i>. Splits <i>person</i> (actor) vs <i>character</i> for live-action (GoT)."),
    ("Entity typing",
     "Entities are <b>person/character only</b> at the axis level; places/objects/things appear only inside shot <code>entities</code> strings.",
     "Entities are <b>typed into 6 buckets</b> as first-class objects (incl. place/object/concept/thing), so it surfaces more non-person entities."),
    ("Thumbnails / grounding",
     "Emits real <b>bbox coordinates</b> + 3 thumbnails per entity <code>[time, y_min, x_min, y_max, x_max]</code> normalized to [0,1000] — spatially grounded.",
     "Thumbnail is <code>time_mmss</code> only; <b>bbox is almost always null</b> — temporally located but not spatially grounded."),
    ("Shot granularity",
     "Finer — e.g. Arcane E01: <b>78 shots</b>; tracks shot_type/camera_motion/camera_angle + inline entity list per shot.",
     "Coarser — e.g. Arcane E01: <b>37 shots</b>; adds <code>narrative_descriptor</code> + <code>footage_type</code> (e.g. 'narrative') that Pegasus lacks."),
    ("Relationships",
     "Relationships attached <b>per entity</b> as <code>dst;type;rel;desc</code> (family/friend/rival…), incl. <code>actor_portrays_character</code>.",
     "Separate top-level <code>entity_relationships</code> list (0 for Arcane E01, up to 7 for GoT E01)."),
]
html += "<table><tr><th></th><th><span class='peg tag'>Pegasus</span></th><th><span class='gem tag'>Gemini</span></th></tr>"
for d in diffs:
    html += f"<tr><td><b>{d[0]}</b></td><td>{d[1]}</td><td>{d[2]}</td></tr>"
html += "</table>\n"

# ---------- COUNTS ----------
html += '<h2 id="counts">3 · Output volume per chunk</h2>\n'
html += """<table><tr>
<th rowspan=2>chunk</th>
<th colspan=3 style="text-align:center">Pegasus segments</th>
<th colspan=3 style="text-align:center">Gemini</th></tr>
<tr><th class=num>entity</th><th class=num>shot</th><th class=num>video</th>
<th class=num>entities (typed)</th><th class=num>shots</th><th class=num>rels</th></tr>"""
for it in data:
    p = it['pegasus_result']['segments']
    g = it['gemini_result']
    gent = sum(len(v) for v in g['entities'].values())
    gbreak = "/".join(str(len(g['entities'][k])) for k in ['person_entities','character_entities','place_entities','object_entities','concept_entities','thing_entities'])
    html += f"""<tr><td><b>{it['label']}</b></td>
    <td class=num>{len(p['entity'])}</td><td class=num>{len(p['shot'])}</td><td class=num>{len(p['video'])}</td>
    <td class=num>{gent} <small class='mono'>({gbreak})</small></td><td class=num>{len(g['shots'])}</td><td class=num>{len(g['entity_relationships'])}</td></tr>"""
html += "</table><div class='legend'>Gemini typed breakdown = person/character/place/object/concept/thing.</div>\n"

# ---------- PER VIDEO ----------
def gem_entity_card(e, ty):
    al = f" · aliases: {', '.join(e['aliases'])}" if e.get('aliases') else ""
    th = e.get('thumbnail')
    tt = f" · thumb @{th['time_mmss']}" if th and th.get('time_mmss') else ""
    tags = f" <span class='kv'>[{', '.join(e.get('tags',[]))}]</span>" if e.get('tags') else ""
    return f"<div class='ent'><span class='nm'>{e['canonical_name']}</span><span class='ty'>{ty}</span>{tags}<div class='meta'>{e['description']}{al}{tt}</div></div>"

def peg_entity_card(s):
    m = s['metadata']
    rels = "<br>".join(m.get('relationships', [])) or "—"
    return (f"<div class='ent'><span class='nm'>{m['canonical_name']}</span><span class='ty'>{m['entity_type']}</span>"
            f"<span class='kv'> [{s['start_time']:.0f}–{s['end_time']:.0f}s]</span>"
            f"<div class='meta'>{m['description']}<br><span class='kv'>rel:</span> {rels}</div></div>")

def mmss_to_sec(t):
    parts = str(t).split(':')
    return int(parts[0])*60 + int(parts[1]) if len(parts)==2 else float(t)

for it in data:
    p = it['pegasus_result']['segments']
    g = it['gemini_result']
    html += f'<h2 id="v-{it["label"]}">▶ {it["label"]}</h2>\n'
    html += f"<div class='legend'><small class='mono'>{it['video_url']}</small></div>\n"

    # video-level description compare
    pv = p['video'][0]['metadata']
    html += "<div class='grid2'>"
    html += f"<div class='card pegc'><h3>Pegasus — video summary</h3><b>{pv.get('title','')}</b> <span class='kv'>(work: {pv.get('work','')})</span><p>{pv['description']}</p></div>"
    html += f"<div class='card gemc'><h3>Gemini — video summary</h3><b>{g.get('title','')}</b> <span class='kv'>(work: {g.get('work')})</span><p>{g['description']}</p></div>"
    html += "</div>\n"

    # entities compare
    html += "<div class='grid2'>"
    html += f"<div class='card pegc'><h3>Pegasus entities ({len(p['entity'])})</h3>"
    for s in p['entity']:
        html += peg_entity_card(s)
    html += "</div>"
    gent = sum(len(v) for v in g['entities'].values())
    html += f"<div class='card gemc'><h3>Gemini entities ({gent})</h3>"
    typemap = {'person_entities':'person','character_entities':'character','place_entities':'place','object_entities':'object','concept_entities':'concept','thing_entities':'thing'}
    for k, ty in typemap.items():
        for e in g['entities'][k]:
            html += gem_entity_card(e, ty)
    html += "</div></div>\n"

    # gemini relationships
    if g['entity_relationships']:
        html += "<details><summary>Gemini entity_relationships ("+str(len(g['entity_relationships']))+")</summary><div><pre>"+json.dumps(g['entity_relationships'], ensure_ascii=False, indent=2)+"</pre></div></details>\n"

    # shot timeline (entity-presence-free, just shot spans)
    dur = it['chunk_duration_sec']
    html += "<details><summary>Shot timelines — Pegasus "+str(len(p['shot']))+" vs Gemini "+str(len(g['shots']))+"</summary><div>"
    html += "<div class='legend'><span class='peg tag'>Pegasus shots</span> &nbsp; <span class='gem tag'>Gemini shots</span> &nbsp; (full chunk = "+str(dur)+"s)</div>"
    # pegasus bar
    html += "<div class='trow'><div class='lab'>Pegasus</div><div class='bar'>"
    for i,s in enumerate(p['shot']):
        l = s['start_time']/dur*100; w = (s['end_time']-s['start_time'])/dur*100
        html += f"<div class='seg p' style='left:{l:.3f}%;width:{max(w,0.2):.3f}%' title='{s['start_time']:.0f}-{s['end_time']:.0f}s'></div>"
    html += "</div></div>"
    # gemini bar
    html += "<div class='trow'><div class='lab'>Gemini</div><div class='bar'>"
    for s in g['shots']:
        st = mmss_to_sec(s['start_time']); en = mmss_to_sec(s['end_time'])
        l = st/dur*100; w = (en-st)/dur*100
        html += f"<div class='seg g' style='left:{l:.3f}%;width:{max(w,0.2):.3f}%' title='{s['start_time']}-{s['end_time']}'></div>"
    html += "</div></div>"
    # sample shot descriptions side by side (first 4)
    html += "<div class='grid2' style='margin-top:12px'>"
    html += "<div class='card pegc'><h3>Pegasus shot #1</h3>"
    s0=p['shot'][0]['metadata']
    html += f"<div class='meta'>[{p['shot'][0]['start_time']:.0f}-{p['shot'][0]['end_time']:.0f}s] · {s0['shot_type']} / {s0['camera_motion']} / {s0['camera_angle']}</div><p>{s0['description']}</p></div>"
    g0=g['shots'][0]
    html += "<div class='card gemc'><h3>Gemini shot #1</h3>"
    html += f"<div class='meta'>[{g0['start_time']}-{g0['end_time']}] · {g0['shot_type']} / {g0['camera_motion']} / {g0['camera_angle']} · footage:{g0.get('footage_type')}</div><p>{g0['description']}</p>"
    if g0.get('narrative_descriptor'): html += f"<div class='kv'>narrative: {g0['narrative_descriptor']}</div>"
    html += "</div></div>"
    html += "</div></details>\n"

# metrics
html += '<h2>4 · Pegasus generation metrics</h2><table><tr><th>chunk</th><th class=num>output chars</th><th class=num>output tokens</th><th class=num>input tokens</th><th>finish</th><th>postprocess fixes</th></tr>'
for it in data:
    m = it['pegasus_result']['metrics']; pp = it['pegasus_result']['postprocess_metrics']['totals']
    html += f"<tr><td>{it['label']}</td><td class=num>{m['total_output_chars']:,}</td><td class=num>{m['token_count']:,}</td><td class=num>{m['input_token_count']:,}</td><td>{m['finish_reason']}</td><td><small class='mono'>dedup {pp['dedup_merges']}, overlaps {pp['overlaps_fixed']}, absorbed {pp['short_segments_absorbed']}</small></td></tr>"
html += "</table>\n"

html += "</div></body></html>"

open('pegasus_vs_gemini.html','w').write(html)
print("wrote pegasus_vs_gemini.html", len(html), "bytes")

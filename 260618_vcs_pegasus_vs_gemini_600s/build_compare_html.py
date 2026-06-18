"""
Build a time-aligned, field-level comparison HTML for the VCS evaluation set
(Pegasus 1.5 SME vs Gemini), linear A-1663.

Input : pegasus-vs-gemini-allchunks.json   (downloaded from the A-1663 Linear comment)
Output: pegasus_vs_gemini_compare.html      (self-contained; data embedded)

Layout: per chunk, the 3-axis prompt (entity/shot/video) is shown, and each
axis's prompt is immediately followed by its answer. Pegasus (left) and Gemini
(right) boxes are rendered as field-name -> value tables and aligned into the
same row when their start timestamps are close (two-pointer merge). All Pegasus
times are formatted as mm:ss.
"""
import json
import sys

# Usage: python3 build_compare_html.py [SRC.json] [OUT.html]
SRC = sys.argv[1] if len(sys.argv) > 1 else "pegasus-vs-gemini-allchunks.json"
OUT = sys.argv[2] if len(sys.argv) > 2 else "pegasus_vs_gemini_compare.html"


def esc(s):
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def axis_prompt_html(ax):
    h = f"<div class=axdesc>{esc(ax['description'])}</div>"
    h += "<table class=fields><tr><th>field</th><th>type</th><th>description</th></tr>"
    for f in ax["fields"]:
        desc = esc(f["description"])
        if f.get("enum"):
            desc += f"<br><small class=mono>enum: {esc(', '.join(f['enum']))}</small>"
        h += f"<tr><td><b>{f['name']}</b></td><td>{f['type']}</td><td>{desc}</td></tr>"
    return h + "</table>"


def main():
    d = json.load(open(SRC))
    ref = d[0]["chunks"][0]["pegasus_prompt"]
    axes = {ax["id"]: ax for ax in ref["params"]["segment_definitions"]}
    AX = {k: axis_prompt_html(axes[k]) for k in ["entity", "shot", "video"]}

    slim = []
    for ep in d:
        e = {k: ep[k] for k in ["label", "video_url", "duration_sec", "num_chunks"]}
        e["chunks"] = []
        for c in ep["chunks"]:
            nc = {k: c[k] for k in ["chunk_index", "start_sec", "end_sec", "chunk_len_sec"]}
            nc["pegasus"] = {"segments": c["pegasus_result"]["segments"]}
            nc["gemini"] = c["gemini_result"]
            e["chunks"].append(nc)
        slim.append(e)
    slim_json = json.dumps(slim, ensure_ascii=False)

    html = TEMPLATE
    html = (html.replace("__DATA__", slim_json)
                .replace("__AXENTITY__", json.dumps(AX["entity"]))
                .replace("__AXSHOT__", json.dumps(AX["shot"]))
                .replace("__AXVIDEO__", json.dumps(AX["video"])))
    open(OUT, "w").write(html)
    print("wrote", OUT)


TEMPLATE = r"""<!DOCTYPE html><html lang=en><head><meta charset=utf-8>
<meta name=viewport content="width=device-width, initial-scale=1">
<title>VCS — Pegasus vs Gemini (time-aligned)</title>
<style>
:root{--peg:#2563eb;--gem:#d97706;--bg:#0f1117;--panel:#171a23;--panel2:#1e222d;--border:#2a2f3a;--text:#e8eaf0;--muted:#9aa3b2;}
*{box-sizing:border-box}body{margin:0;font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;background:var(--bg);color:var(--text);line-height:1.55;font-size:14px}
.top{position:sticky;top:0;z-index:10;background:var(--panel);border-bottom:1px solid var(--border);padding:14px 28px;display:flex;align-items:center;gap:14px;flex-wrap:wrap}
.top h1{margin:0;font-size:17px;font-weight:700}.top .sp{flex:1}
select{background:var(--panel2);color:var(--text);border:1px solid var(--border);border-radius:8px;padding:8px 12px;font-size:14px}
button{background:var(--panel2);color:var(--text);border:1px solid var(--border);border-radius:8px;padding:8px 12px;font-size:13px;cursor:pointer}button:hover{background:#262b38}
.wrap{max-width:1800px;margin:0 auto;padding:18px 28px 90px}
.tag{display:inline-block;padding:2px 10px;border-radius:10px;font-size:12px;font-weight:700}
.peg{background:rgba(37,99,235,.2);color:#9cc0ff}.gem{background:rgba(217,119,6,.2);color:#f6c777}
.pill{display:inline-block;background:var(--panel2);border:1px solid var(--border);border-radius:7px;padding:2px 9px;font-size:12px;color:#cdd4e0;font-weight:600}
.muted{color:var(--muted)}small.mono,.mono{font-family:ui-monospace,Menlo,monospace;color:var(--muted);font-size:12px}
.chunkbar{display:flex;gap:10px;flex-wrap:wrap;align-items:center;margin:4px 0 14px;color:var(--muted);font-size:13px}
.prompt{background:#11151e;border:1px solid var(--border);border-left:4px solid #6b7280;border-radius:10px;margin:26px 0 0;padding:14px 18px}
.prompt .lab{font-size:12px;font-weight:700;letter-spacing:.06em;text-transform:uppercase;color:#cbd3e1;margin-bottom:8px}
.prompt .lab .axn{color:#fff;background:#374151;border-radius:6px;padding:1px 8px;margin-left:4px}
.axdesc{font-size:13px;color:#d7dce6}
table.fields{border-collapse:collapse;width:100%;margin:12px 0 2px;font-size:12px}
table.fields th,table.fields td{border:1px solid var(--border);padding:5px 8px;text-align:left;vertical-align:top}table.fields th{background:var(--panel2)}
.answer{display:grid;grid-template-columns:1fr 1fr;gap:0;border:1px solid var(--border);border-top:none;border-radius:0 0 10px 10px;margin:0 0 8px}
.ahead{grid-column:1 / -1;display:grid;grid-template-columns:1fr 1fr;background:var(--panel)}
.ahead>div{padding:8px 16px;font-weight:700;font-size:13px}
.ahead .ph{border-left:3px solid var(--peg)}.ahead .gh{border-left:3px solid var(--gem)}
.gc{padding:8px 12px;border-top:1px solid var(--border)}
.gc.p{border-left:3px solid rgba(37,99,235,.45);border-right:1px solid var(--border)}
.gc.g{border-left:3px solid rgba(217,119,6,.45)}
.gc.empty{background:repeating-linear-gradient(45deg,#12151d,#12151d 8px,#151922 8px,#151922 16px)}
.item{border:1px solid var(--border);border-radius:8px;background:var(--panel2);overflow:hidden}
.item .ttl{padding:5px 10px;font-weight:700;font-size:13px;background:#222734;border-bottom:1px solid var(--border)}
.item .ttl .count{font-weight:500}
table.fv{border-collapse:collapse;width:100%;font-size:13px}
table.fv td{border-top:1px solid #232836;padding:5px 10px;vertical-align:top}
table.fv td.fn{width:120px;color:#aeb6c4;font-weight:600;font-family:ui-monospace,Menlo,monospace;font-size:12px;background:#1a1f2a;white-space:nowrap}
.val .arr div{padding:1px 0}
.empty{color:#5b6473}
.warn{color:#f6c777;font-weight:600}.count{font-size:12px;color:var(--muted);font-weight:500}
.timeline{grid-column:1 / -1;padding:10px 16px;border-bottom:1px solid var(--border)}
.trow{display:flex;align-items:center;gap:8px;margin:3px 0}.trow .lab{width:64px;font-size:11px;color:var(--muted);text-align:right;flex:0 0 64px}
.bar{position:relative;height:15px;background:#0b0d13;border-radius:4px;flex:1;border:1px solid var(--border)}
.seg{position:absolute;height:100%;border-radius:2px;opacity:.8}.seg.p{background:var(--peg)}.seg.g{background:var(--gem)}
</style></head><body>
<div class=top>
  <h1>VCS · <span class="peg tag">Pegasus</span> vs <span class="gem tag">Gemini</span></h1>
  <span class=muted>prompt → answer · time-aligned · mm:ss</span><span class=sp></span>
  <button id=prev>‹ prev</button><select id=epsel></select><select id=chsel></select><button id=next>next ›</button>
</div>
<div class=wrap>
<div id=chunkmeta class=chunkbar></div>
<div id=body></div>
</div>
<script>
const DATA=__DATA__;
const AX={entity:__AXENTITY__, shot:__AXSHOT__, video:__AXVIDEO__};
const THR_ENT=15, THR_SHOT=5;   // seconds: pair P & G boxes whose start times are within this
function esc(s){return String(s==null?'':s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');}
function mmss(t){if(t==null)return 0;const p=String(t).split(':');return p.length===2?(+p[0]*60+ +p[1]):+t;}
function ts(s){if(s==null||isNaN(s))return '—';s=Math.round(s);const m=Math.floor(s/60),x=s%60;return m+':'+String(x).padStart(2,'0');}
const GT={person_entities:'person',character_entities:'character',place_entities:'place',object_entities:'object',concept_entities:'concept',thing_entities:'thing'};
function gcount(g){return g&&g.entities?Object.values(g.entities).reduce((a,v)=>a+v.length,0):0;}
function fmtThumb(a){if(!a||!a.length)return '<span class=empty>[]</span>';return `<span class=mono>${ts(a[0])} · y ${(a[1]/1000).toFixed(2)}–${(a[3]/1000).toFixed(2)} · x ${(a[2]/1000).toFixed(2)}–${(a[4]/1000).toFixed(2)}</span>`;}
function fmtArr(a){if(!a||!a.length)return '<span class=empty>[]</span>';return '<div class=arr>'+a.map(x=>'<div>'+esc(x)+'</div>').join('')+'</div>';}
function fvTable(title,rows){let h=`<div class=item><div class=ttl>${title}</div><table class=fv>`;
  for(const [k,v] of rows)h+=`<tr><td class=fn>${esc(k)}</td><td class=val>${v==null||v===''?'<span class=empty>—</span>':v}</td></tr>`;
  return h+'</table></div>';}

function pegEntItem(s,i){const m=s.metadata;
  return fvTable(`entity ${i+1} <span class=count>[${ts(s.start_time)}–${ts(s.end_time)}]</span>`,[
   ['start_time',ts(s.start_time)],['end_time',ts(s.end_time)],
   ['canonical_name','<b>'+esc(m.canonical_name)+'</b>'],['entity_type',esc(m.entity_type)],
   ['description',esc(m.description)],['relationships',fmtArr(m.relationships)],
   ['best_thumbnail',fmtThumb(m.best_thumbnail)],['second_best_thumbnail',fmtThumb(m.second_best_thumbnail)],['third_best_thumbnail',fmtThumb(m.third_best_thumbnail)]]);}
function gemEntItem(e,ty,i){const th=e.thumbnail;
  const thv=th?`<span class=mono>${th.time_mmss?ts(mmss(th.time_mmss)):'—'} · bbox=${th.bbox?esc(JSON.stringify(th.bbox)):'null'}</span>`:'<span class=empty>—</span>';
  return fvTable(`entity ${i+1}${th&&th.time_mmss?' <span class=count>[~'+ts(mmss(th.time_mmss))+']</span>':''}`,[
   ['entity_type',ty],['canonical_name','<b>'+esc(e.canonical_name)+'</b>'],
   ['description',esc(e.description)],['aliases',fmtArr(e.aliases)],['tags',fmtArr(e.tags)],['thumbnail',thv]]);}
function pegShotItem(s,i){const m=s.metadata;
  return fvTable(`shot ${i+1} <span class=count>[${ts(s.start_time)}–${ts(s.end_time)}]</span>`,[
   ['start_time',ts(s.start_time)],['end_time',ts(s.end_time)],
   ['shot_type',esc(m.shot_type)],['camera_motion',esc(m.camera_motion)],['camera_angle',esc(m.camera_angle)],
   ['description',esc(m.description)],['entities',fmtArr(m.entities)]]);}
function gemShotItem(s,i){const ents=[];if(s.entities)for(const k in s.entities)for(const e of s.entities[k])ents.push(e.canonical_name+' ('+GT[k]+')');
  return fvTable(`shot ${i+1} <span class=count>[${esc(s.start_time)}–${esc(s.end_time)}]</span>`,[
   ['start_time',esc(s.start_time)],['end_time',esc(s.end_time)],
   ['shot_type',esc(s.shot_type)],['camera_motion',esc(s.camera_motion)],['camera_angle',esc(s.camera_angle)],
   ['footage_type',esc(s.footage_type)],['narrative_descriptor',esc(s.narrative_descriptor)],
   ['description',esc(s.description)],['entities',fmtArr(ents)]]);}

function alignByTime(P,G,thr){let i=0,j=0,rows=[];const v=x=>x==null?Infinity:x;
  while(i<P.length||j<G.length){
    if(i>=P.length){rows.push([null,G[j++]]);continue;}
    if(j>=G.length){rows.push([P[i++],null]);continue;}
    const a=v(P[i].t),b=v(G[j].t);
    if(Math.abs(a-b)<=thr){rows.push([P[i++],G[j++]]);}
    else if(a<b){rows.push([P[i++],null]);}
    else{rows.push([null,G[j++]]);}
  }return rows;}
function gridRows(rows){let h='';for(const [p,g] of rows){
   h+=p?`<div class="gc p">${p.html}</div>`:`<div class="gc p empty"></div>`;
   h+=g?`<div class="gc g">${g.html}</div>`:`<div class="gc g empty"></div>`;
  }return h;}

function promptBlock(n,axisid){return `<div class=prompt><div class=lab>Prompt ${n}<span class=axn>${axisid}</span> axis</div>${AX[axisid]}</div>`;}
function ahead(p,g){return `<div class=ahead><div class=ph><span class="peg tag">Pegasus</span> <span class=count>${p}</span></div><div class=gh><span class="gem tag">Gemini</span> <span class=count>${g}</span></div></div>`;}

function render(epi,chi){
 const ep=DATA[epi],c=ep.chunks[chi],p=c.pegasus.segments,g=c.gemini,dur=c.chunk_len_sec;
 document.getElementById('chunkmeta').innerHTML=
   `<span class=pill>${esc(ep.label)}</span><span>chunk #${c.chunk_index} / ${ep.num_chunks}</span><span>${ts(c.start_sec)}–${ts(c.end_sec)} (len ${ts(dur)})</span><span class=mono>${esc(ep.video_url)}</span>`;
 let h='';
 h+=promptBlock(1,'entity')+`<div class=answer>`+ahead(p.entity.length+' entities',(g?gcount(g):0)+' entities'+(g?'':' · ⚠ null'));
 const pe=p.entity.map((s,i)=>({t:s.start_time,html:pegEntItem(s,i)}));
 let ge=[];if(g){let i=0;for(const k in GT)for(const e of g.entities[k]){const th=e.thumbnail;ge.push({t:(th&&th.time_mmss)?mmss(th.time_mmss):null,html:gemEntItem(e,GT[k],i)});i++;}}
 if(g)h+=gridRows(alignByTime(pe,ge,THR_ENT));
 else h+=gridRows(pe.map(x=>[x,null]));
 h+=`</div>`;
 h+=promptBlock(2,'shot')+`<div class=answer>`+ahead(p.shot.length+' shots',(g?g.shots.length:0)+' shots'+(g?'':' · ⚠ null'));
 h+=`<div class=timeline><div class=trow><div class=lab>Pegasus</div><div class=bar>`;
 for(const s of p.shot){const l=s.start_time/dur*100,w=(s.end_time-s.start_time)/dur*100;h+=`<div class="seg p" style="left:${l.toFixed(2)}%;width:${Math.max(w,.2).toFixed(2)}%" title="${ts(s.start_time)}-${ts(s.end_time)}"></div>`;}
 h+=`</div></div><div class=trow><div class=lab>Gemini</div><div class=bar>`;
 if(g)for(const s of g.shots){const st=mmss(s.start_time),en=mmss(s.end_time),l=st/dur*100,w=(en-st)/dur*100;h+=`<div class="seg g" style="left:${l.toFixed(2)}%;width:${Math.max(w,.2).toFixed(2)}%" title="${esc(s.start_time)}-${esc(s.end_time)}"></div>`;}
 h+=`</div></div></div>`;
 const psh=p.shot.map((s,i)=>({t:s.start_time,html:pegShotItem(s,i)}));
 const gsh=g?g.shots.map((s,i)=>({t:mmss(s.start_time),html:gemShotItem(s,i)})):[];
 if(g)h+=gridRows(alignByTime(psh,gsh,THR_SHOT));
 else h+=gridRows(psh.map(x=>[x,null]));
 h+=`</div>`;
 h+=promptBlock(3,'video')+`<div class=answer>`+ahead('1 segment',g?'top-level':'⚠ null');
 const pv=p.video[0].metadata;
 h+='<div class="gc p">'+fvTable(`video 1 <span class=count>[${ts(p.video[0].start_time)}–${ts(p.video[0].end_time)}]</span>`,[['start_time',ts(p.video[0].start_time)],['end_time',ts(p.video[0].end_time)],['title','<b>'+esc(pv.title)+'</b>'],['work',esc(pv.work)],['description',esc(pv.description)]])+'</div>';
 h+= g?'<div class="gc g">'+fvTable('video',[['title','<b>'+esc(g.title)+'</b>'],['work',esc(g.work)],['description',esc(g.description)]])+'</div>'
      :'<div class="gc g empty"><span class=warn>⚠ Gemini returned null</span></div>';
 h+=`</div>`;
 document.getElementById('body').innerHTML=h;window.scrollTo(0,0);
}
const epsel=document.getElementById('epsel'),chsel=document.getElementById('chsel');
epsel.innerHTML=DATA.map((e,i)=>`<option value=${i}>${esc(e.label)}</option>`).join('');
function fillCh(){const ep=DATA[+epsel.value];chsel.innerHTML=ep.chunks.map((c,i)=>`<option value=${i}>#${c.chunk_index} · ${ts(c.start_sec)}–${ts(c.end_sec)}${c.gemini?'':' ⚠'}</option>`).join('');}
epsel.onchange=()=>{fillCh();render(+epsel.value,0);};
chsel.onchange=()=>render(+epsel.value,+chsel.value);
document.getElementById('prev').onclick=()=>{let e=+epsel.value,c=+chsel.value;if(c>0)c--;else if(e>0){e--;epsel.value=e;fillCh();c=DATA[e].chunks.length-1;}chsel.value=c;render(e,c);};
document.getElementById('next').onclick=()=>{let e=+epsel.value,c=+chsel.value;if(c<DATA[e].chunks.length-1)c++;else if(e<DATA.length-1){e++;epsel.value=e;fillCh();c=0;}chsel.value=c;render(e,c);};
fillCh();render(0,0);
</script></body></html>"""


if __name__ == "__main__":
    main()

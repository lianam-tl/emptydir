import json, statistics as stt, re

FILES={'TV 300s':'pegasus-vs-gemini-allchunks.json',
       'TV 600s':'pegasus-vs-gemini-allchunks-600.json',
       'Real-world 300s':'pegasus-vs-gemini-allchunks-realworld.json'}
GTYPES=['person_entities','character_entities','place_entities','object_entities','concept_entities','thing_entities']
GMAP={'person_entities':'person','character_entities':'character','place_entities':'place','object_entities':'object','concept_entities':'concept','thing_entities':'thing'}
GENERIC=set("man woman girl boy guard soldier person men women child children guy lady figure people enforcer worker villager crowd kid teen player referee".split())

def mmss(t):
    if t is None: return None
    p=str(t).split(':')
    try:
        if len(p)==3: return int(p[0])*3600+int(p[1])*60+float(p[2])
        if len(p)==2: return int(p[0])*60+float(p[1])
        return float(t)
    except ValueError: return None

def slim(path):
    d=json.load(open(path)); out=[]
    for ep in d:
        e={k:ep[k] for k in ['label','video_url','duration_sec','num_chunks']}; e['chunks']=[]
        for c in ep['chunks']:
            nc={k:c[k] for k in ['chunk_index','start_sec','end_sec','chunk_len_sec']}
            nc['pegasus']={'segments':c['pegasus_result']['segments']}
            nc['gemini']=c['gemini_result']
            e['chunks'].append(nc)
        out.append(e)
    return out

def is_generic(name):
    n=(name or '').strip().lower(); toks=re.findall(r"[a-z']+",n)
    if any(t in GENERIC for t in toks): return True
    if ' ' in (name or '') and not any(w[:1].isupper() for w in name.split()): return True
    return False

def summ(xs):
    xs=[x for x in xs if x is not None]
    if not xs: return {'n':0}
    s=sorted(xs)
    def pct(p):
        k=(len(s)-1)*p; f=int(k); c=min(f+1,len(s)-1); return s[f]+(s[c]-s[f])*(k-f)
    return {'n':len(xs),'mean':round(sum(xs)/len(xs),2),'median':round(stt.median(xs),2),
            'p10':round(pct(.1),2),'p90':round(pct(.9),2),'min':round(min(xs),2),'max':round(max(xs),2),'sum':round(sum(xs),2)}

def analyze(data):
    R={'n_chunks':0,'n_chunks_gem':0,'total_video_sec':0,'episodes':[]}
    P={'ent_dur':[],'ent_desc':[],'ent_rel':0,'ent_type':{},'ent_n':0,'bbox_best':0,'bbox_2':0,'bbox_3':0,'ent_generic':0,
       'shot_dur':[],'shot_desc':[],'shot_n':0,'shot_type':{},'cam_motion':{},'cam_angle':{},'ents_per_shot':[],'shot_cov':[],
       'vid_title':[],'vid_desc':[]}
    G={'ent_desc':[],'ent_type':{},'ent_n':0,'ent_alias':0,'ent_tags':0,'thumb_time':0,'thumb_bbox':0,'ent_generic':0,
       'shot_dur':[],'shot_desc':[],'shot_n':0,'shot_type':{},'cam_motion':{},'cam_angle':{},'ents_per_shot':[],'shot_cov':[],
       'vid_title':[],'vid_desc':[],'rel_n':0}
    def inc(dd,k): dd[k]=dd.get(k,0)+1
    for ep in data:
        R['total_video_sec']+=ep['duration_sec']; er={'label':ep['label'],'chunks':[]}
        for c in ep['chunks']:
            R['n_chunks']+=1; dur=c['chunk_len_sec']; seg=c['pegasus']['segments']; g=c['gemini']
            P['ent_n']+=len(seg['entity']); P['shot_n']+=len(seg['shot'])
            for s in seg['entity']:
                m=s['metadata']; P['ent_dur'].append(s['end_time']-s['start_time']); P['ent_desc'].append(len(m.get('description') or ''))
                P['ent_rel']+=len(m.get('relationships') or []); inc(P['ent_type'],m.get('entity_type','?'))
                if m.get('best_thumbnail'): P['bbox_best']+=1
                if m.get('second_best_thumbnail'): P['bbox_2']+=1
                if m.get('third_best_thumbnail'): P['bbox_3']+=1
                if is_generic(m.get('canonical_name','')): P['ent_generic']+=1
            cov=0
            for s in seg['shot']:
                m=s['metadata']; sd=s['end_time']-s['start_time']; P['shot_dur'].append(sd); cov+=sd
                P['shot_desc'].append(len(m.get('description') or '')); inc(P['shot_type'],m.get('shot_type','?'))
                inc(P['cam_motion'],m.get('camera_motion','?')); inc(P['cam_angle'],m.get('camera_angle','?'))
                P['ents_per_shot'].append(len(m.get('entities') or []))
            if dur: P['shot_cov'].append(min(cov/dur,1)*100)
            for s in seg['video']:
                m=s['metadata']; P['vid_title'].append(len(m.get('title') or '')); P['vid_desc'].append(len(m.get('description') or ''))
            ge_n=0
            if g is not None:
                R['n_chunks_gem']+=1
                for k in GTYPES:
                    for e in g['entities'][k]:
                        G['ent_n']+=1; ge_n+=1; inc(G['ent_type'],GMAP[k]); G['ent_desc'].append(len(e.get('description') or ''))
                        if e.get('aliases'): G['ent_alias']+=1
                        if e.get('tags'): G['ent_tags']+=1
                        th=e.get('thumbnail')
                        if th and th.get('time_mmss'): G['thumb_time']+=1
                        if th and th.get('bbox'): G['thumb_bbox']+=1
                        if is_generic(e.get('canonical_name','')): G['ent_generic']+=1
                gcov=0; base=c['start_sec']
                for s in g['shots']:
                    a=mmss(s['start_time']); b=mmss(s['end_time']); sd=(b-a) if (a is not None and b is not None) else None
                    if sd is not None: G['shot_dur'].append(sd); gcov+=sd
                    G['shot_n']+=1; G['shot_desc'].append(len(s.get('description') or ''))
                    inc(G['shot_type'],s.get('shot_type','?')); inc(G['cam_motion'],s.get('camera_motion','?')); inc(G['cam_angle'],s.get('camera_angle','?'))
                    ne=sum(len(s['entities'][kk]) for kk in s.get('entities',{})) if s.get('entities') else 0
                    G['ents_per_shot'].append(ne)
                if dur: G['shot_cov'].append(min(gcov/dur,1)*100)
                G['vid_title'].append(len(g.get('title') or '')); G['vid_desc'].append(len(g.get('description') or '')); G['rel_n']+=len(g.get('entity_relationships') or [])
            er['chunks'].append({'idx':c['chunk_index'],'p_shot':len(seg['shot']),'p_ent':len(seg['entity']),'g_shot':(len(g['shots']) if g else None),'g_ent':(ge_n if g else None)})
        R['episodes'].append(er)
    out={'meta':{'n_chunks':R['n_chunks'],'n_chunks_gem':R['n_chunks_gem'],'total_video_sec':round(R['total_video_sec'])},'episodes':R['episodes']}
    out['P']={'ent_n':P['ent_n'],'shot_n':P['shot_n'],'ent_rel':P['ent_rel'],'ent_dur':summ(P['ent_dur']),'shot_dur':summ(P['shot_dur']),
        'ent_desc':summ(P['ent_desc']),'shot_desc':summ(P['shot_desc']),'vid_title':summ(P['vid_title']),'vid_desc':summ(P['vid_desc']),
        'ents_per_shot':summ(P['ents_per_shot']),'shot_cov':summ(P['shot_cov']),'ent_type':P['ent_type'],'shot_type':P['shot_type'],
        'cam_motion':P['cam_motion'],'cam_angle':P['cam_angle'],'bbox_best':P['bbox_best'],'bbox_2':P['bbox_2'],'bbox_3':P['bbox_3'],
        'ent_generic':P['ent_generic'],'shot_dur_list':[round(x,2) for x in P['shot_dur']],'shot_desc_list':P['shot_desc'],'ent_desc_list':P['ent_desc']}
    out['G']={'ent_n':G['ent_n'],'shot_n':G['shot_n'],'rel_n':G['rel_n'],'shot_dur':summ(G['shot_dur']),'ent_desc':summ(G['ent_desc']),
        'shot_desc':summ(G['shot_desc']),'vid_title':summ(G['vid_title']),'vid_desc':summ(G['vid_desc']),'ents_per_shot':summ(G['ents_per_shot']),
        'shot_cov':summ(G['shot_cov']),'ent_type':G['ent_type'],'shot_type':G['shot_type'],'cam_motion':G['cam_motion'],'cam_angle':G['cam_angle'],
        'thumb_time':G['thumb_time'],'thumb_bbox':G['thumb_bbox'],'ent_alias':G['ent_alias'],'ent_tags':G['ent_tags'],'ent_generic':G['ent_generic'],
        'shot_dur_list':[round(x,2) for x in G['shot_dur']],'shot_desc_list':G['shot_desc'],'ent_desc_list':G['ent_desc']}
    return out

SETS={k:slim(v) for k,v in FILES.items()}
STATS={k:analyze(SETS[k]) for k in FILES}
ref=json.load(open(FILES['TV 600s']))[0]['chunks'][0]['pegasus_prompt']
axes={ax['id']:ax for ax in ref['params']['segment_definitions']}
def esc(s): return str(s).replace('&','&amp;').replace('<','&lt;').replace('>','&gt;')
def axhtml(ax):
    h=f"<div class=axdesc>{esc(ax['description'])}</div><table class=fields><tr><th>field</th><th>type</th><th>description</th></tr>"
    for f in ax['fields']:
        de=esc(f['description'])
        if f.get('enum'): de+=f"<br><small class=mono>enum: {esc(', '.join(f['enum']))}</small>"
        h+=f"<tr><td><b>{f['name']}</b></td><td>{f['type']}</td><td>{de}</td></tr>"
    return h+"</table>"
AX={k:axhtml(axes[k]) for k in ['entity','shot','video']}

TPL=open('combined_template.html').read()
html=(TPL.replace('__SETS__',json.dumps(SETS,ensure_ascii=False))
        .replace('__STATS__',json.dumps(STATS,ensure_ascii=False))
        .replace('__AXENTITY__',json.dumps(AX['entity']))
        .replace('__AXSHOT__',json.dumps(AX['shot']))
        .replace('__AXVIDEO__',json.dumps(AX['video'])))
open('vcs_combined.html','w').write(html)
import os;print('wrote vcs_combined.html',round(os.path.getsize('vcs_combined.html')/1e6,2),'MB')

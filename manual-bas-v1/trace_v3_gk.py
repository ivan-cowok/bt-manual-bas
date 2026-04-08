"""Trace GK pass flight in video 3, frames 85-180."""
import json, math

with open('data/3/output.json') as f:
    data = json.load(f)
frames_raw = {fd['frame_number']: fd for fd in data['frames']}
fw = data['video_resolution'][0]
PROX = 0.5

def feet_dist(bc, bbox):
    fx=(bbox[0]+bbox[2])/2.0; fy=bbox[3]
    return math.sqrt((bc[0]-fx)**2+(bc[1]-fy)**2)

def bbox_dist(bc, bbox):
    x1,y1,x2,y2=bbox
    cx=max(x1,min(bc[0],x2)); cy=max(y1,min(bc[1],y2))
    return math.sqrt((bc[0]-cx)**2+(bc[1]-cy)**2)

prev_bc=None; prev_fn=None
for fn in range(83, 180):
    fd = frames_raw.get(fn)
    if not fd:
        print(f"f{fn:4d}: MISSING"); prev_bc=None; prev_fn=None; continue
    objs = fd.get('objects',[])
    ball = next((o for o in objs if o['role']=='ball'), None)
    if ball:
        b=ball['bbox']; bc=((b[0]+b[2])/2.0,(b[1]+b[3])/2.0)
        spd=0.0
        if prev_bc and prev_fn:
            spd=math.sqrt((bc[0]-prev_bc[0])**2+(bc[1]-prev_bc[1])**2)/(fn-prev_fn)
        prev_bc=bc; prev_fn=fn
    else:
        bc=None; prev_bc=None; prev_fn=None

    poss_list=[]
    for p in objs:
        if p['role'] not in ('player','goalkeeper'): continue
        if p['detection_confidence']<0.70: continue
        if p['role']=='goalkeeper':
            cx=(p['bbox'][0]+p['bbox'][2])/2.0
            team='left' if cx<fw/2 else 'right'
        else:
            team=p.get('team_name','')
        if team not in ('left','right'): continue
        if not bc: continue
        d=bbox_dist(bc,p['bbox']) if p['role']=='goalkeeper' else feet_dist(bc,p['bbox'])
        h=p['bbox'][3]-p['bbox'][1]; thresh=PROX*h
        role_str='GK' if p['role']=='goalkeeper' else 'pl'
        if d<=thresh:
            poss_list.append(f"tid={p['track_id']}({team},{role_str}) d={d:.0f}<{thresh:.0f}")
        elif d<thresh*2:
            poss_list.append(f"tid={p['track_id']}({team},{role_str}) d={d:.0f}/near{thresh:.0f}")

    ball_str=f"({bc[0]:.0f},{bc[1]:.0f}) spd={spd:.1f}" if bc else "ABSENT"
    poss_str=', '.join(poss_list) if poss_list else 'LOOSE'
    print(f"f{fn:4d} {poss_str:<55} ball={ball_str}")

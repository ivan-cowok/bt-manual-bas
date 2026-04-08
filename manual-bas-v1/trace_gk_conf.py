"""Check GK detection confidence around save frames in video 3."""
import json, math

with open('data/3/output.json') as f:
    data = json.load(f)
frames_raw = {fd['frame_number']: fd for fd in data['frames']}
fw = data['video_resolution'][0]

for fn in [512,513,514,515,516,517,518,519,520,521,522,560,561,562,563,564]:
    fd = frames_raw.get(fn)
    if not fd: print(f"f{fn}: MISSING"); continue
    objs = fd.get('objects', [])
    ball = next((o for o in objs if o['role']=='ball'), None)
    bc_str = str([round((ball['bbox'][0]+ball['bbox'][2])/2), round((ball['bbox'][1]+ball['bbox'][3])/2)]) if ball else "ABSENT"
    for p in objs:
        if p['role'] == 'goalkeeper':
            cx = (p['bbox'][0]+p['bbox'][2])/2.0
            team = 'left' if cx < fw/2 else 'right'
            if ball:
                b = ball['bbox']; bc = ((b[0]+b[2])/2.0, (b[1]+b[3])/2.0)
                x1,y1,x2,y2 = p['bbox']
                dcx = max(x1, min(bc[0], x2)); dcy = max(y1, min(bc[1], y2))
                d = math.sqrt((bc[0]-dcx)**2 + (bc[1]-dcy)**2)
            else:
                d = -1
            print(f"f{fn}: GK tid={p['track_id']} team={team} conf={p['detection_confidence']:.2f} bbox={p['bbox']} ball={bc_str} d={d:.0f}")

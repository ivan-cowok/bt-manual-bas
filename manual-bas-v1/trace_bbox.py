import json, math
with open('data/4/output.json') as f:
    data = json.load(f)
frames = {fd['frame_number']: fd for fd in data['frames']}
for fn in range(420, 432):
    fd = frames.get(fn)
    if not fd: continue
    objs = fd.get('objects', [])
    ball = next((o for o in objs if o['role']=='ball'), None)
    p17  = next((o for o in objs if o.get('track_id')==17), None)
    if not ball or not p17: continue
    b = ball['bbox']; p = p17['bbox']
    bc = ((b[0]+b[2])/2.0, (b[1]+b[3])/2.0)
    feet = ((p[0]+p[2])/2.0, p[3])
    h = p[3] - p[1]
    thresh = 0.5 * h
    d = math.sqrt((bc[0]-feet[0])**2 + (bc[1]-feet[1])**2)
    status = "POSS" if d < thresh else "LOOSE"
    print(
        f"f{fn}  ball_bbox={b}  p17_bbox={p}\n"
        f"      ball_center=({bc[0]:.0f},{bc[1]:.0f})  "
        f"p17_feet=({feet[0]:.0f},{feet[1]:.0f})  "
        f"h={h}  thresh={thresh:.0f}  dist={d:.1f}  --> {status}\n"
    )

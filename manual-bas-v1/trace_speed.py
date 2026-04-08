import json, math
with open('data/4/output.json') as f:
    data = json.load(f)
frames = {fd['frame_number']: fd for fd in data['frames']}

print("Frame-by-frame ball speed (430-445) and why flight was suppressed\n")
print(f"  pass_min_peak_flight_speed = 12.0")
print(f"  ball_velocity_threshold    =  5.0  (min speed to enter IN_FLIGHT)")
print(f"  pass_min_flight_frames     =  4    (min consecutive flight frames)\n")

prev_bc = None
peak = 0.0
flight_speeds = []
for fn in range(425, 446):
    fd = frames.get(fn)
    if not fd:
        prev_bc = None
        print(f"  f{fn}: ABSENT")
        continue
    ball = next((o for o in fd['objects'] if o['role'] == 'ball'), None)
    if not ball:
        prev_bc = None
        print(f"  f{fn}: ABSENT")
        continue
    b = ball['bbox']
    bc = ((b[0]+b[2])/2.0, (b[1]+b[3])/2.0)
    if prev_bc:
        dx = bc[0] - prev_bc[0]; dy = bc[1] - prev_bc[1]
        spd = math.sqrt(dx*dx + dy*dy)
        note = ""
        if fn >= 430:
            flight_speeds.append(spd)
            peak = max(peak, spd)
            note = f"  <-- flight frame #{len(flight_speeds)}"
        print(f"  f{fn}: ball=({bc[0]:.0f},{bc[1]:.0f})  speed={spd:.2f} px/frame{note}")
    else:
        print(f"  f{fn}: ball=({bc[0]:.0f},{bc[1]:.0f})  speed=n/a (first frame)")
    prev_bc = bc

print(f"\nFlight frames 430-436: speeds = {[round(s,2) for s in flight_speeds]}")
print(f"Peak flight speed = {peak:.2f}  vs  threshold = 12.0")
print(f"--> {'PASS fires' if peak >= 12.0 else 'SUPPRESSED (peak < 12.0 => treated as rolling ball)'}")

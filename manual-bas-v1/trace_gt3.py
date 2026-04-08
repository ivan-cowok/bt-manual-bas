import json
with open('data/3/ground_truth.json') as f:
    gt = json.load(f)
for e in gt:
    fr = round(e['chunk_time_ms'] * 25 / 1000)
    if 460 <= fr <= 580:
        print(f"  frame={fr:4d}  ms={e['chunk_time_ms']:6d}  type={e['type']}")

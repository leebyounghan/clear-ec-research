import json
for t in ('t3', 't2', 'p'):
    J = json.load(open(f'/content/t4test/out/t4c6_{t}_fp16b1.json'))['jobs'][0]
    print(t, J['errors'], sorted((J.get('stage_s') or {}).items(), key=lambda kv: -kv[1])[:6], round((J.get('hooks') or {}).get('read_s', 0), 1))

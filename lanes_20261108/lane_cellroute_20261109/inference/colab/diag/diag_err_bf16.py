import json
J = json.load(open('/content/t4test/out/t4c6_t1_bf16.json'))['jobs'][0]; print(J['errors'], J.get('stage_s'))
for m in ('fp32', 'mix'):
    J = json.load(open(f'/content/t4test/out/t4c6_t1_{m}.json'))['jobs'][0]; print(m, J['rc'], J['wall_s'], (J.get('stderr_tail') or '')[-200:])

import json
J = json.load(open('/content/t4test/out/t4c4_default.json'))['jobs'][0]
print(J['errors']); print(J.get('stage_s')); print((J.get('hooks') or {}))

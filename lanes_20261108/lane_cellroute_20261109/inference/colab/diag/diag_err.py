import json, glob, subprocess
for f in sorted(glob.glob('/content/t4test/out/t4c2_*.json')):
    J = json.load(open(f))
    print(f, J.get('env', {}).get('CLEAR_EC_PREC'), [(j['frame'], j['entry_mode'], j['errors']) for j in J['jobs']][:2])
print(subprocess.run('tail -c 1500 /content/t4test/run2.log', shell=True, capture_output=True, text=True).stdout)

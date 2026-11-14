p = 'source/cv_stack5_20261108.py'; s = open(p).read()
old = "def cmix_comp(track='raw'):\n"
assert s.count(old) == 1
s = s.replace(old, "def cmix_comp(track='raw', prefix='cmix'):\n")
s = s.replace("        run = X / f'lane_backbone_20261108/runs/cmix_f{f}_dinov3b16_s46/{track}/runs/fold{f}/dinov3_b16'", "        run = X / f'lane_backbone_20261108/runs/{prefix}_f{f}_dinov3b16_s46/{track}/runs/fold{f}/dinov3_b16'")
s = s.replace("    elif s.startswith('cmix'): c = cmix_comp(s.split(':')[1] if ':' in s else 'raw')\n", "    elif s.startswith('cmix'): c = cmix_comp(s.split(':')[1] if ':' in s else 'raw')\n    elif s.startswith('bbrun:'): c = cmix_comp(s.split(':')[2] if s.count(':') > 1 else 'raw', s.split(':')[1])   # bbrun:<prefix>[:<track>], backbone lane runs/<prefix>_f{f}_dinov3b16_s46\n")
open(p, 'w').write(s); print('ok', s.count("bbrun:"))

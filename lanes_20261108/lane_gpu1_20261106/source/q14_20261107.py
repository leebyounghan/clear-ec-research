from pathlib import Path
L = Path('/workspace/clear-ec-research/experiments/lane_gpu1_20261106')
RT = '/workspace/clear-ec-research/experiments/phase2_foundation_20261002/runtime'
job = f"cpsam_c512_f4_s46\tPYTHONPATH={RT} /workspace/venv/bin/python experiments/lane_gpu3_20261106/source/family_recipe_20261106.py all --family cpsam_c512 --fold 4 --seed 46 --root experiments/lane_gpu3_20261106/runs/cpsam_c512_f4_s46"
assert not Path('/workspace/clear-ec-research/experiments/lane_gpu3_20261106/runs/cpsam_c512_f4_s46').exists()
q = (L/'queue.tsv').read_text().splitlines()
names = [r.split('\t')[0] for r in q]
assert 'cpsam_c512_f4_s46' not in names
q.insert(names.index('dinov2_b14_f0_s46')+1, job)
t = L/'queue.tmp'; t.write_text('\n'.join(q)+'\n'); t.replace(L/'queue.tsv')
d = {r.split('\t')[0] for r in (L/'done.tsv').read_text().splitlines()}
print('pending', [r.split('\t')[0] for r in q if r.split('\t')[0] not in d])

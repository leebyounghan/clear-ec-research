"""Per-stage T4 seconds of tier p from the Colab real-T4 logs (deploy lane, 2026-11-14 02:30 EST, coordinator 02:16). Reads only the three Colab
result files already on the pod (out/colab_t4c6_p_fp16b1.json = VM 1, out/colab_b/t4b{2,3}_p_fp16b1.json = VM 2) and the p model file sizes."""
import json, statistics as st, sys
D = sys.argv[1]
jobs = [('VM1', j) for j in json.load(open(D + 'colab_t4c6_p_fp16b1.json'))['jobs']] + \
       [('VM2', j) for f in ('t4b2_p_fp16b1.json', 't4b3_p_fp16b1.json') for j in json.load(open(D + f))['jobs']]
assert len(jobs) == 5 and all(j['rc'] == 0 and j['entry_mode'] == 'recipe' and j['gpu'] == 'Tesla T4' for _, j in jobs)
MB = {l.split('\t')[1].strip(): int(l.split('\t')[0]) for l in open(D + 'model_p_files.tsv')}
s47 = sum(v for k, v in MB.items() if 'dinov3_b16_f99_s47/' in k); s46b = sum(v for k, v in MB.items() if 'dinov3_b16_f99_s46b/' in k)
vl = [v for k, v in MB.items() if 'vitl16_f99_s4' in k]
rows = []
def add(name, f): rows.append((name, [round(f(j), 2) for _, j in jobs]))
g = lambda k: (lambda j: j['stage_s'].get(k, 0.0))
add('P1 crop members (5 x central 512 on cpsam_v2) [p1]', g('p1'))
add('CPSAM segmentation [seg]', g('seg'))
add('recipe A s46+s47 TTA (2 x 4 ViT-B/8 fwd) [recipeA_tta]', g('recipeA_tta'))
add('  of which one seed (est. = half)', lambda j: j['stage_s']['recipeA_tta'] / 2)
add('DINOv3-B dino_cd: s46b + s47 (2 x 5 ep x 4 flips = 40 fwd) [dino_cd]', g('dino_cd'))
add('  DINOv3-B s46b CD half, 20 fwd (est. = half)', lambda j: j['stage_s']['dino_cd'] / 2)
add('  DINOv3-B s47 CD half, 20 fwd (est. = half)', lambda j: j['stage_s']['dino_cd'] / 2)
add('DINOv3-B s46b CV read (5 single fwd, rereads the 5 files) [dino_cv]', g('dino_cv'))
add('ViT-L s46+s47 TTA (2 x 4 fwd) [vitl_tta_only]', g('vitl_tta_only'))
add('  ViT-L one seed (est. = half)', lambda j: j['stage_s']['vitl_tta_only'] / 2)
add('DINOv3-B cvw3 HEX (10 single fwd) [family_mean]', g('family_mean:dinov3_b16'))
add('DINOv3-B attention-pool HEX (5 single fwd) [attnpool_hex]', g('attnpool_hex'))
add('3 pre-only detectors, TTA [peaks]', g('peaks'))
add('13 other detectors, single view [other_peaks]', g('other_peaks'))
add('hex6 corrector (10 detectors; 6 also run in peaks / other_peaks) [hex6]', g('hex6'))
add('HEX features + HEX graph nets [hex_features]', g('hex_features'))
add('CV graph/sim nets + frozen heads (+1 recipe A s46 token pass) [cv_parts]', g('cv_parts'))
add('cell-route features [features]', g('features'))
add('model unload between stages, ~14 calls [_stage_end]', g('_stage_end'))
add('preprocess', g('preprocess'))
add('setup: imports + Recipe init [setup_s]', lambda j: j['setup_s'])
add('rest of the entry: image read, CD/HEX/CV combine, write', lambda j: j['entry_seconds'] - j['setup_s'] - sum(j['stage_s'].values()))
add('process start/exit outside the entry', lambda j: j['wall_s'] - j['entry_seconds'])
add('WALL per job', lambda j: j['wall_s'])
add('model loading = torch.load/safetensors read time (inside the stages)', lambda j: j['hooks']['read_s'])
add('  read throughput MB/s', lambda j: j['hooks']['read_bytes'] / j['hooks']['read_s'] / 1e6)
tops = [n for n, _ in rows if not n.startswith(' ') and not n.startswith(('WALL', 'model loading'))]
for (vm, j), w in zip(jobs, [r for r in rows if r[0] == 'WALL per job'][0][1]):
    tot = sum(v[[x for x, _ in jobs].index(vm) if False else 0] for n, v in rows if False)
chk = [round(sum(v[i] for n, v in rows if n in tops) - jobs[i][1]['wall_s'], 2) for i in range(5)]
print('check: sum of top-level rows - wall per job', chk)
hdr = ' | '.join(lab for lab in ('VM1 f0', 'VM1 f447', 'VM2 run b2 f0', 'VM2 run b3 f0', 'VM2 run b3 f447'))
print(f'| stage | median | {hdr} |'); print('|' + '---|' * 7)
for n, v in rows: print(f'| {n} | {st.median(v):.1f} | ' + ' | '.join(f'{x:.1f}' for x in v) + ' |')
# savings of the CD-only pieces, per job, with load time from the job's own read throughput
print()
for i, (vm, j) in enumerate(jobs):
    tp = j['hooks']['read_bytes'] / j['hooks']['read_s']; s = j['stage_s']
    rest_cv = s['dino_cv'] - s46b / tp           # build + 5 single fwd + state-dict/GPU copies of the CV read
    print(f'{vm} f{j["frame"]}: s47 half {s["dino_cd"]/2:.1f} (>= {s["dino_cd"]/2 - rest_cv/2:.1f}; its read {s47/tp:.1f}) | '
          f'one ViT-L {s["vitl_tta_only"]/2:.1f} (read {vl[0]/tp:.1f}) | both ViT-L {s["vitl_tta_only"] + s["_stage_end"]/14:.1f} | '
          f'all three {s["dino_cd"]/2 + s["vitl_tta_only"] + s["_stage_end"]/14:.1f}')
print('bytes: s47 half', s47 / 1e6, 'MB; s46b', s46b / 1e6, 'MB; ViT-L final.pt', [v / 1e6 for v in vl])

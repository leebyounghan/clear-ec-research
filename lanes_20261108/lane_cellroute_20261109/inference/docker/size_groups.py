"""Per member-group size of the deploy census (cell-route lane, 2026-11-11). os.stat only (no du). Groups come from manifests/deploy.json; files in
docker/files.txt that no group claims are listed under 'other'. A file claimed by several groups is counted in each and flagged as shared; the
total is over unique files. Missing CPSAM f99 epochs are sized as epoch_08. usage: size_groups.py  -> docker/size_groups.json"""
import os, json, glob, collections
ROOT = '/workspace/clear-ec-research/'; os.chdir(ROOT); H = 'experiments/lane_cellroute_20261109/inference/'
m = json.load(open(H + 'manifests/deploy.json')); census = [l.strip() for l in open(H + 'docker/files.txt') if l.strip()]
def under(d): d = d.rstrip('/') + '/'; return [f for f in census if f.startswith(d)]
def flat(v):
    if isinstance(v, str): return [v]
    if isinstance(v, list): return [q for u in v for q in flat(u)]
    if isinstance(v, dict): return [q for u in v.values() for q in flat(u)]
    return []
ALL16_PRE = 3
G = collections.OrderedDict()
G['P1 (5 Phase-1 members + their CPSAM backbone)'] = (m['p1_members'] + [m['p1_cpsam']], 'CD, CV, HEX')
ra = sorted(set(flat(m['recipeA_tta']) + flat(m['recipeA_cd']) + [m['cv_parts']['token_encoder']]))
G['recipe A s46+s47, epoch 20 (TTA4)'] = ([f for f in ra if f.endswith('epoch_20.pt')], 'CV, HEX (A); CD (A_cd half); CV frozen heads + 4 HEX frozen detector heads (s46 encoder)')
G['recipe A s46+s47, epochs 8-19 (single view)'] = ([f for f in ra if not f.endswith('epoch_20.pt')], 'CD (A_cd half)')
G['DINOv3-B s46b epochs 16-20 (D, TTA4; also DINO CV seed 1)'] = (m['dino_cd'], 'CD; CV')
G['DINOv3-B CV, other 2 seeds epochs 16-20'] = ([f for f in flat(m['dino_cv']) if f not in m['dino_cd']], 'CV')
G['ViT-L s46+s47, epochs 8-11 + final'] = (flat(m['vitl_seeds']), 'CD')
G['order-DINOv3-B epochs 16-20'] = (m['orderdino_cv'], 'CV')
G['cvw3 DINOv3-B epochs 8-12'] = (m['cvw3'], 'HEX')
G['CPSAM 512 (cps5) epochs 8-12'] = (m['cps5'], 'HEX')
G['CPSAM segmentation (cpsam_v2)'] = ([m['cpsam_seg']], 'CD (mask), HEX (seg features, cell-route points)')
G['pre-only detectors s46/47/48'] = (m['preonly_detectors'], 'HEX (cell-route, prior, ridge, graph nets), CV (graph, quartets, sim ridge)')
d16 = list(m['detectors_all16'].values())
G['9 full detectors (all16)'] = ([p for k, p in d16 if k == 'full'], 'HEX (all16 features; 5 shared with hex6)')
G['4 frozen detector heads (all16)'] = ([p for k, p in d16 if k == 'frozen'], 'HEX (all16 features)')
G['hex6 corrector (green-track deploy detectors)'] = ([f for f in under(m['hex6_deploy_dir'])], 'HEX (f6, have)')
G['HEX graph nets'] = (under(m['hex_graph_models']), 'HEX (g_s10x3, g_hex46)')
G['CV graph nets (GRAPH, quartets)'] = (under(m['cv_parts']['graph_models']), 'CV (graph, quartets); HEX (graph, best, g_s10x3)')
G['CV sim nets (sim2_l8)'] = (under('experiments/meta_ceiling_20261103/sweep/sim_pred'), 'CV (ridge_cv), HEX (best)')
G['CV frozen heads'] = (under(m['cv_parts']['frozen_heads']), 'CV (fz_*)')
G['base weights read at model build (overwritten by strict loads?)'] = (under('experiments/phase2_backbone_screen_20261001/hf_cache') + under('experiments/phase2_foundation_20261002/weights'), 'see attribution run')
size = {}
def sz(f):
    if f not in size:
        if os.path.isfile(f): size[f] = os.stat(f).st_size
        elif 'mtcps_c512_f99' in f: size[f] = os.stat(os.path.dirname(f) + '/epoch_08.pt').st_size
        else: size[f] = 0
    return size[f]
claim = collections.defaultdict(list)
for g, (fs, _) in G.items():
    for f in fs: claim[f].append(g)
other = [f for f in census if f not in claim]
out = []
for g, (fs, col) in G.items():
    fs = sorted(set(fs)); shared = [f for f in fs if len(claim[f]) > 1]
    out.append(dict(group=g, files=len(fs), gb=round(sum(sz(f) for f in fs) / 1e9, 3), shared_files=len(shared), shared_with=sorted({h for f in shared for h in claim[f] if h != g}), feeds=col))
oth = collections.Counter(); ob = collections.Counter()
for f in other: k = f.endswith('.py') and 'code (.py)' or f.endswith('.json') and 'params (.json)' or os.path.dirname(f); oth[k] += 1; ob[k] += sz(f)
uni = sorted(set(census) | set(claim))
res = dict(groups=out, other=[dict(what=k, files=oth[k], gb=round(ob[k] / 1e9, 3)) for k in sorted(oth, key=lambda k: -ob[k])],
           total=dict(files=len(uni), gb=round(sum(sz(f) for f in uni) / 1e9, 3)))
json.dump(res, open(H + 'docker/size_groups.json', 'w'), indent=1)
for r in out: print('%7.2f GB %4d files  %-62s %s%s' % (r['gb'], r['files'], r['group'][:62], r['feeds'][:60], ('  [shared %d: %s]' % (r['shared_files'], '; '.join(s[:25] for s in r['shared_with'])) if r['shared_files'] else '')))
for r in res['other'][:12]: print('   other %6.3f GB %4d %s' % (r['gb'], r['files'], r['what']))
print('TOTAL', res['total'])

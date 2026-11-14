"""Tier p3 / p3b generator (deploy lane, 2026-11-14, coordinator 09:16 EST). Builds package_20261114/<VARIANT>/ next to p2 from a copy of the p2
context (p2 itself is only read): the ViT-L seeds named in the swaps point at the encoder-LR-2e-5 full-data final.pt, stored like every other
regression member as the fp16 copy (docker/fp16_files.txt gets the new paths; the copies live in lane_cellroute_20261109/deploy_fp16/<path>), and the
CD constants are the 'deploy' block of the CD lane's freeze_cd_params_<VARIANT>_20261114.json (md5 checked). p3 = s46 swapped; p3b = s46 and s47
swapped (same call with two swaps and the p3b constants file). Writes, under package_20261114/<VARIANT>/:
  context/          the p2 context with manifests/deploy_<V>.json, params/deploy_<V>.json, model_files.txt (old ViT-L line -> new), fp16_files.txt
                    (+ new paths), Dockerfile header, MANIFEST.json (tier, tag, swap record); build with --build-arg TIER=<V>
  model_add/<path>  the fp16 copy of each new weight file (what goes into the model archive at <path>), sha256-checked against deploy_fp16
  model_swap.tsv    old_path <tab> new_path <tab> sha256 of the new fp16 file: the only change to the tier-p model tree
  box/              build_test_p3_20261114.sh (the variant is read from the folder name) for the PI's go
  laptop/           rebuild_model_archive_20261114.py (old tier-p archive + model_add -> new archive, contents checked against the box tree)
Refuses an existing <VARIANT> folder; nothing is overwritten. usage: make_p3_20261114.py VARIANT CD_PARAMS_JSON CD_MD5 SEED=NEWPATH [SEED=NEWPATH]
  e.g. make_p3_20261114.py p3 experiments/lane_cd_20261108/reads/freeze_cd_params_p3_20261114.json <md5> \
       46=experiments/lane_cd_20261108/arms/vitlfu_elr2e5_f99_s46/raw/runs/fold-1/dinov3_l16t/final.pt"""
import sys, os, json, shutil, hashlib, re
ROOT = '/workspace/clear-ec-research/'; L = ROOT + 'experiments/lane_cellroute_20261109/'; PK = L + 'package_20261114/'
V, CDF, CDMD5 = sys.argv[1], sys.argv[2], sys.argv[3]
SW = dict(a.split('=', 1) for a in sys.argv[4:]); assert SW and set(SW) <= {'46', '47'}, SW
assert re.fullmatch(r'p3b?', V), V
SRC = PK + 'p2/'; DST = os.environ.get('P3_DST') or PK + V + '/'; HERE = os.path.dirname(os.path.abspath(__file__))
DRY = os.environ.get('P3_DRY') == '1'   # test of the edit logic only (P3_DST elsewhere): no input checks on the new weights, no model_add copy
assert not DRY or (os.environ.get('P3_DST') and not os.path.abspath(DST).startswith(PK)), 'a dry run writes outside package_20261114 only'
INF = 'context/code/experiments/lane_cellroute_20261109/inference/'
if os.path.exists(DST): sys.exit(f'refusing: {DST} exists')


def md5(p): return hashlib.md5(open(p, 'rb').read()).hexdigest()


def sha256(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 24), b''): h.update(b)
    return h.hexdigest()


# 1 inputs: CD constants file (md5, tier, model_files consistent with the swaps), new weights (fp32 original + verified fp16 copy)
assert md5(ROOT + CDF) == CDMD5, ('CD params md5', md5(ROOT + CDF), CDMD5)
cdj = json.load(open(ROOT + CDF)); dep = cdj['deploy']
p2par = json.load(open(SRC + INF + 'params/deploy_p2.json')); assert set(dep) == set(p2par['cd']), (sorted(dep), sorted(p2par['cd']))
cd_paths = json.dumps(cdj.get('model_files', ''))
for s, new in SW.items(): assert DRY or new in cd_paths, f'CD params model_files do not name {new}'
conv = json.load(open(HERE + '/weights_check_out.json')) if os.path.exists(HERE + '/weights_check_out.json') else {}
for s, new in ({} if DRY else SW).items():
    assert os.path.isfile(ROOT + new), new
    assert os.path.isfile(L + 'deploy_fp16/' + new), 'no fp16 copy (run p3_weights_20261114.py first): ' + new
    assert conv.get(new, {}).get('fp16', {}).get('n_mismatch') == 0, 'fp16 copy not verified: ' + new

# 2 copy the p2 context (read only)
shutil.copytree(SRC + 'context', DST + 'context', symlinks=True)
man = json.load(open(SRC + INF + 'manifests/deploy_p2.json')); mf = open(SRC + 'context/model_files.txt').read().splitlines()
swaps = []
for s, new in sorted(SW.items()):
    hit = [e for e in man['vitl_seeds'] if f'_s{s}/' in e['tta']]; assert len(hit) == 1, (s, man['vitl_seeds'])
    old = hit[0]['tta']; assert mf.count(old) == 1 and new not in mf, (old, new)
    hit[0]['tta'] = new; mf[mf.index(old)] = new
    swaps.append((old, new))
cd_note = (f'CD constants {CDF} (md5 {CDMD5}): the deploy block; strict five-fold CD {cdj.get("strict_five_fold_cd")}, '
           f'delta vs tier p {cdj.get("delta_vs_tier_p")}, vs p2 {cdj.get("delta_vs_p2", "n/a")}')
man['decisions'] = man['decisions'] + [f'D-COORD-P3-PREP tier {V} (coordinator 09:16 EST 11-14): p2 with the ViT-L '
                                       + ' and '.join(f's{s}' for s in sorted(SW)) + ' half of V = the encoder-LR-2e-5 full-data run (raw final.pt, fp16 copy); ' + cd_note]
man['vitl_note'] = f'tier {V}: ' + '; '.join(f'{o} -> {n}' for o, n in swaps)
json.dump(man, open(DST + INF + f'manifests/deploy_{V}.json', 'w'), indent=1)
par = dict(p2par); par['cd'] = dep
par['cd_note'] = (par.get('cd_note', '') + f' | tier {V}: ' + cd_note).strip(' |')
par['decisions'] = list(par.get('decisions', [])) + [f'D-COORD-P3-PREP tier {V}: cd = deploy block of {CDF} (md5 {CDMD5})']
json.dump(par, open(DST + INF + f'params/deploy_{V}.json', 'w'), indent=1)
open(DST + 'context/model_files.txt', 'w').write('\n'.join(mf) + '\n')
fl = DST + INF + 'docker/fp16_files.txt'; fpl = open(fl).read().splitlines()
open(fl, 'w').write('\n'.join(fpl + [n for _, n in swaps if n not in fpl]) + '\n')

# 3 Dockerfile header and MANIFEST.json
df = open(DST + 'context/Dockerfile').read()
h2 = df[:df.index('# 11-12 freeze candidate image')]
assert h2.startswith('# Tier p2 (deploy lane'), h2[:80]
h3 = (f'# Tier {V} (deploy lane, 2026-11-14, D-COORD-P3-PREP): the p2 context (package_20261114/p2) with the ViT-L '
      + ' and '.join(f's{s}' for s in sorted(SW)) + ' half of V replaced by\n# the encoder-LR-2e-5 full-data run (fp16 copy, model_swap.tsv) and the CD constants of '
      + os.path.basename(CDF) + f';\n# build with --build-arg TIER={V}, tag clear-ec:<build date>-freeze-9687076-{V} (never latest). Model archive = the tier-p archive with the\n'
      + f'# files of model_swap.tsv swapped (rebuild_model_archive_20261114.py). Everything else is the p2 image:\n' + h2)
open(DST + 'context/Dockerfile', 'w').write(h3 + df[len(h2):])
mj = json.load(open(SRC + 'context/MANIFEST.json'))
mj['tier'] = V; mj['tag'] = f'clear-ec:<build date>-freeze-9687076-{V}'
mj[V] = dict(decision='D-COORD-P3-PREP', base='package_20261114/p2/context (tier p2)',
             added=[INF.replace('context/', '') + f'manifests/deploy_{V}.json', INF.replace('context/', '') + f'params/deploy_{V}.json'],
             changed=['model_files.txt', INF.replace('context/', '') + 'docker/fp16_files.txt', 'Dockerfile (header)', 'MANIFEST.json'],
             model_files_swapped=[dict(old=o, new=n) for o, n in swaps], cd_params=dict(path=CDF, md5=CDMD5),
             switches_env_on=mj.get('p2', {}).get('switches_env_on'),
             model_archive='the tier-p archive with model_swap.tsv applied: laptop/rebuild_model_archive_20261114.py, contents checked against the box-tested tree')
json.dump(mj, open(DST + 'context/MANIFEST.json', 'w'), indent=1)

# 4 model_add (fp16 copies), model_swap.tsv, box and laptop scripts
rows = []
for o, n in swaps:
    src = L + 'deploy_fp16/' + n; dst = DST + 'model_add/' + n
    if DRY: rows.append(f'{o}\t{n}\tDRY'); continue
    os.makedirs(os.path.dirname(dst), exist_ok=True); shutil.copy2(src, dst); os.chmod(dst, 0o644)
    hs, hd = sha256(src), sha256(dst); assert hs == hd, (n, hs, hd)
    rows.append(f'{o}\t{n}\t{hd}')
open(DST + 'model_swap.tsv', 'w').write('# old path (removed from the tier-p model tree)\tnew path (added)\tsha256 of the new file\n' + '\n'.join(rows) + '\n')
for sub, f in (('box', 'build_test_p3_20261114.sh'), ('laptop', 'rebuild_model_archive_20261114.py')):
    os.makedirs(DST + sub, exist_ok=True); shutil.copy2(HERE + '/' + f, DST + sub + '/' + f)
os.chmod(DST + 'box/build_test_p3_20261114.sh', 0o755)
n_ctx = sum(len(fs) for _, _, fs in os.walk(DST + 'context')); b_ctx = sum(os.path.getsize(os.path.join(d, f)) for d, _, fs in os.walk(DST + 'context') for f in fs)
b_add = 0 if DRY else sum(os.path.getsize(DST + 'model_add/' + n) for _, n in swaps)
print(json.dumps(dict(variant=V, dst=DST, swaps=swaps, cd_params=CDF, cd_md5=CDMD5, model_files=len(mf), context_files=n_ctx, context_bytes=b_ctx,
                      model_add_bytes=b_add, model_swap=rows), indent=1))

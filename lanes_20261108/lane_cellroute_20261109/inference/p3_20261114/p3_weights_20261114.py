"""Tier p3 weight check and fp16 copy (deploy lane, 2026-11-14, D-COORD-P3-PREP). For one new ViT-L final.pt (the encoder-LR-2e-5 full-data run):
  1 the run finished (complete.json next to it says completed; the arm's exit.json if present)
  2 load test on CPU against the tier-p ViT-L final.pt of the same seed: same container type, same tensor names, shapes and dtypes (so the strict
    member load that reads the old file reads the new one), every tensor finite, and the weights differ (it is a new model)
  3 the fp16 copy written exactly as the 11-11 copies were (inference/docker/convert_fp16.py one(): floating tensors with > 16 elements -> float16,
    reloaded and compared bit for bit with original.to(fp16)); an existing copy is never overwritten, it is re-verified instead
Result merged into weights_check_out.json next to this script (keyed by the new path). usage: p3_weights_20261114.py NEWPATH OLDPATH"""
import sys, os, json, time, hashlib, math
ROOT = '/workspace/clear-ec-research/'; L = ROOT + 'experiments/lane_cellroute_20261109/'
NEW, OLD = sys.argv[1], sys.argv[2]; HERE = os.path.dirname(os.path.abspath(__file__)); OUTF = HERE + '/weights_check_out.json'
sys.path.insert(0, L + 'inference/docker')
import torch
import convert_fp16 as C
torch.set_num_threads(2)


def same_nan(orig, copy, path='$'):
    """convert_fp16.same with one change: two float NaN leaves count as equal (NaN != NaN made a faithful copy look different)."""
    if isinstance(orig, float) and isinstance(copy, float) and math.isnan(orig) and math.isnan(copy): return []
    if isinstance(orig, dict):
        if not isinstance(copy, dict) or list(orig.keys()) != list(copy.keys()): return [path + ' keys']
        return [m for k in orig for m in same_nan(orig[k], copy[k], f'{path}[{k!r}]')]
    if isinstance(orig, (list, tuple)):
        if type(orig) != type(copy) or len(orig) != len(copy): return [path + ' len']
        return [m for i, (a, b) in enumerate(zip(orig, copy)) for m in same_nan(a, b, f'{path}[{i}]')]
    return C.same(orig, copy, torch, path)


def leaves(o, pre='$', out=None):
    out = {} if out is None else out
    if isinstance(o, torch.Tensor): out[pre] = o
    elif isinstance(o, dict):
        for k, v in o.items(): leaves(v, f'{pre}[{k!r}]', out)
    elif isinstance(o, (list, tuple)):
        for i, v in enumerate(o): leaves(v, f'{pre}[{i}]', out)
    else: out[pre] = ('value', repr(o)[:80])
    return out


def sha256(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 24), b''): h.update(b)
    return h.hexdigest()


t0 = time.time(); r = dict(new=NEW, old=OLD, checked_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
d = os.path.dirname(ROOT + NEW)
r['complete_json'] = json.load(open(d + '/complete.json')) if os.path.exists(d + '/complete.json') else None
arm = ROOT + NEW.split('/raw/')[0]; r['arm_exit_json'] = json.load(open(arm + '/exit.json')) if os.path.exists(arm + '/exit.json') else None
assert (r['complete_json'] or {}).get('status') == 'completed', ('run not complete', r['complete_json'])
r['bytes_new'], r['bytes_old'] = os.path.getsize(ROOT + NEW), os.path.getsize(ROOT + OLD)
a = torch.load(ROOT + NEW, map_location='cpu', weights_only=False); b = torch.load(ROOT + OLD, map_location='cpu', weights_only=False)
la, lb = leaves(a), leaves(b)
r['container'] = [type(a).__name__, type(b).__name__]
ta, tb = [k for k in la if isinstance(la[k], torch.Tensor)], [k for k in lb if isinstance(lb[k], torch.Tensor)]
r['tensor_names_equal'] = ta == tb; r['tensor_only_new'], r['tensor_only_old'] = [k for k in ta if k not in tb][:10], [k for k in tb if k not in ta][:10]
r['non_tensor_only_new'] = [k for k in la if k not in lb and k not in ta][:20]; r['non_tensor_only_old'] = [k for k in lb if k not in la and k not in tb][:20]
r['log_mean_new'], r['log_std_new'], r['log_mean_old'], r['log_std_old'] = a.get('log_mean'), a.get('log_std'), b.get('log_mean'), b.get('log_std')
r['stats_nonfinite_channels_new'] = [i for i, (m_, s_) in enumerate(zip(a['log_mean'], a['log_std'])) if not (math.isfinite(m_) and math.isfinite(s_))]
ten = [k for k in la if isinstance(la[k], torch.Tensor) and isinstance(lb.get(k), torch.Tensor)]
r['n_tensors'] = len(ten); r['n_params'] = int(sum(la[k].numel() for k in ten))
r['shape_mismatch'] = [k for k in ten if la[k].shape != lb[k].shape][:10]
r['dtype_mismatch'] = [k for k in ten if la[k].dtype != lb[k].dtype][:10]
r['nonfinite'] = [k for k in ten if la[k].is_floating_point() and not torch.isfinite(la[k]).all()][:10]
fl = [k for k in ten if la[k].is_floating_point() and la[k].shape == lb[k].shape]
r['n_float_tensors_differ'] = int(sum(not torch.equal(la[k], lb[k]) for k in fl))
r['max_abs_diff_vs_old'] = float(max((la[k].float() - lb[k].float()).abs().max() for k in fl))
r['non_tensor_leaves_differ'] = [k for k in la if not isinstance(la[k], torch.Tensor) and la[k] != lb.get(k)][:10]
del a, b, la, lb
# V uses channel 0 (CD) only (members.vitl_tta_only -> recipe_infer parts['V'][0]): its stats must be finite; other channels are reported
r['load_ok'] = bool(r['tensor_names_equal'] and not r['shape_mismatch'] and not r['dtype_mismatch'] and not r['nonfinite'] and 0 not in r['stats_nonfinite_channels_new'])
dst = L + 'deploy_fp16/' + NEW
if os.path.exists(dst):
    o = torch.load(ROOT + NEW, map_location='cpu', weights_only=False); c = torch.load(dst, map_location='cpu', weights_only=False)
    bad = same_nan(o, c); r['fp16'] = dict(file=NEW, existed=True, n_mismatch=len(bad), mismatches=bad[:5], check='same_nan'); del o, c
else:
    r['fp16'] = C.one(NEW)
    if r['fp16'].get('n_mismatch'):   # re-check NaN-aware (convert_fp16.same counts NaN leaves as different)
        o = torch.load(ROOT + NEW, map_location='cpu', weights_only=False); c = torch.load(dst, map_location='cpu', weights_only=False)
        bad = same_nan(o, c); r['fp16'].update(n_mismatch_convert_fp16=r['fp16']['n_mismatch'], n_mismatch=len(bad), mismatches=bad[:5], check='same_nan'); del o, c
r['fp16']['bytes'] = os.path.getsize(dst); r['fp16']['sha256'] = sha256(dst)
od = L + 'deploy_fp16/' + OLD; r['fp16_old_bytes'] = os.path.getsize(od) if os.path.exists(od) else None
r['seconds'] = round(time.time() - t0, 1)
allr = json.load(open(OUTF)) if os.path.exists(OUTF) else {}
allr.setdefault(NEW + ' (earlier checks)', []).extend([allr.pop(NEW)] if NEW in allr else []); allr[NEW] = r; json.dump(allr, open(OUTF + '.tmp', 'w'), indent=1); os.replace(OUTF + '.tmp', OUTF)
print(json.dumps(r, indent=1, default=str))
sys.exit(0 if r['load_ok'] and r['fp16'].get('n_mismatch') == 0 else 1)

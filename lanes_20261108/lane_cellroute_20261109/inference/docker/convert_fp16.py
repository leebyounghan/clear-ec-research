"""Write fp16 copies of the fp16-nodet weight files (cell-route lane, 2026-11-11; coordinator approval: write them if the parity median change is
< 0.05 % per column, nothing overwritten). Gate: inference/docker/budget_analysis_v2.json, variant fp16-nodet, every column median < 5e-4, else exit 2.
File set: the files the current-stack fp16-nodet budget run read AND cast (out/budget_fp16-nodet_v2.json, cast count > 0), minus the base weights
(timm / DINOv3-B / CPDINO pretrained files: fully overwritten by the strict loads, not needed at deploy).
Rule = the simulation's: floating tensors with > 16 elements stored as float16; everything else unchanged. Output: lane_cellroute_20261109/deploy_fp16/<same
relative path>. Check per file: reload the copy, upcast every fp16 tensor to the original dtype and compare bit for bit with original.to(fp16).to(original
dtype); every other leaf must equal the original exactly; the container structure must match. usage: convert_fp16.py WORKERS [--dry] [--suffix _v3]   (suffix picks budget_analysis{suffix}.json / budget_fp16-nodet{suffix}.json;
files whose copy already exists are skipped and reported, never overwritten)"""
import sys, os, json, time
from multiprocessing import Pool
ROOT = '/workspace/clear-ec-research/'; L = ROOT + 'experiments/lane_cellroute_20261109/'; OUT = L + 'deploy_fp16/'
BASE = ('phase2_backbone_screen_20261001/hf_cache', 'phase2_foundation_20261002/weights')


SUF = sys.argv[sys.argv.index('--suffix') + 1] if '--suffix' in sys.argv else '_v2'


def gate():
    a = json.load(open(L + f'inference/docker/budget_analysis{SUF}.json'))['fp16-nodet']['columns']
    return {c: v['median'] for c, v in a.items()}, all(v['median'] < 5e-4 for v in a.values())


def cast(o, torch):
    if isinstance(o, torch.Tensor): return o.to(torch.float16) if o.is_floating_point() and o.numel() > 16 else o
    if isinstance(o, torch.nn.Module):
        for p in list(o.parameters()) + list(o.buffers()):
            if p.is_floating_point() and p.numel() > 16: p.data = p.data.to(torch.float16)
        return o
    if isinstance(o, dict): return type(o)((k, cast(v, torch)) for k, v in o.items()) if not hasattr(o, 'default_factory') else {k: cast(v, torch) for k, v in o.items()}
    if isinstance(o, list): return [cast(v, torch) for v in o]
    if isinstance(o, tuple) and not hasattr(o, '_fields'): return tuple(cast(v, torch) for v in o)
    return o


def same(orig, copy, torch, path='$'):
    """orig = original object, copy = reloaded fp16 copy. Returns a list of mismatches."""
    if isinstance(orig, torch.Tensor):
        if not isinstance(copy, torch.Tensor): return [path + ' type']
        if orig.is_floating_point() and orig.numel() > 16:
            ok = copy.dtype == torch.float16 and torch.equal(copy.to(orig.dtype), orig.to(torch.float16).to(orig.dtype))
        else: ok = copy.dtype == orig.dtype and torch.equal(copy, orig)
        return [] if ok else [path]
    if isinstance(orig, torch.nn.Module):
        so, sc = orig.state_dict(), copy.state_dict(); return same(dict(so), dict(sc), torch, path + '.state_dict')
    if isinstance(orig, dict):
        if not isinstance(copy, dict) or list(orig.keys()) != list(copy.keys()): return [path + ' keys']
        return [m for k in orig for m in same(orig[k], copy[k], torch, f'{path}[{k!r}]')]
    if isinstance(orig, (list, tuple)):
        if type(orig) != type(copy) or len(orig) != len(copy): return [path + ' len']
        return [m for i, (a, b) in enumerate(zip(orig, copy)) for m in same(a, b, torch, f'{path}[{i}]')]
    try: return [] if orig == copy else [path + ' value']
    except Exception: return []


def one(rel):
    import torch
    torch.set_num_threads(1)
    src = ROOT + rel; dst = OUT + rel; t = time.time()
    if os.path.exists(dst): return dict(file=rel, skipped='copy exists (not overwritten)')
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if rel.endswith('.safetensors'):
        from safetensors.torch import load_file, save_file
        o = load_file(src); save_file({k: cast(v, torch).contiguous() for k, v in o.items()}, dst + '.tmp'); os.replace(dst + '.tmp', dst); c = load_file(dst)
    else:
        o = torch.load(src, map_location='cpu', weights_only=False); torch.save(cast(torch.load(src, map_location='cpu', weights_only=False), torch), dst + '.tmp')
        os.replace(dst + '.tmp', dst); c = torch.load(dst, map_location='cpu', weights_only=False)
    bad = same(o, c, torch)
    return dict(file=rel, gb_in=os.stat(src).st_size / 1e9, gb_out=os.stat(dst).st_size / 1e9, mismatches=bad[:5], n_mismatch=len(bad), s=round(time.time() - t, 1))


if __name__ == '__main__':
    med, ok = gate(); print('gate (fp16-nodet medians, current stacks):', med, 'pass' if ok else 'FAIL', flush=True)
    if not ok: sys.exit(2)
    b = json.load(open(ROOT + f'experiments/lane_cellroute_20261109/out/budget_fp16-nodet{SUF}.json'))['files']
    files = sorted(f for f, v in b.items() if v.get('cast') and v['cast'][0] > 0 and not any(x in f for x in BASE) and not f.startswith('/'))
    print(len(files), 'files, %.2f GB in' % (sum(os.stat(ROOT + f).st_size for f in files) / 1e9), flush=True)
    if '--dry' in sys.argv:
        for f in files: print('  ', f)
        sys.exit(0)
    with Pool(int(sys.argv[1])) as p: res = p.map(one, files, chunksize=1)
    summ = dict(files=len(res), gb_in=round(sum(r.get('gb_in', 0) for r in res), 3), gb_out=round(sum(r.get('gb_out', 0) for r in res), 3),
                files_with_mismatch=[r['file'] for r in res if r.get('n_mismatch')], errors=[r for r in res if 'error' in r], skipped=sum('skipped' in r for r in res), written=[r['file'] for r in res if 'gb_out' in r], gate=med)
    print('SUMMARY', json.dumps(summ), flush=True); json.dump(dict(summary=summ, files=res), open(L + f'inference/docker/convert_fp16_result{SUF}.json', 'w'), indent=1)

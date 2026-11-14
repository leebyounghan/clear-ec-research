"""From-scratch combination search for HEX (HEX lane, 2026-11-11, PI via the coordinator). CPU only.
Forward selection from the best pipeline with the whole pool available at every round, nested per outer fold: for outer fold
f, every round applies each pool step to the current candidates and keeps, for f, the step with the lowest HEX on the rows of the
other four folds (the step's own options and shares are already chosen per outer fold on the other folds, as in chain.py);
fold f stops when no step lowers that score. The reported value is the five-fold OOF HEX of the per-fold candidates (each row
scored by its own fold's candidate and stretch), nothing chosen on the rows it is scored on. `ap1` rebuilds the base from scratch,
so it is offered in round 1 only. Steps are applied at most once per fold.
usage: combo_search.py OUT.json [workers] [max_rounds]"""
import os, sys, json, time
os.environ.setdefault('OMP_NUM_THREADS', '1')
import numpy as np
LANE = '/workspace/clear-ec-research/experiments/lane_hex_20261108'
sys.path.insert(0, LANE + '/source')
import chain as C

N, fold, y, ok = C.N, C.fold, C.y[:, 2], C.ok
GNN = ['hex_pre46_s5', 'hex_pre47_s5', 'hex_pre4647_s5', 'hex_pre46_s10', 'hex_pre46ctx_s5', 'hex_imgctx4_s5', 'hex_ridge14_s5', 'hex_pre46_s5b']
RIDGE = ['box2', 'fprefrz', 'frz46', 'k1', 'p47', 'p48', 'sg2s46', 'sg2s49', 'sig5']
POOL = ['ap1', 'h6w', 'graphens', 'gscale', 'quad', 'fuse16', 'prior', 'priorall', 'imgstats', 'cdcv', 'cdcvimg', 'fuseall', 'legib', 'imgmape',
        'imgvig', 'imgrich', 'legwin', 'mapelin', 'uncridge', 'ridgepool', 'ridgeev', 'ridgeev_k5', 'ridgeev_fall16'] + [f'ridgeev:{t}' for t in RIDGE] + \
       ['seg', 'segm', 'segb', 'segg', 'segv', 'segm2', 'segr', 'segvig', 'cnn5', 'mtcnn5', 'mtcnn5_fix', 'cvdino_fix', 'cvw3_fix', 'cvw3s47_fix',
        'cvw3m_fix', 'attnp5_fix'] + [f'gnn:{g}' for g in GNN]
CAND = None


def apply(s, cand):
    if s.startswith('gnn:'): return C.step_gnn(cand, s[4:])
    if s.startswith('ridgeev:'): return C.step_ridgeev(cand, s[8:])
    return C.STEPS[s](cand)


def scores(cand):
    P = C.final(cand)[:, 2]; E = np.abs(P - y) / np.where(ok, y, 1)
    return [float(E[(fold != f) & ok].mean()) for f in range(5)], [float(E[(fold == f) & ok].mean()) for f in range(5)], float(E[ok].mean())


def evaluate(s):
    t0 = time.time()
    try:
        new, picks = apply(s, CAND); inner, outer, _ = scores(new)
        return s, new, picks, inner, outer, round(time.time() - t0, 1), None
    except Exception as e:
        return s, None, None, None, None, round(time.time() - t0, 1), repr(e)[:200]


if __name__ == '__main__':
    out = sys.argv[1]; nw = int(sys.argv[2]) if len(sys.argv) > 2 else 4; R = int(sys.argv[3]) if len(sys.argv) > 3 else 25
    from multiprocessing import get_context
    BASE = os.environ.get('COMBO_BASE', 'hb')   # 'hb' = best pipeline; 'ap1' = ap1 applied first (its own nested picks), ap1 then leaves the pool
    CAND = {f: C.L.HB[f].copy() for f in range(5)}
    if BASE == 'ap1': CAND, ap1_picks = C.STEPS['ap1'](CAND); POOL = [p for p in POOL if p != 'ap1']; print('base ap1, picks', ap1_picks, flush=True)
    chosen = {f: [] for f in range(5)}; active = {f: True for f in range(5)}
    inner0, outer0, all0 = scores(CAND); cur = list(inner0); log = dict(pool=POOL, base=dict(inner=inner0, outer=outer0, all5=all0 * 100), rounds=[], errors={})
    print('base five folds', round(all0 * 100, 4), flush=True)
    for r in range(R):
        if not any(active.values()): break
        todo = [s for s in POOL if s not in log['errors'] and (r == 0 or s != 'ap1') and any(active[f] and s not in chosen[f] for f in range(5))]
        with get_context('fork').Pool(nw) as pool: res = pool.map(evaluate, todo, chunksize=1)
        new_cand = dict(CAND); rd = dict(round=r + 1, picks={}, evaluated=len(todo))
        for s, new, picks, inner, outer, sec, err in res:
            if err: log['errors'][s] = err
        for f in range(5):
            if not active[f]: continue
            best = None
            for s, new, picks, inner, outer, sec, err in res:
                if err or s in chosen[f]: continue
                if best is None or inner[f] < best[1]: best = (s, inner[f], new, picks)
            if best is not None and best[1] < cur[f]:
                chosen[f].append(best[0]); new_cand[f] = best[2][f]; rd['picks'][f] = dict(step=best[0], inner=round(best[1] * 100, 5), gain=round((best[1] - cur[f]) * 100, 5), option=best[3][f]); cur[f] = best[1]
            else:
                active[f] = False; rd['picks'][f] = 'stop'
        CAND = new_cand; inner, outer, a5 = scores(CAND); rd.update(all5=round(a5 * 100, 5), outer=[round(o * 100, 4) for o in outer], seconds={s: sec for s, _, _, _, _, sec, _ in res})
        log['rounds'].append(rd); print(json.dumps({k: rd[k] for k in ('round', 'picks', 'all5')}), flush=True)
        json.dump(dict(log, chosen=chosen), open(out, 'w'), indent=1)
    inner, outer, a5 = scores(CAND); f04 = float(np.mean(np.abs(C.final(CAND)[:, 2] - y)[ok & np.isin(fold, [0, 4])] / y[ok & np.isin(fold, [0, 4])]))
    log.update(final=dict(all5=round(a5 * 100, 5), f04=round(f04 * 100, 5), outer=[round(o * 100, 4) for o in outer]), chosen=chosen)
    json.dump(log, open(out, 'w'), indent=1)
    np.savez_compressed(out.replace('.json', '.npz'), **{f'c{f}': CAND[f] for f in range(5)}, final_hex=C.final(CAND)[:, 2], idx=C.idx, fold=fold)
    print('final five folds', log['final'], 'chosen', chosen, 'errors', list(log['errors']), flush=True)

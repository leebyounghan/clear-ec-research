import sys, time, os, numpy as np
sys.path.insert(0, 'experiments/green_day3_20261103/source')
def rss(): return int([l for l in open('/proc/self/status') if l.startswith('VmRSS')][0].split()[1])/1024
def cg(): return int(open('/sys/fs/cgroup/memory.current').read())/2**30
r0, c0 = rss(), cg()
import green_day3_e11_nolabel_20261103 as M
r1, c1 = rss(), cg(); t = time.time()
rows, pk = M.load_pseudo(); dt = time.time()-t
r2, c2 = rss(), cg()
n = [len(pk[r]) for r in rows]; pos = [int((pk[r][:, 2] >= M.TH_POS).sum()) for r in rows]
tot = sum(a.nbytes for a in pk.values())/2**20; bases = len({id(a.base) for a in pk.values()})
print(dict(rows=len(rows), peaks_total=int(sum(n)), median_per_row=float(np.median(n)), max_per_row=int(max(n)), median_pos=float(np.median(pos)), gated_rows=int(sum(p >= M.GATE_N for p in pos)), dict_MB=round(tot, 1), distinct_base_arrays=bases, seconds=round(dt, 1)))
print(dict(rss_MB_start=round(r0), rss_MB_after_imports=round(r1), rss_MB_after_load=round(r2), cgroup_GB_start=round(c0, 2), cgroup_GB_after_imports=round(c1, 2), cgroup_GB_after_load=round(c2, 2)))

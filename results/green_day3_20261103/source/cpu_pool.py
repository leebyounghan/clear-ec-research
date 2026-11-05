"""PI rule (2026-11-03) for CPU-only passes on the pod: pool sized to the free CPU quota, fork after loading shared
inputs, OMP_NUM_THREADS=1 in workers, chunked imap, never all 16 cores."""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1'); os.environ.setdefault('OPENBLAS_NUM_THREADS', '1'); os.environ.setdefault('MKL_NUM_THREADS', '1')
import multiprocessing as mp

QUOTA = 16


def n_workers():
    load1 = os.getloadavg()[0]
    return int(min(12, max(2, round(QUOTA-load1-1))))


def pmap(fn, items, chunksize=16, workers=None, ordered=True):
    """Fork-pool map over items (shared inputs must be loaded in the parent before the call)."""
    w = workers or n_workers()
    items = list(items)
    if len(items) <= 2*chunksize or w <= 1:
        return [fn(x) for x in items]
    with mp.get_context('fork').Pool(w) as pool:
        it = pool.imap(fn, items, chunksize=chunksize) if ordered else pool.imap_unordered(fn, items, chunksize=chunksize)
        return list(it)

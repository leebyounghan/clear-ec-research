"""Dependency trace of one bb_runner_20261108.py run (deploy lane, 2026-11-13, build-box training prep). GPU (one, tiny), seconds.
Runs the runner unchanged through runpy with the arguments given here (use --smoke N: N updates, the runner writes nothing) and records, in
every process (main and forked DataLoader workers), each file the run touches:
  - audit events 'open', 'os.listdir', 'os.scandir' (Python-level opens: numpy memmap, torch.load, json, imports)
  - file-backed mappings and open fds from /proc/self/{maps,fd}, sampled every 0.25 s (catches Rust / C opens such as safetensors mmap)
and, in the main process at exit, every imported module (repo files by path, third-party packages by distribution and version) and the first
4-D input shape seen per module class (model input size). Output: $TRACE_OUT/{touch_<pid>.txt, modules.json, shapes.json}.
usage: TRACE_OUT=dir python trace_train_20261113.py <bb_runner arguments>"""
import os, sys, json, atexit, threading, runpy, time
ROOT = '/workspace/clear-ec-research/'
OUT = os.environ['TRACE_OUT']; os.makedirs(OUT, exist_ok=True)
_busy = threading.local(); _seen = {}


def _write(kind, p):
    key = (os.getpid(), kind, p)
    if key in _seen: return
    _seen[key] = 1
    _busy.on = True
    try:
        fd = os.open(f'{OUT}/touch_{os.getpid()}.txt', os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
        os.write(fd, f'{kind}\t{p}\n'.encode()); os.close(fd)
    finally:
        _busy.on = False


def _hook(event, args):
    if getattr(_busy, 'on', False) or event not in ('open', 'os.listdir', 'os.scandir'): return
    try:
        a = args[0]
        if a is None or isinstance(a, int): return
        p = os.path.abspath(os.fsdecode(a))
    except Exception:
        return
    _write(event, p)


sys.addaudithook(_hook)


def _sample():
    while True:
        try:
            _busy.on = True
            try:
                with open('/proc/self/maps') as f: maps = [l.split(None, 5)[-1].strip() for l in f if '/' in l]
                fds = []
                for d in os.listdir('/proc/self/fd'):
                    try: fds.append(os.readlink('/proc/self/fd/' + d))
                    except OSError: pass
            finally:
                _busy.on = False
            for p in maps: _write('map', p)
            for p in fds:
                if p.startswith('/'): _write('fd', p)
        except Exception:
            pass
        time.sleep(0.25)


def _start_sampler():
    threading.Thread(target=_sample, daemon=True).start()


_start_sampler(); os.register_at_fork(after_in_child=_start_sampler)
SHAPES = {}


def _pre(mod, inp):
    n = type(mod).__name__
    if n not in SHAPES:
        for x in inp:
            if hasattr(x, 'ndim') and x.ndim == 4: SHAPES[n] = [list(x.shape), str(x.dtype)]; break


def _dump():
    if os.getpid() != MAIN: return
    import importlib.metadata as md
    repo, third = set(), {}
    dist_of = md.packages_distributions()
    for name, m in list(sys.modules.items()):
        f = getattr(m, '__file__', None)
        if f and os.path.abspath(f).startswith(ROOT): repo.add(os.path.relpath(os.path.abspath(f), ROOT))
        elif f and '.' not in name:
            for d in dist_of.get(name, [name]):
                try: third[name] = [d, md.version(d)]
                except Exception: third[name] = [d, None]
    json.dump(dict(repo_modules=sorted(repo), third_party=third), open(f'{OUT}/modules.json', 'w'), indent=1)
    json.dump(SHAPES, open(f'{OUT}/shapes.json', 'w'), indent=1)


MAIN = os.getpid(); atexit.register(_dump)
import torch  # noqa: E402
torch.nn.modules.module.register_module_forward_pre_hook(_pre)
RUNNER = ROOT + 'experiments/lane_backbone_20261108/source/bb_runner_20261108.py'
sys.argv = [RUNNER] + sys.argv[1:]
os.chdir(ROOT)
runpy.run_path(RUNNER, run_name='__main__')

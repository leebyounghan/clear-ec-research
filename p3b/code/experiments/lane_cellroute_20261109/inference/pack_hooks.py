"""Packaging hooks for the deploy image (deploy lane, 2026-11-12, D-PKG-001). Inactive unless the environment variables are set, so the research
entry behaves exactly as before. recipe_infer imports this module first.
  CLEAR_EC_FP16_LIST  text file of repo-relative weight paths stored as fp16 copies (docker/fp16_files.txt). For those files every float16 tensor is
                      upcast to float32 right after torch.load. The copies are original fp32 -> fp16 for tensors with > 16 elements (all other tensors
                      untouched), so the loaded tensors equal what the fp16-nodet parity runs simulated.
  CLEAR_EC_FP16_ROOT  optional directory holding the copies at <root>/<repo-relative path> (research pod). Unset in the container, where the copies sit
                      at the original paths.
  CLEAR_EC_SKIP_BASE  base_shapes.json: the pretrained base files read only to construct architectures are not read. torch.load / safetensors load_file
                      of those paths return zero tensors with the recorded names, shapes and dtypes; transformers AutoModel.from_pretrained of the
                      DINOv3-B directory builds the model from its config.json. Every member's strict checkpoint load overwrites them (verified
                      2026-11-11, test_base_weights.py: outputs bit-identical with noise in place of the base tensors).
  Tier p2 read-once switches (deploy lane, 2026-11-14; each off by default, all off = tier p exactly). With a switch on, the first torch.load of a
  file in its group in a job loads it once to CPU and keeps it; later loads of the same file in the same job return a copy of the container
  structure (dicts / lists rebuilt, OrderedDict _metadata kept) whose tensors are the cached CPU tensors, or copies on the requested device when
  map_location names one (cellpose loads cpsam_v2 onto the GPU). recipe_infer releases a group after its last stage and everything at the end of
  predict. No consumer of these files changes a loaded tensor in place (they only call load_state_dict, which copies).
  CLEAR_EC_RO_RECIPEA  recipe A s46 epoch_20.pt: recipeA_tta, the 4 frozen-head detectors' encoder (other_peaks), cv_parts token pass
  CLEAR_EC_RO_HEX6     the 6 detector files hex6 shares with peaks / other_peaks: sg2_s46-s50.pt, sg2pre_s46.pt
  CLEAR_EC_RO_CPSAM    cpsam_models/cpsam_v2: P1 and seg each build a CellposeModel, which loads it twice (get_backbone mmap + load_model)"""
import os, json
ROOT = '/workspace/clear-ec-research/'
FP16 = set(); FP16_ROOT = os.environ.get('CLEAR_EC_FP16_ROOT') or None; BASE = None; AUTOCAST = 'bf16'
STATS = dict(fp16_loads=0, base_zero=0, base_from_config=0, read_bytes=0, read_s=0.0, autocast='bf16', ro_hits=0, ro_hit_bytes=0, ro_peak_bytes=0)
_DET = 'experiments/green_track_20261028/deploy/detectors/'
RO_SETS = (('CLEAR_EC_RO_RECIPEA', 'recipeA', ['experiments/fulldata_recipeA_20261028/regression_std_s46/runs/fold99/cpdino_vitb/epoch_20.pt']),
           ('CLEAR_EC_RO_HEX6', 'hex6', [_DET + f'{n}.pt' for n in ('sg2_s46', 'sg2_s47', 'sg2_s48', 'sg2_s49', 'sg2_s50', 'sg2pre_s46')]),
           ('CLEAR_EC_RO_CPSAM', 'cpsam', ['cpsam_models/cpsam_v2']))
RO_GROUPS = {p: g for flag, g, ps in RO_SETS if os.environ.get(flag) == '1' for p in ps}
STATS['ro_groups'] = sorted(set(RO_GROUPS.values()))
_RO, _RO_SIZE, _RO_CLOSED = {}, {}, set()


def ro_reset():
    """start of a job (recipe_infer.predict): empty cache, every group open"""
    _RO.clear(); _RO_SIZE.clear(); _RO_CLOSED.clear()


def ro_release(group=None):
    """after a group's last stage (or every group): drop its cached files; later loads of them read from disk as in tier p"""
    for r in [r for r in _RO if group is None or RO_GROUPS.get(r) == group]: _RO.pop(r, None); _RO_SIZE.pop(r, None)
    _RO_CLOSED.update(set(RO_GROUPS.values()) if group is None else {group})


def _ro_copy(o, torch, dev):
    import copy
    if isinstance(o, torch.Tensor): return o if dev is None else o.to(dev)
    if isinstance(o, dict):
        c = copy.copy(o)        # same type, attributes kept (state-dict OrderedDict _metadata)
        for k in list(c.keys()): c[k] = _ro_copy(c[k], torch, dev)
        return c
    if isinstance(o, list): return [_ro_copy(v, torch, dev) for v in o]
    if isinstance(o, tuple) and not hasattr(o, '_fields'): return tuple(_ro_copy(v, torch, dev) for v in o)
    return o


def rel(p):
    try: return os.path.relpath(os.path.abspath(os.fspath(p)), ROOT)
    except TypeError: return None


def _upcast(o, torch):
    if isinstance(o, torch.Tensor): return o.float() if o.dtype == torch.float16 else o
    if isinstance(o, torch.nn.Module):
        for t in list(o.parameters()) + list(o.buffers()):
            if t.dtype == torch.float16: t.data = t.data.float()
        return o
    if isinstance(o, dict):
        for k in list(o.keys()): o[k] = _upcast(o[k], torch)
        return o
    if isinstance(o, list): return [_upcast(v, torch) for v in o]
    if isinstance(o, tuple) and not hasattr(o, '_fields'): return tuple(_upcast(v, torch) for v in o)
    return o


def _zeros(spec, torch):
    STATS['base_zero'] += 1
    return {k: torch.zeros(tuple(s), dtype=getattr(torch, d)) for k, (s, d) in spec.items()}


PREC = None; FP32_MODULES = set()
EMU_OPS = ('mm', 'addmm', 'bmm', 'baddbmm', 'addbmm', '_addmm_activation', 'convolution', 'cudnn_batch_norm', 'native_batch_norm',
           '_native_batch_norm_legit_no_training')


def _no_tf32(torch):
    """CLEAR_EC_NO_TF32=1: TF32 off for fp32 matmuls and convolutions, and later attempts to switch it on ignored (a T4 has no TF32, so the H200
    tests of the T4 paths run fp32 operations as true fp32)."""
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    CM = type(torch.backends.cuda.matmul); _sa = CM.__setattr__
    def sa(self, name, value):
        if name == 'allow_tf32' and value: STATS['tf32_refused'] = STATS.get('tf32_refused', 0) + 1; return
        return _sa(self, name, value)
    CM.__setattr__ = sa
    get = torch._C._get_cudnn_allow_tf32
    type(torch.backends.cudnn).allow_tf32 = property(lambda self: get(), lambda self, v: torch._C._set_cudnn_allow_tf32(False))
    torch.set_float32_matmul_precision = lambda *a, **k: None


def _emu_mode(torch):
    """bf16 emulation (CLEAR_EC_PREC=emu): inside a bf16 autocast region (and in the bf16 CPSAM segmentation net) every matmul, convolution and
    batch norm that receives a bf16 cuda tensor runs in fp32 on the bf16 values and returns its result rounded to bf16, as a bf16 kernel with
    fp32 accumulation does. Elementwise bf16 kernels run as they are. Needs no bf16 matmul / convolution kernel, so it runs on any CUDA GPU."""
    from torch.utils._python_dispatch import TorchDispatchMode
    ops = {getattr(torch.ops.aten, n).default for n in EMU_OPS if hasattr(torch.ops.aten, n)}
    seen = STATS.setdefault('emu_ops', {}); left = STATS.setdefault('bf16_native_ops', {})
    def up(a): return a.float() if isinstance(a, torch.Tensor) and a.dtype is torch.bfloat16 else a
    class Emu(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            kwargs = kwargs or {}
            if any(isinstance(a, torch.Tensor) and a.dtype is torch.bfloat16 and a.is_cuda for a in args):
                n = func.__name__
                if func in ops:
                    seen[n] = seen.get(n, 0) + 1
                    out = func(*[up(a) for a in args], **{k: up(v) for k, v in kwargs.items()})
                    if isinstance(out, torch.Tensor): return out.to(torch.bfloat16)
                    return tuple(o.to(torch.bfloat16) if isinstance(o, torch.Tensor) and o.is_floating_point() and k == 0 else o for k, o in enumerate(out))
                left[n] = left.get(n, 0) + 1
            return func(*args, **kwargs)
    return Emu


def autocast_mode():
    """Legacy rule (CLEAR_EC_PREC unset): CLEAR_EC_AUTOCAST=fp16 (or unset on a GPU below compute capability 8.0, e.g. the T4): report bf16 as
    unsupported and map every cuda bf16 autocast to fp16, including the lane modules that hard-code torch.bfloat16 (detectors, hex6, attention
    pool, CD family code); CPSAM segmentation then runs float32 (recipe_infer reads AUTOCAST == 'fp16').
    CLEAR_EC_PREC (D-PKG-004 tests of a T4-safe path; overrides the legacy rule):
      bf16  autocast as written (the validated path)
      fp16  as the legacy fp16 rule
      mix   fp16, except autocast regions entered from a module in CLEAR_EC_FP32_MODULES (comma list of module names), which run fp32
      fp32  every cuda autocast region runs fp32; CPSAM segmentation float32
      emu   bf16 autocast kept and emulated (_emu_mode); CPSAM segmentation bf16, emulated; flash / memory-efficient / cuDNN attention off (the math
            path upcasts bf16 attention to fp32, the only attention kernel a T4 has for bf16)
    CLEAR_EC_NO_TF32=1 with any of them: true fp32 for fp32 operations, as on a T4."""
    global AUTOCAST, PREC, FP32_MODULES
    import torch, sys as _sys
    prec = os.environ.get('CLEAR_EC_PREC')
    if prec is None:
        mode = os.environ.get('CLEAR_EC_AUTOCAST')
        if mode is None and torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] < 8:
            # container default on a T4 (D-PKG-006, measured 2026-11-12): fp16 with one copy in the 13 detectors' single view. On 20 frames fp16
            # and fp32 are equally far from the validated bf16 outputs (strict-score cost +0.0065 vs +0.0053, inside the noise), one copy changes
            # outputs by < 5e-7, and fp32 or the 4-copy batch cost T4 time the 300 s limit does not have.
            mode = 'fp16'; os.environ.setdefault('CLEAR_EC_DET_COPIES', '1'); STATS['t4_default'] = True
        if mode != 'fp16': return
        AUTOCAST = STATS['autocast'] = 'fp16'; torch.cuda.is_bf16_supported = lambda *a, **k: False
        _AC = torch.autocast
        class AC(_AC):
            def __init__(self, device_type, dtype=None, enabled=True, cache_enabled=None):
                if device_type == 'cuda' and dtype is torch.bfloat16: dtype = torch.float16
                super().__init__(device_type, dtype=dtype, enabled=enabled, cache_enabled=cache_enabled)
        torch.autocast = AC; torch.amp.autocast = AC
        return
    assert prec in ('bf16', 'fp16', 'mix', 'fp32', 'emu'), prec
    PREC = STATS['prec'] = prec
    if os.environ.get('CLEAR_EC_NO_TF32') == '1': _no_tf32(torch); STATS['no_tf32'] = True
    FP32_MODULES = {m for m in os.environ.get('CLEAR_EC_FP32_MODULES', '').split(',') if m}
    STATS['fp32_modules'] = sorted(FP32_MODULES); calls = STATS.setdefault('autocast_calls', {})
    AUTOCAST = STATS['autocast'] = {'bf16': 'bf16', 'fp16': 'fp16', 'mix': 'fp16', 'fp32': 'fp16', 'emu': 'bf16'}[prec]   # 'fp16': CPSAM float32
    if prec in ('fp16', 'mix', 'fp32'): torch.cuda.is_bf16_supported = lambda *a, **k: False
    Emu = None
    if prec == 'emu':
        Emu = _emu_mode(torch)
        for f in ('enable_flash_sdp', 'enable_cudnn_sdp'):       # neither has an fp32 kernel; a T4 has neither for bf16
            if hasattr(torch.backends.cuda, f): getattr(torch.backends.cuda, f)(False)
        _install_sdpa_emu(torch)
        _install_seg_emu(Emu)
    _AC = torch.autocast
    class AC(_AC):
        def __init__(self, device_type, dtype=None, enabled=True, cache_enabled=None):
            self._emu = None
            if device_type == 'cuda' and enabled:
                mod = _sys._getframe(1).f_globals.get('__name__', '?'); key = f'{CUR_STAGE}:{mod}'; calls[key] = calls.get(key, 0) + 1
                p = 'fp32' if (prec == 'fp32' or mod in FP32_MODULES) else prec
                if p == 'fp32': enabled = False
                elif p in ('fp16', 'mix') and dtype is torch.bfloat16: dtype = torch.float16
                elif p == 'emu': dtype = torch.bfloat16; self._emu = Emu()
            super().__init__(device_type, dtype=dtype, enabled=enabled, cache_enabled=cache_enabled)
        def __enter__(self):
            r = super().__enter__()
            if self._emu is not None: self._emu.__enter__()
            return r
        def __exit__(self, *a):
            if self._emu is not None: self._emu.__exit__(*a)
            return super().__exit__(*a)
    torch.autocast = AC; torch.amp.autocast = AC


CUR_STAGE = '-'          # set by test harnesses for the autocast call census


def det_copies():
    """CLEAR_EC_DET_COPIES=n: the 13 non-pre-only detectors' single view (deploy_hexkonan_image.single_peaks) runs as a batch of n copies instead
    of 4. The 4 identical copies exist only so that native bf16 numerics match the batch-4 training peaks; off the bf16 path they buy nothing and
    cost 3/4 of those detectors' FLOPs (0.73 of 1.9 PFLOP per t1 frame). Unset: unchanged."""
    n = os.environ.get('CLEAR_EC_DET_COPIES')
    if not n: return
    n = int(n); STATS['det_copies'] = n
    import importlib.abc, importlib.util, sys as _sys
    def patch(mod):
        import torch
        @torch.no_grad()
        def single_peaks(enc, head, image_u8):
            return mod.SG.peaks2(mod.heat(enc, head, mod.preprocess(image_u8).repeat(n, 1, 1, 1))[0], mod.SG.frame_support(), 0.2)
        mod.single_peaks = single_peaks
    name = 'deploy_hexkonan_image'
    if name in _sys.modules: patch(_sys.modules[name]); return
    class Post(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path, target=None):
            if fullname != name: return None
            _sys.meta_path.remove(self)
            try: spec = importlib.util.find_spec(fullname)
            finally: _sys.meta_path.insert(0, self)
            if spec is None or spec.loader is None: return spec
            ex = spec.loader.exec_module
            def exec_module(module): ex(module); patch(module)
            spec.loader.exec_module = exec_module; return spec
    _sys.meta_path.insert(0, Post())


def _install_sdpa_emu(torch):
    """emu: attention on bf16 q / k / v runs in fp32 (memory-efficient kernel, which a T4 has for fp32; the math path would build the full
    20k x 20k attention matrix, 74 GB for one detector) on the bf16 values, autocast off inside, result rounded to bf16."""
    F = torch.nn.functional; _sdpa = F.scaled_dot_product_attention
    def rb(a): return a.to(torch.bfloat16).float() if isinstance(a, torch.Tensor) and a.is_floating_point() else a   # the cast autocast would make
    def sdpa(q, k, v, *a, **kw):
        # bf16 inputs, or fp32 inputs inside a bf16 autocast region (the HF DINOv3 members apply RoPE in fp32; autocast then casts q / k / v to bf16)
        if isinstance(q, torch.Tensor) and q.is_cuda and (q.dtype is torch.bfloat16 or (q.dtype is torch.float32 and torch.is_autocast_enabled('cuda')
                                                                                         and torch.get_autocast_dtype('cuda') is torch.bfloat16)):
            STATS['emu_sdpa'] = STATS.get('emu_sdpa', 0) + 1
            with torch.autocast('cuda', enabled=False):
                return _sdpa(rb(q), rb(k), rb(v), *[rb(x) for x in a], **{n: rb(x) for n, x in kw.items()}).to(torch.bfloat16)
        return _sdpa(q, k, v, *a, **kw)
    F.scaled_dot_product_attention = sdpa


def _install_seg_emu(Emu):
    """emu: CellposeModel.eval (the bf16 CPSAM net) runs under the bf16 emulation."""
    from cellpose import models
    _ev = models.CellposeModel.eval
    def ev(self, *a, **k):
        with Emu(): return _ev(self, *a, **k)
    models.CellposeModel.eval = ev


def install():
    global FP16, BASE
    autocast_mode(); det_copies()
    fl, sb =os.environ.get('CLEAR_EC_FP16_LIST'), os.environ.get('CLEAR_EC_SKIP_BASE')
    if not fl and not sb: return False
    import torch, time
    if fl: FP16 = {l.strip() for l in open(fl) if l.strip()}
    if sb: BASE = json.load(open(sb))
    _load = torch.load
    def load(f, *a, **k):
        r = rel(f) if isinstance(f, (str, os.PathLike)) else None
        if r is not None and BASE and r in BASE['torch']: return _zeros(BASE['torch'][r], torch)
        src = (os.path.join(FP16_ROOT, r) if FP16_ROOT else f) if (r is not None and r in FP16) else f
        g = RO_GROUPS.get(r) if r is not None else None
        if g is not None and g not in _RO_CLOSED and len(a) <= 1:
            kw = dict(k); ml = kw.pop('map_location', a[0] if a else None); kw.pop('mmap', None)
            mls = str(ml) if isinstance(ml, (str, torch.device)) else None
            if mls is not None:      # a named device (all callers of these files pass one); anything else takes the tier-p path below
                if r not in _RO:
                    t = time.perf_counter(); _RO[r] = _load(src, map_location='cpu', **kw); _RO_SIZE[r] = os.stat(src).st_size
                    STATS['read_bytes'] += _RO_SIZE[r]; STATS['read_s'] += time.perf_counter() - t
                    STATS['ro_peak_bytes'] = max(STATS['ro_peak_bytes'], sum(_RO_SIZE.values()))
                else: STATS['ro_hits'] += 1; STATS['ro_hit_bytes'] += _RO_SIZE[r]
                o = _ro_copy(_RO[r], torch, None if mls == 'cpu' else torch.device(mls))
                if r in FP16: STATS['fp16_loads'] += 1; return _upcast(o, torch)
                return o
        t = time.perf_counter()
        o = _load(src, *a, **k)
        if isinstance(src, (str, os.PathLike)): STATS['read_bytes'] += os.stat(src).st_size
        STATS['read_s'] += time.perf_counter() - t
        if r is not None and r in FP16: STATS['fp16_loads'] += 1; return _upcast(o, torch)
        return o
    torch.load = load
    if BASE:
        import safetensors.torch as st
        _lf = st.load_file
        def load_file(f, *a, **k):
            r = rel(f)
            if r in BASE['safetensors']: return _zeros(BASE['safetensors'][r], torch)
            t = time.perf_counter(); o = _lf(f, *a, **k); STATS['read_bytes'] += os.stat(f).st_size; STATS['read_s'] += time.perf_counter() - t; return o
        st.load_file = load_file
        if BASE.get('hf'):
            import transformers
            _fp = transformers.AutoModel.from_pretrained
            def from_pretrained(path, *a, **k):
                if rel(path) in BASE['hf']:
                    STATS['base_from_config'] += 1
                    m = transformers.AutoModel.from_config(transformers.AutoConfig.from_pretrained(path, local_files_only=True))
                    dt = k.get('dtype', k.get('torch_dtype'))
                    return (m.to(dt) if dt is not None else m).eval()
                return _fp(path, *a, **k)
            transformers.AutoModel.from_pretrained = from_pretrained
        if BASE.get('digest'):
            # phase2_foundation_preflight_20261002.build records a sha256 of the CPDINO base file in its metadata dict (weight_sha256); with the base
            # weights not shipped, return the recorded value instead of reading the file (no effect on any computation)
            import importlib.abc, importlib.util, sys as _sys
            def patch(mod):
                _dg = mod.digest
                mod.digest = lambda p: BASE['digest'][rel(p)] if rel(p) in BASE['digest'] else _dg(p)
            class PostImport(importlib.abc.MetaPathFinder):
                def find_spec(self, name, path, target=None):
                    if name != 'phase2_foundation_preflight_20261002': return None
                    _sys.meta_path.remove(self)
                    try: spec = importlib.util.find_spec(name)
                    finally: _sys.meta_path.insert(0, self)
                    if spec is None or spec.loader is None: return spec
                    ex = spec.loader.exec_module
                    def exec_module(module): ex(module); patch(module)
                    spec.loader.exec_module = exec_module; return spec
            if 'phase2_foundation_preflight_20261002' in _sys.modules: patch(_sys.modules['phase2_foundation_preflight_20261002'])
            else: _sys.meta_path.insert(0, PostImport())
    return True


ACTIVE = install()

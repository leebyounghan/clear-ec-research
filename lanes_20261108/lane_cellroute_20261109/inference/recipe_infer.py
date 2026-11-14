"""New-recipe inference entry (cell-route lane draft, 2026-11-10): one raw frame -> CD, CV, HEX.
    R = Recipe('manifests/deploy.json', 'params/deploy.json');  out = R.predict(image_u8)        # out['CD'], out['CV'], out['HEX'], out['parts']
    python recipe_infer.py --manifest deploy --images DIR --out DIR      (GC interface: one .mha/.tif in, three json files out)
Reads no labels, fold ids or OOF files: everything comes from the image, the weights named in the manifest and the parameter file
(written offline by params_build.py). Inputs whose builder or weights do not exist yet are STUBS: in deploy mode they raise MissingPiece;
the fold-0 test harness (test_fold0.py) passes stored values for them through `provided` and says so in out['stubs'].
Columns: CD = lane_cd cd_combine_20261110.combine; HEX = lane_hex hex_apply.apply, then the cell-route ridge (deploy_hexkonan.correct);
CV = lane_cv cv_apply_20261110 (apply with deploy params; fold tests use the fold's weights, recipe A scalar and stretch).
Per-image builders from other lanes: HEX features = lane_hex hex_test_features.build_arrays (single-view peaks of the 16 detectors)."""
import json, os, sys, time, argparse, tempfile
import numpy as np
sys.path.insert(0, '/workspace/clear-ec-research/experiments/lane_cellroute_20261109/inference')
import pack_hooks      # packaging hooks (fp16 copies, base weights skipped); inactive unless CLEAR_EC_* env vars are set
import members as M      # chdir to the repo root, sys.path for the lanes' modules
for p in ('experiments/lane_hex_20261108/source', 'experiments/lane_cv_20261108/source', 'experiments/submission1_test_20261030/source/phase1_app',
          'experiments/green_track_20261028/deploy', 'experiments/meta_ceiling_20261103/source'):
    if p not in sys.path: sys.path.insert(0, p)
import cd_combine_20261110 as CDC
import hex_apply as HA
import cv_apply_20261110 as CA
import deploy_hexkonan as DK
HERE = os.path.dirname(os.path.abspath(__file__))
UNLOAD = os.environ.get('CLEAR_EC_UNLOAD') == '1'    # one image per job: free every model after its stage (GPU peak = one stage)
ALL16 = ['external_srep/preonly', 'external_srep/s47_preonly', 'external_srep/s48_preonly', 'sg2_specialist'] + [f'sg2_specialist_s{s}' for s in range(47, 53)] + \
        ['sg2_sigma3p5', 'sg2_sigma5', 'frozen_heads/sg2f_s46', 'frozen_heads/sg2f_s47', 'frozen_heads/sg2f_s48', 'frozen_heads/sg2pref_s46']   # feat_fuse.SETS['all16']


class MissingPiece(RuntimeError):
    pass


def image_stats(img):
    """eval_oof_20261031.image_stats on one frame: half-size raw frame mean, SD, share < 5, Laplacian variance."""
    a = np.asarray(img[::2, ::2], dtype=np.float32); lap = a[1:-1, 1:-1] * 4 - a[:-2, 1:-1] - a[2:, 1:-1] - a[1:-1, :-2] - a[1:-1, 2:]
    return np.array([a.mean(), a.std(), np.mean(a < 5), lap.var()])


class Recipe:
    def __init__(self, manifest, params):
        self.m = json.load(open(manifest if os.path.isabs(manifest) else os.path.join(HERE, manifest)))
        self.p = json.load(open(params if os.path.isabs(params) else os.path.join(HERE, params)))
        self.crp = json.load(open(os.path.join(HERE, self.p['cellroute_params'])))
        try: self.cvp = CA.load_params(self.p['cv']['deploy_dir']) if self.p['cv'].get('deploy_dir') else self.p['cv']
        except FileNotFoundError: self.cvp = None      # CV deploy params not written yet
        self._seg = self._p1net = self._hk = self._h6 = None; self._other = {}; self._gn = {}; self._sim = self._fh = None

    # ---------- image-level builders ----------
    def seg(self, img):
        import cv2
        from cellpose import models
        import seg_run_feats as SR    # seg_run.feats copied verbatim (seg_run.py starts a run at import)
        if self._seg is None:
            kw = dict(use_bfloat16=False) if pack_hooks.AUTOCAST == 'fp16' else {}     # T4 path: no bf16 (cellpose then runs float32)
            self._seg = models.CellposeModel(gpu=(M.DEV == 'cuda'), pretrained_model=self.m['cpsam_seg'], **kw)
        im = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(np.ascontiguousarray(img))
        mask = np.asarray(self._seg.eval([im], batch_size=8, flow_threshold=0.8, cellprob_threshold=-2.0)[0][0], np.int32)
        return np.asarray(SR.feats(mask), float), DK.seg_points(mask)

    def p1(self, img):
        """Phase-1 five members (candidate5 package), central 512 crop, mean x the OOF SCALE (CD lane 'Deployment must use')."""
        from final_candidate_core import build_model, central512, predict_one
        from base_inference import load_strict
        if self._p1net is None: self._p1net = build_model(self.m['p1_cpsam'], device=M.DEV)
        c = central512(img); v = []
        for path in self.m['p1_members']:
            load_strict(self._p1net, path); v.append(predict_one(self._p1net, c, device=M.DEV))
        return np.mean(v, 0) * np.asarray(self.p['P1_scale'])

    def peaks(self, img):
        import deploy_hexkonan_image as DI
        if self._hk is None: self._hk = DI.HexKonan(self.m['preonly_detectors'], os.path.join(HERE, self.p['cellroute_params']), device=M.DEV)
        return self._hk.peaks(img, single=True)       # (mean-TTA peaks, [TTA peaks s46/47/48], [single-view peaks s46/47/48])

    def other_peaks(self, img):
        """single-view peaks of the 13 non-pre-only detectors of all16 (manifest 'detectors_all16': name -> [kind, path])."""
        import deploy_hexkonan_image as DI
        out = {}
        for d in ALL16[3:]:
            kind, path = self.m['detectors_all16'][d]
            if d not in self._other: self._other[d] = DI.load_any(kind, path, M.DEV, self.m.get('frozen_encoder', DI.load_any.__defaults__[0]))
            out[d] = DI.single_peaks(*self._other[d], img)
            if UNLOAD: self._other.pop(d, None); self._empty()
        return out

    def hex_features(self, img, single3, other, cd_hybB):
        """HEX lane hex_test_features.build_arrays, then its full-data graph nets (gnn_hex_infer.predict) on the graphs it writes:
        manifest hex_graph_nets {key: weight name} in hex_graph_models; keys pre47_s10 / pre48_s10 (g_s10x3) and hex46 (g_hex46)."""
        import hex_test_features as HT
        peaks = {d: [single3[k]] for k, d in enumerate(ALL16[:3])}; peaks.update({d: [other[d]] for d in ALL16[3:]})
        argv = sys.argv; sys.argv = ['feat_fuse.py', '/dev/null', 'all16']     # the HEX feature modules read sys.argv at import
        gout = {}
        try:
            with tempfile.TemporaryDirectory() as td:
                F, G = HT.build_arrays(np.asarray(img, np.uint8)[None], peaks, np.array([cd_hybB]), td, workers=1)
                if self.m.get('hex_graph_nets'):
                    import glob, gnn_hex_infer as GI
                    for key, (name, gset) in self.m['hex_graph_nets'].items():
                        if glob.glob(os.path.join(self.m['hex_graph_models'], f'{name}_s*_e*.pt')): gout[key] = np.asarray(GI.predict(name, f'{td}/g_{gset}.npz', models=self.m['hex_graph_models'], dev=M.DEV))[0]
        finally: sys.argv = argv
        return F, G, gout

    def cv_parts(self, x, pk):
        """CV lane cv_image_parts_20261110 (label-free): GRAPH ensemble on single-view graphs (noctx <- s46, pre47 <- s47, pre48 <- s48), plain and
        order quartets on the 4-flip graphs (s46/s47/s48 tta4, seeds3 tta4), sim2_l8 embeddings -> CV and HEX ridge, frozen heads on the recipe A
        s46 tokens. Returns the 3-column GRAPH / quartet outputs (log residuals vs H), the ridge corrections and the frozen-head log CVs."""
        import cv_image_parts_20261110 as CVP
        gs = {s: CVP.graphs_from_peaks(pk[2][k], 'prep3') for k, s in enumerate(CVP.GSETS)}
        gq = {s: CVP.graphs_from_peaks((pk[1][k] if s != 'seeds3' else pk[0]), 'sp') for k, s in enumerate(CVP.QSETS)}
        def net(pat):
            if pat not in self._gn: self._gn[pat] = CVP.GraphNets(pat)
            return self._gn[pat]
        Gs = {s: np.asarray(net(f'gfull_{s}_s*_s*_e*.pt').out(gs[s]), float) for s in CVP.GSETS}; G = np.mean([Gs[s] for s in CVP.GSETS], 0)
        QP = np.mean([np.asarray(net(f'full_tta_{s}_tta4_s*_e*.pt').out(gq[s]), float) for s in CVP.QSETS], 0)
        QO = np.mean([np.asarray(net(f'gord_{s}_tta4_full_s*_s*_e*.pt').out(gq[s]), float) for s in CVP.QSETS], 0)
        if self._sim is None: self._sim = CVP.SimNets()
        embs = [np.asarray(e, float).reshape(1, -1) for e in self._sim.embed(gs)]
        rid = {c: float(CA.ridge(self.cvp, embs, c)[0]) for c in ('CV', 'HEX')}
        if self._fh is None: self._fh = CVP.FrozenHeads()
        fz = self._fh.logcv(CVP.token_blocks(M.recipeA_tokens(x, self.m['cv_parts']['token_encoder']), dev=M.DEV))
        return dict(G=G, Gs=Gs, QP=QP, QO=QO, ridge=rid, fz={k: float(np.ravel(v)[0]) for k, v in fz.items()})

    def hex6(self, img):
        """f6 / have: the deployed submission-2 hex6 corrector's feature (mean log hex6 over its active detectors); GPU only (its forward calls .cuda())."""
        from hex6_correction import Hex6Corrector
        if self._h6 is None: self._h6 = Hex6Corrector(self.m['hex6_deploy_dir'], device=M.DEV)
        feat, _ = self._h6.feature(img)
        return (float(feat) if feat is not None else 0.0), feat is not None

    # ---------- main ----------
    def predict(self, img, provided=None, skip=()):
        """img: raw (972, 1296) uint8. provided: dict of stored input values (test harness only). skip: builders not to run (use provided)."""
        provided = provided or {}; stubs, parts, t0 = [], {}, time.time(); m = self.m

        def need(name, fn, ok):
            if name in skip or not ok:
                if name in provided: stubs.append(name); return provided[name]
                raise MissingPiece(name)
            r = fn(); self._stage_end(); return r
        x = M.preprocess(img); img = np.asarray(img, np.uint8)
        parts['A'] = need('A', lambda: M.recipeA_tta(x, m['recipeA_tta'], self.p['A_scalars']), bool(m.get('recipeA_tta')))
        parts['A_cd'] = need('A_cd', lambda: M.recipeA_cd(x, m['recipeA_cd'], self.p['recipeA_cd_scalar']), bool(m.get('recipeA_cd'))) if not self.cd_tier() else np.nan
        parts['D'] = need('D', lambda: M.dino_cd(x, m['dino_cd'])[0], bool(m.get('dino_cd'))) if self.cd_weight('D') else 1.0     # weight 0: not computed, w * log 1 = 0
        # CV-only DINOv3 members run only when the loaded CV stack weights them (2026-11-11: the CV freeze dropped orderdino, re-added dino)
        parts['DINO_logcv'] = need('DINO_logcv', lambda: float(np.log(M.dino_cv(x, m['dino_cv'])[1])), bool(m.get('dino_cv'))) if self.cv_weight('dino') else np.nan
        parts['orderdino_logcv'] = need('orderdino_logcv', lambda: float(M.orderdino_cv(x, m['orderdino_cv'])[1]), bool(m.get('orderdino_cv'))) if self.cv_weight('orderdino') else np.nan
        vs = m.get('vitl_seeds') or []
        vok = bool(vs) and all(s.get('tta') for s in vs) and (all(s.get('single') for s in vs) or 'V_single_logs' in provided)
        if not self.cd_weight('V'): parts['V'] = 1.0
        elif self.cd_tier(): parts['V'] = need('V', lambda: M.vitl_tta_only(x, vs)[0], bool(vs) and all(s.get('tta') for s in vs))
        else: parts['V'] = need('V', lambda: M.vitl_cd(x, vs, provided.get('V_single_logs'))[0], vok)
        parts['P1'] = need('P1', lambda: self.p1(img), bool(m.get('p1_members')))
        sg = need('seg', lambda: self.seg(img), bool(m.get('cpsam_seg'))); parts['seg17'], segpts = (sg if isinstance(sg, tuple) else (sg, None))
        parts['img'] = image_stats(img)
        pk = need('peaks', lambda: self.peaks(img), bool(m.get('preonly_detectors'))); parts['peaks'] = pk
        if self.hex_needs('others13'):
            other = need('peaks_other13', lambda: self.other_peaks(img), bool(m.get('detectors_all16')) and all(os.path.exists(v[1]) for v in m['detectors_all16'].values()))
        else: other = {d: np.zeros((0, 3), np.float32) for d in ALL16[3:]}      # HEX tier without ridge_fall16: the 13 other detectors are not run
        cd_hybB = 0.5 * parts['A'][0] + 0.5 * parts['P1'][0]        # submission-1 hybrid-B CD (the HEX prior's spacing), not the new CD
        F, G, gout = self.hex_features(img, pk[2], other, cd_hybB); self._stage_end(); parts['hex_graph_out'] = gout; parts['hexfeat'] = {k: np.asarray(v)[0] for k, v in F.items() if k in ('img', 'prior_h6c', 'ridge_k3', 'ridge_fall16_h6_s15')}
        for k in ('prior_h6c', 'ridge_k3', 'ridge_fall16_h6_s15'): parts[k] = parts['hexfeat'].get(k, np.nan)
        if m.get('hex6_deploy_dir') and 'f6' not in skip: parts['f6'], parts['have'] = self.hex6(img); self._stage_end()
        else: parts['f6'] = need('f6', lambda: None, False); parts['have'] = need('have', lambda: None, False)
        # CV per-image parts (also HEX's graph / best): full-data weights only, so stubs in fold tests
        if m.get('cv_parts') and 'cv_parts' not in skip:
            cp = self.cv_parts(x, pk); parts['cv_parts'] = cp; self._stage_end()
            parts.update(graph_cv=cp['G'][1], ridge_cv=cp['ridge']['CV'], qplain_cv=cp['QP'][1], qorder_cv=cp['QO'][1], graph=cp['G'][2], best=cp['G'][2] + cp['ridge']['HEX'], **cp['fz'])
        else:
            for name in ('graph_cv', 'ridge_cv', 'qplain_cv', 'qorder_cv', 'fz_std', 'fz_spectrum', 'fz_cov_sqrt', 'fz_autocorr', 'graph', 'best'): parts[name] = need(name, lambda: None, False)
        # HEX graph nets (g_s10x3 = mean HEX column of the CV lane's gfull_noctx and the HEX lane's full_d64_l5_pre47_s10 / pre48_s10; g_hex46) and the
        # cvw3 / CPSAM 512 members (exp mean log over epochs 8-12, HEX column)
        ok_s10 = 'cv_parts' in parts and all(k in gout for k in ('pre47_s10', 'pre48_s10'))
        parts['g_s10x3'] = need('g_s10x3', lambda: float(np.mean([parts['cv_parts']['Gs']['noctx'][2], gout['pre47_s10'][2], gout['pre48_s10'][2]])), ok_s10)
        parts['g_hex46'] = need('g_hex46', lambda: float(gout['hex46'][2]), 'hex46' in gout) if self.hex_needs('g_hex46') else np.nan   # dropped by the 11-11 seed audit
        fx = lambda key: bool(m.get(key)) and all(os.path.exists(q) for q in m[key])
        parts['dino_cvw3'] = need('dino_cvw3', lambda: float(M.family_mean(x, m['cvw3'], 'dinov3_b16')[2]), fx('cvw3')) if self.hex_needs('dino_cvw3') else np.nan
        parts['cps5'] = need('cps5', lambda: float(M.family_mean(x, m['cps5'], 'cpsam_c512')[2]), fx('cps5')) if self.hex_needs('cps5') else np.nan
        # ---- CD ----
        seg17 = np.asarray(parts['seg17'], float)
        CD = self.cd_column(parts)
        # ---- HEX ----
        a1 = lambda v: np.array([v], float)
        I = dict(A=a1(parts['A'][2]), P1=a1(parts['P1'][2]), f6=a1(parts['f6']), have=np.array([parts['have']], bool), best=a1(parts['best']), graph=a1(parts['graph']),
                 g_s10x3=a1(parts['g_s10x3']), g_hex46=a1(parts['g_hex46']), img=parts['img'][None], prior_h6c=a1(parts['prior_h6c']), ridge_k3=np.asarray(parts['ridge_k3'], float)[None],
                 ridge_fall16_h6_s15=a1(parts['ridge_fall16_h6_s15']), seg=seg17[None], dino_cvw3=a1(parts['dino_cvw3']), cps5=a1(parts['cps5']))
        for key, spec in ((k, v) for k, v in (m.get('hex_extra') or {}).items() if self.hex_block(k)):      # optional extra HEX members (e.g. a later attention-pool DINOv3-B column): manifest-driven
            I[key] = a1(float(getattr(M, spec.get('read', 'family_mean'))(x, spec['ckpts'], spec['fam'])[spec.get('col', 2)])); self._stage_end()
        hx = self.hex_chain(I)
        cr = need('cellroute_feats', lambda: DK.features(pk[0], pk[1][0], pk[1][1], pk[1][2], segpts), segpts is not None and pk is not None)
        parts['cellroute_feats'] = cr; parts['hex_chain'] = float(hx[0]); HEX = float(DK.correct(hx, np.asarray(cr, float)[None], self.crp)[0])
        # ---- CV ----
        CV = self.cv_column(parts) if self.cvp is not None else need('CV_final', lambda: None, False)
        return dict(CD=CD, CV=CV, HEX=HEX, parts=parts, stubs=stubs, seconds=round(time.time() - t0, 1))

    def _empty(self):
        import gc, torch
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    def _stage_end(self):
        """CLEAR_EC_UNLOAD=1: drop every cached model after a stage (rebuilt if a later stage needs the same family)."""
        if not UNLOAD: return
        M.Member._cache.clear(); self._seg = self._p1net = self._hk = self._h6 = None; self._other.clear(); self._gn.clear(); self._sim = self._fh = None
        self._empty()

    def hex_chain(self, I):
        """HEX before the cell-route step: lane_hex hex_apply.hex_any on the embedded lane params (joint ridge or step chain; fold None = deploy),
        or the older embedded step list."""
        h = self.p['hex']
        return HA.hex_any(h['P'], I, h.get('fold')) if 'P' in h else HA.apply(h['steps'], h['stretch'], I)

    def hex_block(self, block):
        """Whether the loaded joint HEX ridge has this column block (tier files list 'blocks'; the 39-column freeze file lists 'columns' as 'name(n)')."""
        P = self.p['hex'].get('P') or {}
        names = P.get('blocks') or [c.split('(')[0] for c in P.get('columns', [])]
        return block in names

    def hex_needs(self, key):
        """Whether the loaded HEX form reads this input (g_hex46: the joint ridge's gnn46 block, or a step chain's gnn:hex_pre46_s5)."""
        h = self.p['hex']; P = h.get('P')
        if P is not None and str(P.get('kind', '')).startswith('joint'):
            return self.hex_block({'g_hex46': 'gnn46', 'cps5': 'mtcnn5', 'dino_cvw3': 'cvw3_mean', 'others13': 'ridge_fall16'}.get(key, key)) if key in ('g_hex46', 'cps5', 'dino_cvw3', 'others13') else True
        steps = P['steps'] if P is not None else h['steps']
        return key != 'g_hex46' or any(q['step'] == 'gnn:hex_pre46_s5' for q in steps)

    def cd_tier(self):
        """True for the packaging-tier CD params (CD lane freeze_cd_params_t{1,2,3}_20261112): P1 x p1_cd_multiplier, A = recipe A e20 TTA only."""
        return 'p1_cd_multiplier' in self.p['cd']

    def cd_weight(self, name):
        return float(self.p['cd']['weights'].get(name, 0.0))

    def cd_column(self, parts):
        """CD = lane_cd cd_combine_20261110.combine on one image's member values (P1 CD, A_cd, D, V, CPSAM mask CD = seg17[5], mask cell count = seg17[3]).
        Tier params: P1_in = P1 x p1_cd_multiplier, A_in = 0.5 (s46 + s47 e20 TTA) x recipeA_cd_scalar (= parts['A'][0] rescaled to that scalar)."""
        seg17 = np.asarray(parts['seg17'], float)
        if self.cd_tier():
            c = self.p['cd']; a_in = parts['A'][0] / self.p['A_scalars'][0] * c['recipeA_cd_scalar']
            return float(CDC.combine(parts['P1'][0] * c['p1_cd_multiplier'], a_in, parts['D'], parts['V'], seg17[5], seg17[3], c))
        return float(CDC.combine(parts['P1'][0], parts['A_cd'], parts['D'], parts['V'], seg17[5], seg17[3], self.p['cd']))

    def cv_weight(self, comp):
        """Weight of a CV stack component in the loaded CV params (0 when absent or when no CV params are loaded)."""
        return float(((self.cvp or {}).get('weights') or {}).get(comp, 0.0))

    def cv_column(self, parts):
        """CV = lane_cv cv_apply_20261110.apply on one image's member values. parts['A'][1] and parts['P1'][1] already carry the recipe A and
        Phase-1 scalars of this parameter set, so H_cv / A_cv are passed directly (h_cv, a_cv). Components with weight 0 are never read."""
        a1 = lambda v: np.array([v], float)
        mcv = dict(h_cv=a1(0.5 * parts['A'][1] + 0.5 * parts['P1'][1]), a_cv=a1(parts['A'][1]), graph_cv=a1(parts['graph_cv']), ridge_cv=a1(parts['ridge_cv']),
                   qplain_cv=a1(parts['qplain_cv']), qorder_cv=a1(parts['qorder_cv']), dino_logcv=a1(parts['DINO_logcv']), orderdino_logcv=a1(parts['orderdino_logcv']),
                   fz_std=a1(parts['fz_std']), fz_spectrum=a1(parts['fz_spectrum']), fz_cov_sqrt=a1(parts['fz_cov_sqrt']), fz_autocorr=a1(parts['fz_autocorr']))
        return float(CA.apply(self.cvp, mcv)[0])


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--manifest', default='deploy'); ap.add_argument('--images', default='/input/images/corneal-specular-microscopy-image')
    ap.add_argument('--out', default='/output'); a = ap.parse_args()
    import glob, SimpleITK as sitk
    R = Recipe(f'manifests/{a.manifest}.json', f'params/{a.manifest}.json')
    files = sorted(glob.glob(os.path.join(a.images, '*.mha')) + glob.glob(os.path.join(a.images, '*.tif')) + glob.glob(os.path.join(a.images, '*.tiff')))
    if not files: raise FileNotFoundError(a.images)
    im = sitk.GetArrayFromImage(sitk.ReadImage(files[0])).astype(np.uint8)
    if im.ndim == 3 and im.shape[0] in (1, 3): im = im[0]
    o = R.predict(im); os.makedirs(a.out, exist_ok=True)
    for k, fn in (('CD', 'cell-density.json'), ('CV', 'coefficient-of-variation.json'), ('HEX', 'hexagonality.json')):
        json.dump(float(o[k]), open(os.path.join(a.out, fn), 'w'), indent=4)


if __name__ == '__main__':
    main()

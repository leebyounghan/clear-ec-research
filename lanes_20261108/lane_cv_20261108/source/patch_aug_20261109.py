p = '/workspace/clear-ec-research/experiments/lane_cv_20261108/source/family_cv_20261108.py'; s = open(p).read()
anchor = "LANE = Path('experiments/lane_cv_20261108')"
patch = '''# CV lane: AUG_BUNDLE=1 adds the label-preserving perturbations of recipeA_invariance_20261104 during training (raw frame, before CLAHE):
# rotation +-10 deg (reflected border), gamma log-uniform [0.8, 1.25], Gaussian blur sigma [0.3, 2.0] with p 0.5, Gaussian noise sigma [1, 7] with p 0.5; flips as before.
if os.environ.get('AUG_BUNDLE'):
    import cv2 as _cv2
    def _perturb(a, rng, W=1296, H=972):
        ang = rng.uniform(-10.0, 10.0); a = _cv2.warpAffine(a, _cv2.getRotationMatrix2D(((W - 1) / 2, (H - 1) / 2), ang, 1.0), (W, H), flags=_cv2.INTER_LINEAR, borderMode=_cv2.BORDER_REFLECT_101)
        g = np.exp(rng.uniform(np.log(0.8), np.log(1.25))); a = _cv2.LUT(a, np.clip(255 * (np.arange(256) / 255.0) ** g, 0, 255).astype(np.uint8))
        if rng.random() < 0.5: a = _cv2.GaussianBlur(a, (0, 0), rng.uniform(0.3, 2.0))
        if rng.random() < 0.5: a = np.clip(a.astype(np.float32) + rng.normal(0, rng.uniform(1.0, 7.0), a.shape).astype(np.float32), 0, 255).astype(np.uint8)
        return a
    class AugFrames(R.Frames):
        def __getitem__(self, row):
            if not self.use_augmentation: return super().__getitem__(row)
            _cv2.setNumThreads(1)
            if self.raw is None: self.raw = np.memmap(b.RAW, mode='r', dtype=np.uint8, shape=(self.total, 972, 1296))
            index = int(self.indices[row]); a = _perturb(np.asarray(self.raw[index]), np.random.default_rng(np.random.SeedSequence([self.seed, 88, self.epoch, index])))
            a = _cv2.createCLAHE(2.0, (1296 // 32, 972 // 32)).apply(a); lo, hi = np.percentile(a, (1, 99)); a = np.clip((a.astype(np.float32) - lo) / (hi - lo + 1e-6), 0, 1)
            x = torch.from_numpy(np.ascontiguousarray(a[None])); h, v = R.flip_decisions(self.seed, self.epoch, index)
            if h: x = torch.flip(x, (-1,))
            if v: x = torch.flip(x, (-2,))
            return x, torch.from_numpy(self.targets[index]), index
    R.Frames = AugFrames

'''
assert anchor in s and 'AUG_BUNDLE' not in s; s = s.replace(anchor, patch + anchor, 1); open(p, 'w').write(s); print('patched')

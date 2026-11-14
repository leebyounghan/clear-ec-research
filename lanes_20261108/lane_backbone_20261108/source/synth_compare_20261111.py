"""Side-by-side real vs synthetic raw crops (256 px, native scale): top row 2 P1 + 2 P2 real frames (centre crops), bottom row 4
synthetic frames (synth_specular_20261111.render). Writes tmp/synth_sample/compare.png (and compare_more_real.png: 8 real crops)."""
import sys, os
os.chdir('/workspace/clear-ec-research'); sys.path[:0] = ['experiments/oof_recipeA_20261027/staging_regression_zero', 'experiments/phase2_crossfold_20261015/source', 'experiments/lane_backbone_20261108/source']
import numpy as np, cv2
import phase2_foundation_train_20261002 as b
import synth_specular_20261111 as G
with np.load(b.META, allow_pickle=True) as z:
    n = len(z['ids']); Y = z['Y']
raw = np.memmap(b.RAW, mode='r', dtype=np.uint8, shape=(n, 972, 1296))
sp = np.load('experiments/lane_backbone_20261108/data/source_split_20261110.npz', allow_pickle=True)
out = 'experiments/lane_backbone_20261108/tmp/synth_sample/'
def crop(a, y0=358, x0=520):
    return np.asarray(a[y0:y0 + 256, x0:x0 + 256])
def lab(img, t):
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR); cv2.putText(img, t, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA); return img
real = []
for nm, P, picks in (('P1', sp['P1'], (100, 2000)), ('P2', sp['P2'], (100, 2000))):
    for k in picks:
        i = int(P[k]); real.append(lab(crop(raw[i]), f'real {nm} #{i} CD{Y[i,0]:.0f} CV{Y[i,1]:.2f} HX{Y[i,2]:.2f}'))
syn = []
for i in (0, 1, 4, 5):
    img, lb, prm = G.render(i); syn.append(lab(crop(img), f'synth {i} pol{prm["polarity"]} CD{lb[0]:.0f} CVa{lb[1]:.2f} HX{lb[2]:.2f}'))
name = sys.argv[1] if len(sys.argv) > 1 else 'compare.png'
cv2.imwrite(out + name, np.vstack([np.hstack(real), np.hstack(syn)]))
more = []
rng = np.random.default_rng(5)
for nm, P in (('P1', sp['P1']), ('P2', sp['P2'])):
    for i in rng.choice(P, 4, replace=False):
        more.append(lab(crop(raw[int(i)]), f'real {nm} #{int(i)}'))
cv2.imwrite(out + 'compare_more_real.png', np.vstack([np.hstack(more[:4]), np.hstack(more[4:])]))
print('wrote', out + name, out + 'compare_more_real.png')

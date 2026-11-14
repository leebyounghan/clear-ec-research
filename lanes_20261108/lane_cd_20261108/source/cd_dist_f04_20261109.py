"""Distribution-head DINOv3 (TTA over e16-20) on folds 0 and 4, two-fold cross-fit stack (cd_vitl47_f04.stack2 protocol, with mask), against the
recipe-loss DINOv3 s46 TTA; ViT-L = seed mean (s46, s47)."""
import os, numpy as np
os.environ['A2SNAP'] = '1'
import cd_vitl47_f04_20261109 as V4
import cd_read_20261108 as R
from cd_common import *
Dr = R.member('dinov3_s46_tta'); Dd = R.member('dinov3_dist_s46_tta'); V = R.member('vitl_s4647_tt')
print('members alone, folds 0,4: recipe', round(cd(Dr, SCREEN), 3), 'dist', round(cd(Dd, SCREEN), 3), flush=True)
a = V4.stack2([Dr, V], 'recipe DINOv3 TTA + ViT-L mean + mask (folds 0,4)')
b = V4.stack2([Dd, V], 'dist DINOv3 TTA in place of recipe')
c = V4.stack2([Dr, Dd, V], 'recipe + dist + ViT-L')
rep('dist in place vs recipe', b, a, rows=SCREEN); rep('dist added vs recipe', c, a, rows=SCREEN)

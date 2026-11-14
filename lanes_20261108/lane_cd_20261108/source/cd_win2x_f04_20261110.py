"""Backbone lane's recipe A on the 2x technician-window crop (folds 0, 4; read-only) in the CD stack, two-fold cross-fit with mask."""
import os, numpy as np
os.environ['A2SNAP'] = '1'
import cd_vitl47_f04_20261109 as V4
import cd_read_20261108 as R
from cd_common import *
D = R.member('dinov3_s46_tta'); V = R.member('vitl_s4647_tt')
for k in ('win2x_s46', 'win2x_s46_tt'):
    W = R.member(k); print(k, 'alone CD folds 0,4', round(cd(W, SCREEN), 3), flush=True)
W = R.member('win2x_s46_tt')
a = V4.stack2([D, V], 'stack (DINOv3 + ViT-L mean) + mask, folds 0,4')
b = V4.stack2([D, V, W], 'stack + win2x recipe A + mask')
rep('win2x added vs stack', b, a, rows=SCREEN)

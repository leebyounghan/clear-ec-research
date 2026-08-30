"""Fine-tune Cellpose-SAM on the 25 SAM-pipeline pseudo-masks (image+instance tif pairs).
Literature: ~15-25 masks suffice for domain adaptation. Non-circular: masks derive from SAM
boundaries + GT points, not from a cellpose/fNLA teacher. Output model -> cpsam_pseudo25.
"""
import os, numpy as np, tifffile
from glob import glob
from cellpose import models, train as cptrain
ROOT="datasets/clear_ec_official"; SRC=os.environ.get("SRC",f"{ROOT}/sam_pseudomasks")
EP=int(os.environ.get("EPOCHS","150")); BS=int(os.environ.get("BS","1")); LR=float(os.environ.get("LR","1e-5"))
NAME=os.environ.get("NAME","cpsam_pseudo25")
imgs=sorted(glob(f"{SRC}/images/*.tif"));
timg=[]; tlab=[]
for ip in imgs:
    ID=os.path.basename(ip)[:-4]; mp=f"{SRC}/masks/{ID}.tif"
    if not os.path.exists(mp): continue
    im=tifffile.imread(ip).astype(np.float32); lb=tifffile.imread(mp).astype(np.int32)
    if lb.max()>=5: timg.append(im); tlab.append(lb)
print(f"train pairs: {len(timg)}  (mask counts {[int(l.max()) for l in tlab][:5]}...)",flush=True)
base=models.CellposeModel(gpu=True, pretrained_model="cpsam_models/cpsam_v2")
cptrain.train_seg(base.net, train_data=timg, train_labels=tlab,
    batch_size=BS, learning_rate=LR, n_epochs=EP, weight_decay=0.1, normalize=True,
    rescale=False, bsize=256, min_train_masks=5, save_path=ROOT, model_name=NAME)
print(f"DONE -> {ROOT}/models/{NAME}",flush=True)

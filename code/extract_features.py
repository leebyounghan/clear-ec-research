"""Extract frozen Cellpose-SAM encoder features for all training images."""
import numpy as np, glob, os, csv, torch, cv2, SimpleITK as sitk
from cellpose import models, transforms
ROOT="datasets/clear_ec_official"; MHA=f"{ROOT}/train_mha"
gt={r["ID"]:r for r in csv.DictReader(open(f"{ROOT}/final_train_ids.csv"))}
files=sorted(glob.glob(f"{MHA}/*.mha"))
ids=[os.path.splitext(os.path.basename(f)[:])[0].replace('.mha','') for f in files]
ids=[os.path.basename(f).replace('.mha','') for f in files]
files=[f for f,i in zip(files,ids) if i in gt]; ids=[i for i in ids if i in gt]
print("images with labels:",len(files))
m=models.CellposeModel(gpu=True, pretrained_model="cpsam_models/cpsam_v2")
enc=m.net.encoder; enc.eval(); dev=m.device
def load(f):
    im=sitk.GetArrayFromImage(sitk.ReadImage(f)).astype(np.float32)  # (H,W)
    im=transforms.normalize99(im)                                    # ~[0,1]
    im=cv2.resize(im,(256,256),interpolation=cv2.INTER_AREA)
    return np.stack([im]*3,0)                                        # (3,256,256)
BS=32; feats=[]
with torch.no_grad():
    for s in range(0,len(files),BS):
        batch=np.stack([load(f) for f in files[s:s+BS]])
        x=torch.from_numpy(batch).to(dev).to(torch.bfloat16)
        fmap=enc(x).float()                                          # (B,256,32,32)
        gap=fmap.mean((2,3)); gmp=fmap.amax((2,3))                   # (B,256) each
        feats.append(torch.cat([gap,gmp],1).cpu().numpy())
        if s%1600==0: print(f"{s}/{len(files)}")
feats=np.concatenate(feats,0).astype(np.float32)
Y=np.array([[float(gt[i]["CD"]),float(gt[i]["CV"]),float(gt[i]["HEX"])] for i in ids],np.float32)
np.savez(f"{ROOT}/cpsam_feats.npz", feats=feats, Y=Y, ids=np.array(ids))
print("saved features:",feats.shape,"Y:",Y.shape)

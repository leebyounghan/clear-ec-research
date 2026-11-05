"""Fine-tune micro-sam (default vit_b, fits 24GB) starting from *_lm weights on Rotterdam corneal
endothelium (305 img + Voronoi masks) to domain-adapt toward specular. Train on Rotterdam (external),
eval stays on the 25 CLEAR-EC green (no leakage)."""
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES","0")
from glob import glob
from micro_sam.training import train_sam, default_sam_loader
ROOT="datasets/clear_ec_official"; RT=f"{ROOT}/rotterdam_train"
MT=os.environ.get("MT","vit_b"); BS=int(os.environ.get("BS","1")); NOBJ=int(os.environ.get("NOBJ","10")); EP=int(os.environ.get("EP","12"))
imgs=sorted(glob(f"{RT}/images/*.tif")); masks=sorted(glob(f"{RT}/masks/*.tif")); ntr=len(imgs)-25
print(f"model {MT} train {ntr} val 25 bs{BS} nobj{NOBJ}",flush=True)
common=dict(raw_key=None,label_key=None,patch_shape=(256,256),with_segmentation_decoder=True,min_size=20)
train_loader=default_sam_loader(raw_paths=imgs[:ntr],label_paths=masks[:ntr],batch_size=BS,n_samples=ntr*2,is_train=True,**common)
val_loader=default_sam_loader(raw_paths=imgs[ntr:],label_paths=masks[ntr:],batch_size=1,n_samples=25,is_train=False,**common)
CKPT=os.path.expanduser(f"~/.cache/micro_sam/models/{MT}_lm")
FR=os.environ.get("FREEZE"); FR=FR.split(",") if FR else None
NAME=os.environ.get("NAME",f"cornea_{MT}")
train_sam(name=NAME, model_type=MT, checkpoint_path=CKPT, freeze=FR,
    train_loader=train_loader, val_loader=val_loader, n_epochs=EP, early_stopping=5,
    n_objects_per_batch=NOBJ, with_segmentation_decoder=True, device="cuda",
    save_root=f"{ROOT}/micro_sam_ft")
print(f"DONE -> micro_sam_ft/checkpoints/cornea_{MT}/best.pt",flush=True)

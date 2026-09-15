"""External-only native-scale masked EMA feature prediction; not official I-JEPA."""
import argparse
import copy
import fcntl
import hashlib
import json
import math
from pathlib import Path
import random
import time

import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from cellpose import models
import torch.utils.checkpoint as checkpoint

from final_candidate_core import normalize, save_json

ROOT=Path("experiments/region_search_20260905")
SOURCES=("srep_topcon_sp3000","rodrep_confocal_endothelium")


def native_crop(image):
    """Reflect-pad small fields; never rescale cell size."""
    h,w=image.shape
    if min(h,w)<256:
        image=np.pad(image,((0,max(0,256-h)),(0,max(0,256-w))),mode="reflect")
    h,w=image.shape; r=np.random.randint(h-255); c=np.random.randint(w-255)
    tile=image[r:r+256,c:c+256]
    v=np.random.randint(8); tile=np.rot90(tile,v%4)
    return np.ascontiguousarray(tile[:,::-1] if v>=4 else tile)


def block_mask(size=32):
    mask=np.zeros((size,size),np.float32)
    target=int(np.random.uniform(.40,.60)*size*size)
    while mask.sum()<target:
        h,w=np.random.randint(4,11,size=2); r=np.random.randint(size-h+1); c=np.random.randint(size-w+1)
        mask[r:r+h,c:c+w]=1
    return mask


class Online(nn.Module):
    def __init__(self,encoder):
        super().__init__(); self.encoder=encoder
        self.predictor=nn.Sequential(nn.Conv2d(256,512,3,padding=1),nn.GroupNorm(32,512),nn.GELU(),nn.Conv2d(512,256,3,padding=1))
    def forward(self,x):
        return self.predictor(self.encoder(x))


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--smoke",action="store_true"); args=parser.parse_args()
    out=ROOT/("pretrain_smoke" if args.smoke else "pretrain"); out.mkdir(parents=True,exist_ok=True)
    with (out/"train.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (out/"complete.json").exists():
            print("already complete",flush=True); return
        if (out/"config.json").exists():
            raise RuntimeError("Partial pretrain exists; preserve and inspect before restart")
        seed=20260905; random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        cv2.setNumThreads(1); torch.set_num_threads(4)
        torch.backends.cuda.matmul.allow_tf32=True; torch.backends.cudnn.allow_tf32=True; torch.set_float32_matmul_precision("high")
        manifest=Path("datasets/external/ssl_manifest.jsonl")
        entries=[json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
        groups={source:[e for e in entries if e["source"]==source] for source in SOURCES}
        if [len(groups[s]) for s in SOURCES]!=[378,52]:
            raise ValueError("Unexpected external corpus; review provenance before training")
        cache={}; hashes={}
        for source,rows in groups.items():
            cache[source]=[]
            for entry in rows:
                path=Path(entry["path"]); image=cv2.imread(str(path),cv2.IMREAD_UNCHANGED)
                if image is None: raise ValueError(f"Cannot decode {path}")
                if image.ndim==3: image=cv2.cvtColor(image[...,:3],cv2.COLOR_BGR2GRAY)
                if image.ndim!=2 or image.dtype!=np.uint8: raise ValueError(f"Unexpected external format {path}")
                cache[source].append(image)
                hashes[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
        steps=2 if args.smoke else 6000; bs=4 if args.smoke else 32
        config=dict(external_only=True,clear_ec_images_used=0,sources=list(SOURCES),source_counts={s:len(groups[s]) for s in SOURCES},
            weights=[.85,.15],steps=steps,batch=bs,seed=seed,native_scale=True,mask_range=[.4,.6],lr=1e-5,
            method="CPSAM masked EMA feature prediction with spatial predictor; JEPA-inspired, not official I-JEPA",
            manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),image_sha256=hashes,
            source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        save_json(out/"config.json",config)
        base=models.CellposeModel(gpu=True,pretrained_model="cpsam_models/cpsam_v2")
        online=Online(base.net.encoder.float()).cuda(); del base
        teacher=copy.deepcopy(online.encoder).eval()
        for p in teacher.parameters(): p.requires_grad_(False)
        for block in online.encoder.blocks:
            original=block.forward
            block.forward=(lambda fn:lambda x:checkpoint.checkpoint(fn,x,use_reentrant=False))(original)
        opt=torch.optim.AdamW(online.parameters(),lr=1e-5,weight_decay=.05)
        started=time.time(); counts={s:0 for s in SOURCES}
        for step in range(1,steps+1):
            contexts=[];targets=[];masks=[]
            for _ in range(bs):
                s=SOURCES[0] if np.random.rand()<.85 else SOURCES[1]; counts[s]+=1
                tile=normalize(native_crop(random.choice(cache[s])))
                context=np.clip(tile*np.random.uniform(.95,1.05)+np.random.uniform(-.02,.02),0,1)
                mask=block_mask(); pixel_mask=cv2.resize(mask,(256,256),interpolation=cv2.INTER_NEAREST)
                contexts.append(np.stack((context*(1-pixel_mask)+.5*pixel_mask,)*3))
                targets.append(np.stack((tile,)*3)); masks.append(mask)
            x=torch.from_numpy(np.stack(contexts)).cuda(); y=torch.from_numpy(np.stack(targets)).cuda()
            mask=torch.from_numpy(np.stack(masks)[:,None]).cuda()
            progress=(step-1)/max(steps-1,1); lr=1e-5*min(step/200,1)*(.05+.95*.5*(1+math.cos(math.pi*progress)))
            for g in opt.param_groups: g["lr"]=lr
            opt.zero_grad(set_to_none=True)
            with torch.no_grad(),torch.autocast("cuda",dtype=torch.bfloat16): target=teacher(y)
            target=F.layer_norm(target.float().permute(0,2,3,1),(256,)).permute(0,3,1,2)
            with torch.autocast("cuda",dtype=torch.bfloat16): pred=online(x)
            pred=F.layer_norm(pred.float().permute(0,2,3,1),(256,)).permute(0,3,1,2)
            loss=(F.smooth_l1_loss(pred,target,reduction="none").mean(1,keepdim=True)*mask).sum()/mask.sum()
            if not torch.isfinite(loss): raise FloatingPointError("Nonfinite pretrain loss")
            loss.backward(); torch.nn.utils.clip_grad_norm_(online.parameters(),1.,error_if_nonfinite=True);opt.step()
            ema=.996+(.9999-.996)*progress
            with torch.no_grad():
                for a,b in zip(teacher.parameters(),online.encoder.parameters()): a.mul_(ema).add_(b,alpha=1-ema)
            if step%20==0 or step==steps:
                stats=dict(step=step,target_steps=steps,loss=loss.item(),target_token_std=target.std(dim=(0,2,3)).mean().item(),
                    source_counts=counts,lr=lr,elapsed_seconds=time.time()-started,updated_at=time.strftime("%FT%T%z"))
                if not np.isfinite(stats["target_token_std"]) or stats["target_token_std"]<1e-4:
                    raise FloatingPointError("Collapsed/nonfinite target features")
                save_json(out/"progress.json",stats); print(stats,flush=True)
            if step%1000==0 or step==steps:
                temp=out/"encoder.partial.pt"
                torch.save(dict(teacher_encoder=teacher.state_dict(),config=config,step=step),temp)
                temp.replace(out/("encoder_final.pt" if step==steps else f"encoder_step{step:04d}.pt"))
                if not args.smoke:
                    state=dict(online=online.state_dict(),teacher=teacher.state_dict(),optimizer=opt.state_dict(),step=step,
                        config=config,python_rng=random.getstate(),numpy_rng=np.random.get_state(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all())
                    temp=out/"last_training.partial.pt"; torch.save(state,temp);temp.replace(out/"last_training.pt")
        save_json(out/"complete.json",dict(steps=steps,external_only=True,source_counts=counts,elapsed_seconds=time.time()-started,
            encoder_sha256=hashlib.sha256((out/"encoder_final.pt").read_bytes()).hexdigest(),completed_at=time.strftime("%FT%T%z")))


if __name__=="__main__": main()

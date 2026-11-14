"""Mean-pool CPSAM candidate, with the exact OOF checkpoint key layout."""
import contextlib
import json
from pathlib import Path
import cv2
import numpy as np
import torch
from torch import nn

LM=np.array([7.919919967651367,-0.9824771285057068,-0.6211431622505188],np.float32)
LS=np.array([0.18971863389015198,0.18813571333885193,0.2748945653438568],np.float32)
C5=[(0,0),(0,256),(256,0),(256,256),(128,128)]
CLAHE=cv2.createCLAHE(2.0,(8,8))


def save_json(path,value):
    path=Path(path); temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(value,indent=2,allow_nan=False)); temp.replace(path)


def normalize(image):
    image=CLAHE.apply(np.ascontiguousarray(image))
    lo,hi=np.percentile(image,(1,99))
    return np.clip((image.astype(np.float32)-lo)/(hi-lo+1e-6),0,1)


def central512(image):
    if image.ndim==3 and image.shape[0]==1: image=image[0]
    if image.ndim!=2 or image.dtype!=np.uint8:
        raise ValueError(f'Expected uint8 grayscale MHA, got {image.shape} {image.dtype}')
    h,w=image.shape
    if min(h,w)<512: raise ValueError('Image smaller than validated central512 input')
    r,c=(h-512)//2,(w-512)//2
    return image[r:r+512,c:c+512]


def eval_views(image):
    views=[]
    for r,c in C5:
        tile=normalize(image[r:r+256,c:c+256])
        for k in range(8):
            v=np.rot90(tile,k%4)
            if k>=4: v=v[:,::-1]
            views.append(np.stack([np.ascontiguousarray(v)]*3))
    return np.stack(views)


def train_view(image):
    r,c=np.random.randint(0,257),np.random.randint(0,257)
    tile=normalize(image[r:r+256,c:c+256])
    if np.random.rand()<.5: tile=tile[:,::-1]
    if np.random.rand()<.5: tile=tile[::-1,:]
    tile=np.ascontiguousarray(np.rot90(tile,np.random.randint(4)))
    return np.stack([tile]*3)


class CandidateNet(nn.Module):
    def __init__(self,encoder):
        super().__init__(); self.encoder=encoder; self.pool=nn.AdaptiveAvgPool2d(1)
        self.norm=nn.LayerNorm(256)
        def head():
            m=nn.Sequential(nn.Linear(256,128),nn.GELU(),nn.Dropout(.2),nn.Linear(128,1))
            nn.init.zeros_(m[-1].weight); nn.init.zeros_(m[-1].bias)
            return m
        self.cd,self.cv,self.hex=head(),head(),head()

    def forward(self,x):
        g=self.norm(self.pool(self.encoder(x)).flatten(1))
        return torch.cat([self.cd(g),self.cv(g),self.hex(g)],1)


def build_model(pretrained,device='cuda',model_class=CandidateNet):
    from cellpose import models
    base=models.CellposeModel(gpu=device.startswith('cuda'),pretrained_model=str(pretrained))
    # Construct the head exactly once, including for prediction-bag training.
    encoder=base.net.encoder.float(); net=model_class(encoder).to(device)
    del base
    return net


def load_member(net,path,device='cuda'):
    ck=torch.load(path,map_location='cpu',weights_only=True)
    config=ck['config']
    assert config['pool']=='mean' and config['k_crops']==1
    np.testing.assert_array_equal(config['lm'],LM.tolist())
    np.testing.assert_array_equal(config['ls'],LS.tolist())
    net.load_state_dict(ck['model'],strict=True); net.eval()
    return ck


@torch.no_grad()
def predict_one(net,image,dtype=torch.bfloat16,view_batch=8,device='cuda'):
    arrays=eval_views(image); parts=[]
    for start in range(0,len(arrays),view_batch):
        x=torch.from_numpy(arrays[start:start+view_batch]).to(device)
        amp=torch.autocast('cuda',dtype=dtype) if device.startswith('cuda') else contextlib.nullcontext()
        with amp: parts.append(net(x).float())
    encoded=torch.cat(parts).mean(0)
    pred=torch.exp(encoded*torch.as_tensor(LS,device=device)+torch.as_tensor(LM,device=device))
    value=pred.cpu().numpy()
    if not np.isfinite(value).all(): raise ValueError('Non-finite inference output')
    return value

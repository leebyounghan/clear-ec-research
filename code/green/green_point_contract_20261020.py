"""Prospective green point/ROI CPU contract; no model or pseudo-mask training."""
import json,hashlib,datetime
from pathlib import Path
import numpy as np
ROOT=Path('/workspace/clear-ec-research')
def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def flip_pair(image, points, box, horizontal, vertical):
    h,w=image.shape[-2:]; p=points.copy(); b=box.copy(); out=image
    if horizontal:
        out=out[...,::-1]; p[:,0]=w-1-p[:,0]; b[[0,2]]=w-1-b[[2,0]]
    if vertical:
        out=out[...,::-1,:]; p[:,1]=h-1-p[:,1]; b[[1,3]]=h-1-b[[3,1]]
    return np.ascontiguousarray(out),p,b

def main():
    annotation=ROOT/'datasets/clear_ec_official/green_eval/annotations.json'
    split=ROOT/'experiments/target_fold4_20260912/preflight/fold4.npz'
    meta=ROOT/'datasets/clear_ec_fullframe_20260904/fullframe_meta.npz'
    a=json.loads(annotation.read_text())
    with np.load(meta,allow_pickle=True) as z: ids=z['ids'].astype(str)
    with np.load(split) as z: tr=set(ids[z['train']]); va=set(ids[z['validation']])
    train=sorted(set(a)&tr); validation=sorted(set(a)&va)
    assert len(train)==20 and len(validation)==5 and not set(train)&set(validation)
    assert set(train)|set(validation)==set(a)
    raw=np.memmap(ROOT/'datasets/clear_ec_fullframe_20260904/img_full_raw.dat',mode='r',dtype=np.uint8,shape=(len(ids),972,1296))
    checks=0
    for name in train+validation:
        im=np.array(raw[np.flatnonzero(ids==name)[0]])
        p=np.asarray(a[name]['markers_mha'],dtype=float); b=np.asarray(a[name]['roi_box_mha'],dtype=float)
        for fh,fv in [(False,False),(True,False),(False,True),(True,True)]:
            aug,pp,bb=flip_pair(im,p,b,fh,fv)
            restored,rp,rb=flip_pair(aug,pp,bb,fh,fv)
            np.testing.assert_array_equal(im,restored);np.testing.assert_allclose(p,rp,rtol=0,atol=1e-12);np.testing.assert_array_equal(b,rb)
            # Independent integer pixel-index check, separate from inverse mapping.
            xy=np.rint(p).astype(int);xy2=xy.copy()
            if fh:xy2[:,0]=1295-xy[:,0]
            if fv:xy2[:,1]=971-xy[:,1]
            np.testing.assert_array_equal(im[xy[:,1],xy[:,0]],aug[xy2[:,1],xy2[:,0]])
            assert ((pp>=bb[:2])&(pp<=bb[2:])).all()
            checks+=1
    print(json.dumps(dict(status='cpu_point_roi_contract_passed',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),train_ids=train,validation_ids=validation,train_n=20,validation_n=5,paired_transform_checks=checks,coordinate_convention='pixel centers x in [0,W-1], y in [0,H-1]; inclusive ROI bounds',augmentation='Shared H/V flips only; no independent spatial transforms for paired consistency',inputs={str(p):digest(p) for p in [annotation,split,meta]},limitations=['Only preserved annotations/overlays available; original TIFF completeness unverified','Points and ROI are not boundary masks','Fold4 is reused development data, not independent generalization','Pixel/slide-connected split preflight still required by training launcher'],gpu_training_started=False),indent=2))
if __name__=='__main__':main()

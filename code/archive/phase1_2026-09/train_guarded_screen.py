"""One fresh terminal 35-epoch crop fit, using audited fixed-step training helpers."""
import argparse
import json
from pathlib import Path
import sys
import time
import cv2
import numpy as np
import torch
import torch.utils.checkpoint as checkpoint

ROOT=Path("experiments/crop_dino_screen_20260910")
sys.path.insert(0,str(ROOT/"source/frozen"))
import train_specialize_screen as shared
from guarded_region_views import GuardedRegionViews, VERSION


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--fold",type=int,choices=(0,1),required=True)
    p.add_argument("--policy",choices=("control","guarded"),required=True)
    p.add_argument("--smoke",action="store_true")
    args=p.parse_args()
    out=ROOT/("smoke" if args.smoke else "runs")/f"{args.policy}_fold{args.fold}"
    out.mkdir(parents=True,exist_ok=False)
    preflight=json.loads((ROOT/"preflight/complete.json").read_text())
    assert preflight["ready_for_crop_screen"]
    selection=ROOT/"preflight/guarded_selection.npz"
    assert shared.digest(selection)==preflight["cache_sha256"]
    assert shared.digest("cpsam_models/cpsam_v2")==preflight["initial_weights_sha256"]
    torch.set_num_threads(4); cv2.setNumThreads(1); shared.seed(shared.SEED)
    torch.backends.cuda.matmul.allow_tf32=True; torch.backends.cudnn.allow_tf32=True
    torch.set_float32_matmul_precision("high")
    ids,y,tr,va,z,views,old=shared.load_data(args.fold,out)
    if args.policy=="guarded":
        central=np.memmap("datasets/clear_ec_official/img_native_512.dat",mode="r",dtype=np.uint8,shape=(len(ids),512,512))
        views=GuardedRegionViews(central,ids,selection)
    net=shared.build_model("cpsam_models/cpsam_v2")
    for block in net.encoder.blocks:
        fn=block.forward
        block.forward=lambda x,original=fn:checkpoint.checkpoint(original,x,use_reentrant=False)
    steps=2 if args.smoke else 35*(len(tr)//shared.BS)
    config=dict(fold=args.fold,policy=args.policy,version=VERSION,smoke=args.smoke,
        seed=shared.SEED,updates=steps,epochs_equivalent=35 if not args.smoke else None,
        k_crops=shared.K,batch_size=shared.BS,encoder_lr=1e-5,head_lr=1e-3,
        objective="mixed_mape",select_mode="fixed_terminal_no_validation_selection",
        lm=shared.LM.tolist(),ls=shared.LS.tolist(),initialization="original_cpsam_v2",
        initial_sha256=preflight["initial_weights_sha256"],selection_sha256=preflight["cache_sha256"],
        train_rows=len(tr),validation_rows=len(va),split_sha256=shared.digest(old/"split_indices.npz"),
        data_order="identical seed77 epoch permutations; crop selection is the only policy change",
        historic_validation_reuse=True)
    shared.save_json(out/"config.json",config)
    training=shared.train(net,views,tr,z,None,steps,out,1e-5,shared.SEED)
    torch.save(dict(model=net.state_dict(),config=config),out/"selected.pt")
    state=torch.load(out/"selected.pt",map_location="cpu",weights_only=True)
    net.load_state_dict(state["model"],strict=True); del state
    evaluation=shared.evaluate(net,views,va[:4] if args.smoke else va,y,ids,out)
    result=dict(completed_at=time.strftime("%FT%T%z"),training=training,evaluation=evaluation,
        checkpoint_sha256=shared.digest(out/"selected.pt"),strict_reload_passed=True,
        deployment_approved=False,smoke=args.smoke)
    shared.save_json(out/"result.json",result)
    print(json.dumps(result,indent=2),flush=True)


if __name__=="__main__":main()

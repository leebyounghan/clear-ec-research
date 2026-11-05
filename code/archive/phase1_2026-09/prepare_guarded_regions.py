"""CPU preflight, unlabeled per-image cache and training-only quality audit.

Never exports images, labels, embeddings or individual predictions to a service.
Original data/caches are read-only; all writes are in a new experiment folder.
"""
import csv
import hashlib
import json
from pathlib import Path
import time
import cv2
import numpy as np
from native_region_views import NativeRegionViews
from guarded_region_views import tile_quality, choose, native_origins, VERSION

ROOT = Path("experiments/crop_dino_screen_20260910")


def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(8*1024**2), b""): h.update(b)
    return h.hexdigest()


def main():
    out=ROOT/"preflight"; out.mkdir(parents=True,exist_ok=False)
    cv2.setNumThreads(1)
    data=Path("datasets/clear_ec_official")
    with np.load(data/"cache_native_meta_512.npz",allow_pickle=True) as d:
        ids,y=d["ids"].astype(str),d["Y"].astype(np.float32)
    assert len(ids)==8954 and len(np.unique(ids))==8954 and np.isfinite(y).all()
    clean=lambda s:Path(str(s)).stem
    with (data/"final_train_ids.csv").open() as f:
        mapping={clean(r["ID"]):r["slide_id"] for r in csv.DictReader(f)}
    slides=np.array([mapping[clean(i)] for i in ids])
    hashes={}; train_sets=[]; validation=[]
    for fold in (0,1):
        old=Path("experiments/region_seedcheck_20260906/runs")/f"full_quality_f{fold}_seed77_c256k2_mean_multitask_slidesmooth"
        with np.load(old/"split_indices.npz") as d: tr,va=d["train"],d["validation"]
        assert len(va)==1791 and not set(tr)&set(va) and not set(slides[tr])&set(slides[va])
        with np.load(old/"best_oof.npz") as d:
            for a,b in ((va,d["indices"]),(ids[va],d["ids"].astype(str)),(y[va],d["target"])):
                np.testing.assert_array_equal(a,b)
        for p in (old/"split_indices.npz",old/"best_oof.npz"): hashes[str(p)]=sha(p)
        train_sets.append(set(tr)); validation.extend(va)
    audit_indices=np.array(sorted(train_sets[0]&train_sets[1]))
    assert not set(audit_indices)&set(validation)
    central=np.memmap(data/"img_native_512.dat",dtype=np.uint8,mode="r",shape=(8954,512,512))
    views=NativeRegionViews(central,ids,"full_quality")
    n,m=len(ids),len(views.origins)
    occupancy=np.empty((n,m),np.float32); saturation=np.empty((n,m),np.float32)
    acceptable=np.empty((n,m),bool); selected=np.empty((n,8,2),np.int32)
    chosen_indices=np.empty((n,8),np.int32)
    start=time.monotonic()
    for i in range(n):
        for j,(r,c) in enumerate(views.origins):
            occupancy[i,j],saturation[i,j],acceptable[i,j]=tile_quality(views.images[i,r:r+256,c:c+256])
        selected[i],chosen_indices[i]=choose(views.origins,views.scores[i],occupancy[i],saturation[i],acceptable[i])
        if (i+1)%250==0: print(json.dumps(dict(prepared=i+1,total=n,seconds=time.monotonic()-start)),flush=True)
    # Verify recomputed native image path produces the same cache choices.
    for i in audit_indices[:4]: np.testing.assert_array_equal(native_origins(views.images[i]), selected[i])
    lookup={tuple(p):j for j,p in enumerate(views.origins)}
    old_indices=np.array([[lookup[tuple(p)] for p in views.selected[i,:5]] for i in audit_indices])
    new_indices=chosen_indices[audit_indices,:5]
    def report(ix):
        good=acceptable[audit_indices[:,None],ix]
        return dict(tile_rejection_fraction=float((~good).mean()),
                    images_with_rejected_tile_fraction=float((~good).any(1).mean()),
                    mean_informative_fraction=float(occupancy[audit_indices[:,None],ix].mean()),
                    mean_clipped_fraction=float(saturation[audit_indices[:,None],ix].mean()))
    path=out/"guarded_selection.npz"
    np.savez_compressed(path,ids=ids,selected=selected,version=VERSION)
    old_report,new_report=report(old_indices),report(new_indices)
    result=dict(created_at=time.strftime("%FT%T%z"),version=VERSION,n=n,
        audit_train_only_n=len(audit_indices),old=old_report,new=new_report,
        changed_images_fraction=float(np.any(selected[audit_indices,:5]!=views.selected[audit_indices,:5],axis=(1,2)).mean()),
        audit_split="intersection of fold0 and fold1 training sets; neither validation fold inspected",
        quality_is_not_roi_gt=True,validation_labels_used_for_selection=False,
        cache_is_per_image_fixed_function=True,cache_sha256=sha(path),input_sha256=hashes,
        initial_weights_sha256=sha("cpsam_models/cpsam_v2"),
        ready_for_crop_screen=bool(old_report["tile_rejection_fraction"]>0 and new_report["tile_rejection_fraction"]<old_report["tile_rejection_fraction"]),
        elapsed_seconds=time.monotonic()-start,performance_improvement_established=False,
        dino_status="awaiting challenge pretrained-model eligibility; no DINO data processing started")
    with (out/"complete.json").open("x") as f: json.dump(result,f,indent=2)
    print(json.dumps(result,indent=2),flush=True)


if __name__=="__main__": main()

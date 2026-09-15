"""Diagnostic plots of proposed image-only crop choices; never GT annotation."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from native_region_views import NativeRegionViews


def main():
    out=Path("experiments/region_search_20260905/diagnostics");out.mkdir(parents=True,exist_ok=True)
    meta=np.load("datasets/clear_ec_official/cache_native_meta_512.npz",allow_pickle=True)
    central=np.memmap("datasets/clear_ec_official/img_native_512.dat",dtype="uint8",mode="r",shape=(8954,512,512))
    views=NativeRegionViews(central,meta["ids"],"anchor_quality")
    with np.load("experiments/methods_20260904/runs/metric35_f0_seed42_c256k1_mean_multitask_slidesmooth/split_indices.npz") as data:
        indices=data["train"][[0,200,400,600]]
    fig,axs=plt.subplots(2,2,figsize=(12,9))
    for ax,ix in zip(axs.flat,indices):
        ax.imshow(views.images[ix],cmap="gray",vmin=0,vmax=255)
        for j,(r,c) in enumerate(views.eval_origins_for(ix)):
            ax.add_patch(Rectangle((c,r),256,256,fill=False,edgecolor="lime" if j<3 else "orange",lw=1.5))
        ax.set_title(f"training-row {int(ix)}; proposed crops, NOT ROI GT");ax.axis("off")
    fig.tight_layout();fig.savefig(out/"proposed_regions.png",dpi=120);plt.close(fig)


if __name__=="__main__":main()

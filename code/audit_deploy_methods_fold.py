"""Full held-out numerical audit of both new recipes through submission code."""
import argparse
import fcntl
import importlib.util
from pathlib import Path
import time

import numpy as np
import torch

from audit_method_screen import run_path, load_fold
from audit_submission_ensembles import metrics
from final_candidate_core import build_model, predict_one, save_json

ROOT = Path("experiments/deploy_review_20260905")


def interface():
    path = Path(__file__).resolve().parent.parent / "submission_candidate_20260905/inference.py"
    spec = importlib.util.spec_from_file_location("review_interface", path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, choices=range(5), required=True)
    parser.add_argument("--limit", type=int, default=0, help="Smoke only; never writes full result")
    args = parser.parse_args()
    out = ROOT / ("audit_smoke" if args.limit else "inference_audit") / f"fold{args.fold}"
    out.mkdir(parents=True, exist_ok=True)
    with (out / "audit.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (out / "result.json").exists():
            print("already complete", flush=True); return
        torch.set_num_threads(4)
        torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
        meta = np.load("datasets/clear_ec_official/cache_native_meta_512.npz", allow_pickle=True)
        imgs = np.memmap("datasets/clear_ec_official/img_native_512.dat", dtype="uint8", mode="r", shape=(8954,512,512))
        api = interface(); start = time.time(); net = build_model("cpsam_models/cpsam_v2")
        results, preds, reference = {}, {}, None
        for recipe in ("metric35", "predbag2_35"):
            path = run_path(recipe, args.fold); data = load_fold(path)
            ix, y, expected, ids = (data[k] for k in ("indices", "target", "pred", "ids"))
            if args.limit:
                ix, y, expected, ids = (a[:args.limit] for a in (ix,y,expected,ids))
            if reference is not None:
                for a,b in zip((ix,y,ids),reference):
                    np.testing.assert_array_equal(a,b)
            else:
                reference = (ix.copy(),y.copy(),ids.copy())
            np.testing.assert_array_equal(meta["ids"][ix].astype(str), ids.astype(str))
            api.load_strict(net, path / "best.pt")
            for name, dtype in (("bf16",torch.bfloat16),("fp16",torch.float16)):
                predict_one(net, imgs[ix[0]], dtype=dtype); torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats(); values, durations = [], []
                for k, index in enumerate(ix):
                    t = time.perf_counter(); values.append(predict_one(net, imgs[index], dtype=dtype))
                    torch.cuda.synchronize(); durations.append(time.perf_counter()-t)
                    if (k+1) % 100 == 0:
                        progress = dict(recipe=recipe, precision=name, images=k+1, total=len(ix),
                                        elapsed_seconds=time.time()-start, updated_at=time.strftime("%FT%T%z"))
                        save_json(out / "progress.json", progress); print(progress, flush=True)
                values = np.asarray(values); key = f"{recipe}_{name}"; preds[key] = values
                rel = np.abs(values-expected)/expected
                delta = float(metrics(values,y).mean()-metrics(expected,y).mean())
                results[key] = dict(max_relative_difference=rel.max(0).tolist(), mean_relative_difference=rel.mean(0).tolist(),
                    raw_score_delta=delta, mean_seconds=float(np.mean(durations)), p95_seconds=float(np.quantile(durations,.95)),
                    peak_allocated_gib=torch.cuda.max_memory_allocated()/1024**3)
                if not np.isfinite(values).all() or not (values>0).all():
                    raise ValueError(f"Nonfinite or nonpositive predictions: {key}")
                if name == "bf16":
                    if rel.max() > 1e-5:
                        raise ValueError(f"BF16 parity failed: {key}: {rel.max()}")
                # A two-image interface smoke is not a population score audit.
                # Keep prediction-level gates in smoke; score drift requires the full fold.
                elif rel.mean() > .005 or rel.max() > .05 or (not args.limit and abs(delta) > .01):
                    raise ValueError(f"FP16 parity failed: {key}: {results[key]}")
        np.savez_compressed(out / "predictions.npz", indices=reference[0], target=reference[1], ids=reference[2], **preds)
        save_json(out / "result.json", dict(fold=args.fold, n=len(reference[0]), smoke=bool(args.limit), comparisons=results,
            device=torch.cuda.get_device_name(), elapsed_seconds=time.time()-start, completed_at=time.strftime("%FT%T%z"),
            scope="Numerical parity on existing OOF, not new independent validation. H200 not target-device certification."))


if __name__ == "__main__":
    main()

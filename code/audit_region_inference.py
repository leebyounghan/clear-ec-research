"""Finite read-only prediction audits; new exclusive output per job."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

FROZEN = Path("experiments/region_search_20260905/source/code")
sys.path.append(str(FROZEN))
import numpy as np
import torch
from audit_method_screen import aligned, load_fold
from audit_submission_ensembles import metrics
from final_candidate_core import build_model, save_json
from native_region_views import NativeRegionViews
from region_inference_core import native_views, load_region, predict_views

ROOT = Path("experiments/region_inference_20260907")


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8*1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, choices=range(5))
    parser.add_argument("--all-data", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    assert args.all_data != (args.fold is not None)
    key = "all_data" if args.all_data else f"fold{args.fold}"
    out = ROOT / ("smoke" if args.smoke else "audits") / key
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    meta = np.load("datasets/clear_ec_official/cache_native_meta_512.npz", allow_pickle=True)
    central = np.memmap("datasets/clear_ec_official/img_native_512.dat", dtype="uint8", mode="r", shape=(8954,512,512))
    net = build_model("cpsam_models/cpsam_v2")
    start = time.time()
    results, predictions, inputs = {}, {}, {}
    reference = None
    for policy in ("anchor_quality", "full_quality"):
        cached = NativeRegionViews(central, meta["ids"], policy)
        if args.all_data:
            import SimpleITK as sitk
            sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(1)
            final_meta = np.load("datasets/clear_ec_final9000_20260904/meta.npz")
            all_ids = final_meta["ids"].astype(str)
            old_ids = set(meta["ids"].astype(str))
            ids = list(meta["ids"].astype(str)[:4]) + [v for v in all_ids if v not in old_ids][:4]
            if args.smoke:
                ids = [ids[0], ids[4]]
            images = [sitk.GetArrayFromImage(sitk.ReadImage(str(Path("datasets/clear_ec_fullframe_20260904/train_mha") / (image_id+".mha")))) for image_id in ids]
            path = Path("experiments/deploy_region_20260907/runs") / policy / "selected.pt"
            expected = None
        else:
            folder = "region_seedcheck_20260906" if args.fold < 2 else "region_seedconfirm_20260906"
            run = Path("experiments") / folder / "runs" / f"{policy}_f{args.fold}_seed77_c256k2_mean_multitask_slidesmooth"
            data = load_fold(run)
            if reference is not None:
                aligned(reference, data)
            reference = data
            ix = data["indices"][:2] if args.smoke else data["indices"]
            ids = data["ids"][:len(ix)]
            expected = data["pred"][:len(ix)].astype(float)
            targets = data["target"][:len(ix)].astype(float)
            images = (cached.images[i] for i in ix)
            path = run / "best.pt"
        inputs[str(path)] = digest(path)
        cfg = load_region(net, path, policy)
        if args.all_data:
            assert cfg["selected_steps"] == (4470 if policy == "anchor_quality" else 4917)
            assert cfg["train_rows"] == 8998 and cfg["validation_rows"] == 0
        parts, durations = {"bf16": [], "fp16": []}, {"bf16": [], "fp16": []}
        torch.cuda.reset_peak_memory_stats()
        for i, image in enumerate(images):
            arrays = native_views(image, policy)
            if not args.all_data:
                np.testing.assert_array_equal(arrays, cached.evaluate(int(ix[i]), 1))
            for precision, dtype in (("bf16", torch.bfloat16), ("fp16", torch.float16)):
                torch.cuda.synchronize()
                t = time.perf_counter()
                parts[precision].append(predict_views(net, arrays, dtype))
                torch.cuda.synchronize()
                durations[precision].append(time.perf_counter() - t)
            if (i+1) % 100 == 0:
                save_json(out / "progress.json", dict(policy=policy, images=i+1, total=len(ids), updated_at=time.strftime("%FT%T%z")))
                print(policy, i+1, len(ids), flush=True)
        for precision in parts:
            values = np.asarray(parts[precision], dtype=float)
            comparison = expected if expected is not None else np.asarray(parts["bf16"], dtype=float)
            rel = np.abs(values-comparison)/comparison
            drift = None if args.all_data else float(metrics(values, targets).mean()-metrics(comparison, targets).mean())
            passed = (rel.max() <= 1e-5 if precision == "bf16" else
                      rel.max() <= .05 and rel.mean() <= .005 and (args.smoke or args.all_data or abs(drift) <= .01))
            name = f"{policy}_{precision}"
            results[name] = dict(n=len(ids), passed=bool(passed), max_relative=float(rel.max()), mean_relative=float(rel.mean()),
                raw_score_delta=drift, mean_seconds=float(np.mean(durations[precision])), p95_seconds=float(np.quantile(durations[precision], .95)),
                peak_allocated_gib=torch.cuda.max_memory_allocated()/1024**3)
            predictions[name] = values
            if not passed:
                save_json(out / "failure.json", results)
                raise ValueError(f"Predeclared parity gate failed: {name}: {results[name]}")
    arrays = dict(ids=np.asarray(ids), **predictions)
    if not args.all_data:
        arrays.update(indices=ix, target=targets)
    np.savez_compressed(out / "predictions.npz", **arrays)
    save_json(out / "result.json", dict(key=key, smoke=args.smoke, comparisons=results, checkpoint_sha256=inputs,
        elapsed_seconds=time.time()-start, completed_at=time.strftime("%FT%T%z"), deployment_approved=False,
        scope="Numerical OOF parity or all-data native smoke, not independent validation; H200 not competition-device or container certification"))


if __name__ == "__main__":
    main()

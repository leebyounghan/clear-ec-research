"""Fixed-step, matched multitask versus specialist continuation; never selects on validation.

One fold worker owns one GPU. Fresh CPSAM -> common train-only model -> four
independently restored branches. Existing checkpoints are NOT initialization.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time

import cv2
import numpy as np
import torch
import torch.utils.checkpoint as checkpoint

ROOT = Path("experiments/specialize_screen_20260908")
FROZEN = ROOT / "source/frozen"
sys.path.insert(0, str(FROZEN))
from final_candidate_core import build_model, LM, LS, save_json
from method_training_core import regression_loss, cosine_multiplier
from native_region_views import NativeRegionViews

SEED, BS, K = 77, 48, 2
COMMON_STEPS, BRANCH_STEPS = 2980, 745
BRANCHES = {"multitask": None, "cd": 0, "cv": 1, "hex": 2}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def seed(value):
    random.seed(value); np.random.seed(value)
    torch.manual_seed(value); torch.cuda.manual_seed_all(value)


def clean_id(value):
    value = str(value)
    for suffix in (".bmp", ".mha", ".tif", ".tiff"):
        if value.lower().endswith(suffix):
            return value[:-len(suffix)]
    return value


def load_data(fold, destination):
    data = Path("datasets/clear_ec_official")
    meta = np.load(data / "cache_native_meta_512.npz", allow_pickle=True)
    ids, y = meta["ids"].astype(str), meta["Y"].astype(np.float32)
    old = Path("experiments/region_seedcheck_20260906/runs") / (
        f"full_quality_f{fold}_seed77_c256k2_mean_multitask_slidesmooth")
    split = np.load(old / "split_indices.npz")
    tr, va = split["train"].copy(), split["validation"].copy()
    assert len(ids) == 8954 and len(va) == 1791
    assert len(np.unique(tr)) == len(tr) and len(np.unique(va)) == len(va)
    assert not set(tr) & set(va) and (y[tr] > 0).all() and np.isfinite(y).all()
    with (data / "final_train_ids.csv").open(newline="") as f:
        mapping = {clean_id(r["ID"]): str(r["slide_id"]) for r in csv.DictReader(f)}
    slides = np.asarray([mapping[clean_id(i)] for i in ids])
    assert not set(slides[tr]) & set(slides[va])
    targets = y.copy()
    for slide in np.unique(slides[tr]):
        ix = tr[slides[tr] == slide]
        if len(ix) > 1:
            targets[ix] = y[ix].mean(0)
    old_oof = np.load(old / "best_oof.npz")
    np.testing.assert_array_equal(va, old_oof["indices"])
    np.testing.assert_array_equal(ids[va], old_oof["ids"].astype(str))
    np.testing.assert_array_equal(y[va], old_oof["target"])
    images = np.memmap(data / "img_native_512.dat", mode="r", dtype=np.uint8,
                       shape=(len(ids), 512, 512))
    views = NativeRegionViews(images, ids, "full_quality")
    np.savez_compressed(destination / "split_indices.npz", train=tr, validation=va)
    z = torch.tensor(((np.log(targets.clip(1e-6)) - LM) / LS).astype(np.float32), device="cuda")
    return ids, y, tr, va, z, views, old


def target_loss(output, targets, scale, metric):
    if metric is not None:
        output, targets, scale = output[:, metric:metric+1], targets[:, metric:metric+1], scale[metric:metric+1]
    return regression_loss(output, targets, scale, "mixed_mape")


def train(net, views, indices, z, metric, updates, destination, lr, rng_seed):
    seed(rng_seed)
    for name, p in net.named_parameters():
        p.requires_grad_(metric is None or not any(name.startswith(h + ".")
            for j, h in enumerate(("cd", "cv", "hex")) if j != metric))
    encoder = [p for p in net.encoder.parameters() if p.requires_grad]
    encoder_ids = {id(p) for p in encoder}
    heads = [p for p in net.parameters() if p.requires_grad and id(p) not in encoder_ids]
    optimizer = torch.optim.AdamW([dict(params=encoder, lr=lr), dict(params=heads, lr=lr*100)], weight_decay=1e-4)
    assert {id(p) for g in optimizer.param_groups for p in g["params"]} == {id(p) for p in net.parameters() if p.requires_grad}
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda n: cosine_multiplier(n, updates))
    scale = torch.tensor(LS, device="cuda")
    started = time.perf_counter(); net.train(); torch.cuda.reset_peak_memory_stats()
    epoch, ix, losses = -1, np.array([], dtype=int), []
    sampling_hash = hashlib.sha256()
    for step in range(updates):
        if step % (len(indices)//BS) == 0:
            epoch += 1
            ix = np.random.default_rng(rng_seed*1000+epoch).permutation(indices)
        offset = (step % (len(indices)//BS))*BS
        batch = ix[offset:offset+BS]
        array = np.stack([views.train(int(i), K) for i in batch])
        if step < 3:
            sampling_hash.update(batch.tobytes()); sampling_hash.update(array.tobytes())
        x = torch.from_numpy(array).to("cuda")
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            encoded = net(x.flatten(0, 1)).reshape(BS, K, 3).mean(1)
            loss = target_loss(encoded, z[batch], scale, metric)
        if not torch.isfinite(loss):
            raise FloatingPointError("nonfinite loss")
        loss.backward()
        if step == 0:
            assert all(p.grad is None for p in net.parameters() if not p.requires_grad)
            # Zero-initialized heads can legitimately give zero encoder gradients
            # on the very first common-model update, but the graph must connect.
            assert any(p.grad is not None for p in encoder)
        torch.nn.utils.clip_grad_norm_([p for p in net.parameters() if p.requires_grad], 1., error_if_nonfinite=True)
        optimizer.step(); scheduler.step()
        losses.append(loss.detach().item())
        if (step+1) % 25 == 0 or step+1 == updates:
            elapsed = time.perf_counter()-started
            state = dict(stage="training", step=step+1, total_steps=updates, metric=metric,
                elapsed_seconds=elapsed, seconds_per_step=elapsed/(step+1),
                mean_loss=float(np.mean(losses[-25:])), peak_gib=torch.cuda.max_memory_allocated()/2**30,
                updated_at=time.strftime("%FT%T%z"))
            save_json(destination / "progress.json", state)
            print(json.dumps(state), flush=True)
    torch.cuda.synchronize()
    result = dict(updates=updates, elapsed_seconds=time.perf_counter()-started,
        peak_gib=torch.cuda.max_memory_allocated()/2**30, first_three_batches_sha256=sampling_hash.hexdigest())
    del optimizer, scheduler
    for p in net.parameters():
        p.grad = None
    return result


@torch.no_grad()
def evaluate(net, views, indices, y, ids, destination):
    net.eval(); rows = []; start = time.perf_counter()
    for n, i in enumerate(indices):
        x = torch.from_numpy(views.evaluate(int(i), 1)).to("cuda")
        parts = []
        for offset in range(0, len(x), 8):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                parts.append(net(x[offset:offset+8]).float())
        encoded = torch.cat(parts).mean(0)
        rows.append(torch.exp(encoded*torch.as_tensor(LS, device="cuda")+torch.as_tensor(LM, device="cuda")).cpu().numpy())
        if (n+1) % 100 == 0:
            save_json(destination / "progress.json", dict(stage="evaluation", images=n+1,
                total_images=len(indices), elapsed_seconds=time.perf_counter()-start,
                updated_at=time.strftime("%FT%T%z")))
    prediction = np.stack(rows)
    assert np.isfinite(prediction).all() and (prediction > 0).all()
    values = [float((abs(prediction[y[indices,j]>0,j]-y[indices[y[indices,j]>0],j]) /
        y[indices[y[indices,j]>0],j]).mean()*100) for j in range(3)]
    np.savez_compressed(destination / "oof.npz", indices=indices, ids=ids[indices], target=y[indices], pred=prediction)
    return dict(metrics=values, elapsed_seconds=time.perf_counter()-start)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--fold", type=int, choices=(0,1), required=True)
    parser.add_argument("--smoke", action="store_true"); args = parser.parse_args()
    destination = ROOT / ("smoke" if args.smoke else "runs") / f"fold{args.fold}"
    destination.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4); cv2.setNumThreads(1); seed(SEED)
    torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    ids, y, tr, va, z, views, old = load_data(args.fold, destination)
    net = build_model("cpsam_models/cpsam_v2")
    for block in net.encoder.blocks:
        original = block.forward
        block.forward = lambda x, fn=original: checkpoint.checkpoint(fn, x, use_reentrant=False)
    common = destination / "common"; common.mkdir()
    config = dict(fold=args.fold, seed=SEED, region_policy="full_quality", k_crops=K, batch_size=BS,
        lm=LM.tolist(), ls=LS.tolist(), train_rows=len(tr), validation_rows=len(va),
        common_updates=2 if args.smoke else COMMON_STEPS, branch_updates=2 if args.smoke else BRANCH_STEPS,
        common_encoder_lr=1e-5, branch_encoder_lr=3e-6, head_lr_multiplier=100,
        objective="mixed_mape", select_mode="fixed_terminal_no_validation_selection",
        initialization="fresh_cpsam_no_CLEAR_EC_checkpoint", split_source=str(old / "split_indices.npz"),
        split_sha256=digest(old / "split_indices.npz"), smoke=args.smoke)
    save_json(destination / "config.json", config)
    timing = train(net, views, tr, z, None, config["common_updates"], common, 1e-5, SEED)
    torch.save(dict(model=net.state_dict(), config=config), common / "selected.pt")
    common_hash = digest(common / "selected.pt")
    save_json(common / "result.json", dict(training=timing, sha256=common_hash, no_validation_used=True))
    results = {}; sampling = None
    for name, metric in BRANCHES.items():
        branch = destination / name; branch.mkdir()
        saved = torch.load(common / "selected.pt", map_location="cpu", weights_only=True)
        net.load_state_dict(saved["model"], strict=True); del saved
        training = train(net, views, tr, z, metric, config["branch_updates"], branch, 3e-6, SEED+100)
        if sampling is None:
            sampling = training["first_three_batches_sha256"]
        assert sampling == training["first_three_batches_sha256"], "branches did not see identical initial crops"
        torch.save(dict(model=net.state_dict(), config=dict(config, specialist_metric=metric,
            common_sha256=common_hash)), branch / "selected.pt")
        evaluation = evaluate(net, views, va[:4] if args.smoke else va, y, ids, branch)
        results[name] = dict(training=training, evaluation=evaluation,
            target_metric=metric, other_metrics_not_specialist_outputs=metric is not None,
            checkpoint_sha256=digest(branch / "selected.pt"), common_sha256=common_hash)
        save_json(branch / "result.json", results[name])
        print(name, json.dumps(results[name]), flush=True)
    save_json(destination / "result.json", dict(completed_at=time.strftime("%FT%T%z"),
        branches=results, deployment_approved=False, historical_validation_reuse=True))


if __name__ == "__main__":
    main()

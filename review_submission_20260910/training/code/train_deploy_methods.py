"""All-data fits for the two five-fold-confirmed September 5 recipes."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.utils.checkpoint as checkpoint

from final_candidate_core import CandidateNet, LM, LS, build_model, save_json, train_view
from method_training_core import cosine_multiplier, regression_loss


ROOT = Path("experiments/deploy_methods_20260905")
DATA = Path("datasets/clear_ec_final9000_20260904")
OOF_TOTAL_STEPS = 35 * 149
RECIPES = {
    "metric35": {"selected_steps": 4470, "k_crops": 1, "objective": "mixed_mape"},
    "predbag2_35": {"selected_steps": 3874, "k_crops": 2, "objective": "log_l1"},
}


class PredictionBagNet(CandidateNet):
    def forward(self, inputs):
        if inputs.ndim != 5:
            return super().forward(inputs)
        batch, views = inputs.shape[:2]
        outputs = super().forward(inputs.reshape(batch * views, *inputs.shape[2:]))
        return outputs.reshape(batch, views, 3).mean(1)


def build_recipe_model(recipe):
    ordinary = build_model("cpsam_models/cpsam_v2")
    if RECIPES[recipe]["k_crops"] == 1:
        return ordinary
    net = PredictionBagNet(ordinary.encoder).to("cuda")
    del ordinary
    return net


def training_array(images, indices, k_crops):
    if k_crops == 1:
        return np.stack([train_view(images[i]) for i in indices])
    return np.stack([[train_view(images[i]) for _ in range(k_crops)] for i in indices])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--recipe", choices=RECIPES, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    recipe = RECIPES[args.recipe]
    run = ROOT / ("smoke" if args.smoke else "runs") / args.recipe
    run.mkdir(parents=True, exist_ok=True)
    with (run / "train.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (run / "result.json").exists():
            print("already complete", flush=True)
            return
        if (run / "config.json").exists():
            raise RuntimeError("partial run exists; inspect before any restart")
        seed = 42
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
        meta = np.load(DATA / "meta.npz")
        indices = meta["train_indices"]
        assert len(indices) == 8998 and len(np.unique(indices)) == len(indices)
        y = meta["Y_train"]
        assert (y[indices] > 0).all()
        z = torch.tensor(((np.log(y.clip(1e-6)) - LM) / LS).astype(np.float32), device="cuda")
        ls = torch.tensor(LS, device="cuda")
        images = np.memmap(DATA / "img_native_512.dat", dtype=np.uint8, mode="r", shape=(9000, 512, 512))
        net = build_recipe_model(args.recipe)
        for block in net.encoder.blocks:
            block._deploy_forward = block.forward
            block.forward = (lambda module: lambda x: checkpoint.checkpoint(
                module._deploy_forward, x, use_reentrant=False))(block)
        heads = list(net.norm.parameters()) + list(net.cd.parameters()) + list(net.cv.parameters()) + list(net.hex.parameters())
        optimizer = torch.optim.AdamW([
            {"params": net.encoder.parameters(), "lr": 1e-5},
            {"params": heads, "lr": 1e-3}], weight_decay=1e-4)
        optimized = [id(p) for group in optimizer.param_groups for p in group["params"]]
        assert len(optimized) == len(set(optimized))
        assert set(optimized) == {id(p) for p in net.parameters() if p.requires_grad}
        target = 1 if args.smoke else int(recipe["selected_steps"])
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda step: cosine_multiplier(step, OOF_TOTAL_STEPS))
        batch_size = 4 if args.smoke else 48
        batches_per_epoch = math.ceil(len(indices) / batch_size)
        config = dict(recipe=args.recipe, seed=seed, selected_steps=target,
            oof_selected_steps=recipe["selected_steps"], oof_total_schedule_steps=OOF_TOTAL_STEPS,
            schedule="cosine_floor005_warmup200_matched_by_optimizer_update",
            batch_size=batch_size, k_crops=recipe["k_crops"], objective=recipe["objective"],
            pool="mean", lm=LM.tolist(), ls=LS.tolist(), train_rows=len(indices), validation_rows=0,
            selection="median of five OOF best optimizer-update counts, frozen before all-data fit",
            last_batch="included", all_9000_reconciled=True, excluded_nonpositive_hex=2,
            source_sha256={p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in
                ("code/train_deploy_methods.py", "code/final_candidate_core.py", "code/method_training_core.py")})
        save_json(run / "config.json", config)

        def save_checkpoint(name, epoch, step):
            destination = run / name
            temporary = destination.with_suffix(".partial.pt")
            torch.save({"model": net.state_dict(), "config": config, "epoch": epoch, "step": step}, temporary)
            temporary.rename(destination)

        step = 0
        epoch = 0
        started = time.time()
        while step < target:
            epoch += 1
            epoch_started = time.time()
            permutation = np.random.default_rng(seed * 1000 + epoch - 1).permutation(indices)
            total_loss = 0.0
            used_batches = 0
            net.train()
            for batch in range(batches_per_epoch):
                if step >= target:
                    break
                ix = permutation[batch * batch_size:(batch + 1) * batch_size]
                array = training_array(images, ix, recipe["k_crops"])
                inputs = torch.from_numpy(array).to("cuda", non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    output = net(inputs)
                    loss = regression_loss(output, z[ix], ls, recipe["objective"])
                if not torch.isfinite(loss):
                    raise FloatingPointError("nonfinite all-data loss")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0, error_if_nonfinite=True)
                optimizer.step(); scheduler.step()
                step += 1; used_batches += 1; total_loss += loss.detach().item()
                if not args.smoke and step % 1000 == 0:
                    save_checkpoint(f"step{step:04d}.pt", epoch, step)
                if step % 20 == 0 or step == target:
                    save_json(run / "progress.json", dict(recipe=args.recipe, epoch=epoch, step=step,
                        target=target, loss=float(loss), elapsed_seconds=time.time() - started,
                        updated_at=time.strftime("%FT%T%z")))
            print(f"epoch={epoch:02d} step={step}/{target} loss={total_loss/max(used_batches,1):.6f} "
                  f"seconds={time.time()-epoch_started:.1f}", flush=True)
        save_checkpoint("selected.pt", epoch, step)
        save_json(run / "result.json", dict(recipe=args.recipe, selected=str(run / "selected.pt"),
            selected_steps=target, epochs_touched=epoch, elapsed_seconds=time.time() - started,
            completed_at=time.strftime("%FT%T%z"),
            note="No all-data validation score; recipe and step were fixed from OOF before fitting."))


if __name__ == "__main__":
    main()

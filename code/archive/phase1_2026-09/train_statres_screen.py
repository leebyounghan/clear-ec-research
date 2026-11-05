"""Two-fold, two-branch finite screen, reusing a verified TRAIN-ONLY common model."""
import argparse
import json
from pathlib import Path
import sys
import time
import cv2
import numpy as np
import torch
import torch.utils.checkpoint as checkpoint

ROOT = Path("experiments/statres_screen_20260908")
sys.path.insert(0, str(ROOT / "source/frozen"))
from train_specialize_screen import load_data, train, evaluate, digest, seed, SEED, BS, K
from final_candidate_core import build_model, save_json
from statres_core import StatisticsResidualNet

BRANCHES = ("mean", "std")
UPDATES = 1490
PREVIOUS = Path("experiments/specialize_screen_20260908")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, choices=(0,1), required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    destination = ROOT / ("smoke" if args.smoke else "runs") / f"fold{args.fold}"
    destination.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4); cv2.setNumThreads(1); seed(SEED)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    ids, y, tr, va, z, views, old = load_data(args.fold, destination)
    prior = PREVIOUS / "runs" / f"fold{args.fold}"
    common = prior / "common/selected.pt"
    common_result = json.loads((prior / "common/result.json").read_text())
    assert common_result["no_validation_used"] and common_result["training"]["updates"] == 2980
    common_hash = digest(common)
    assert common_hash == common_result["sha256"]
    with np.load(prior / "split_indices.npz") as split:
        np.testing.assert_array_equal(split["train"], tr)
        np.testing.assert_array_equal(split["validation"], va)
    saved = torch.load(common, map_location="cpu", weights_only=True)
    assert saved["config"]["fold"] == args.fold
    assert saved["config"]["initialization"] == "fresh_cpsam_no_CLEAR_EC_checkpoint"
    assert saved["config"]["select_mode"] == "fixed_terminal_no_validation_selection"
    assert saved["config"]["common_updates"] == 2980 and not saved["config"]["smoke"]
    net = build_model("cpsam_models/cpsam_v2", model_class=StatisticsResidualNet)
    for block in net.encoder.blocks:
        original = block.forward
        block.forward = lambda x, fn=original: checkpoint.checkpoint(fn,x,use_reentrant=False)
    config = dict(fold=args.fold, common_sha256=common_hash, common_path=str(common),
        common_updates=2980, branch_updates=2 if args.smoke else UPDATES,
        branch_encoder_lr=3e-6, head_lr=3e-4, seed=SEED+300,
        batch_size=BS, k_crops=K, region_policy="full_quality",
        objective="mixed_mape", residual_outputs=["cv","hex"],
        initialization="verified_same_fold_fixed_terminal_train_only_common",
        select_mode="fixed_terminal_no_validation_selection", smoke=args.smoke,
        split_sha256=digest(old / "split_indices.npz"),
        caveat="Prior folds reused historically; feature variance is not cell-area variance.")
    save_json(destination / "config.json", config)
    sampling = None; results = {}
    for name in BRANCHES:
        branch = destination / name; branch.mkdir()
        missing, unexpected = net.load_state_dict(saved["model"], strict=False)
        assert not unexpected and set(missing) == {
            "residual_norm.weight","residual_norm.bias","residual.0.weight",
            "residual.0.bias","residual.2.weight","residual.2.bias"}
        seed(SEED+200); net.reset_residual(); net.statistic = name
        net.eval()
        with torch.no_grad(), torch.autocast("cuda",dtype=torch.bfloat16):
            x = torch.from_numpy(views.evaluate(int(tr[0]),0)[:2]).to("cuda")
            features = net.encoder(x)
            old_output = net.mean_output(features)
            initial_output = net.forward_features(features)
            torch.testing.assert_close(initial_output, old_output, rtol=0, atol=0)
        del x, features, old_output, initial_output
        training = train(net,views,tr,z,None,config["branch_updates"],branch,3e-6,SEED+300)
        if sampling is None: sampling = training["first_three_batches_sha256"]
        assert sampling == training["first_three_batches_sha256"]
        assert net.residual[-1].weight.detach().abs().sum().item() > 0, "residual never learned"
        torch.save(dict(model=net.state_dict(),config=dict(config,statistic=name)),branch / "selected.pt")
        evaluation = evaluate(net,views,va[:4] if args.smoke else va,y,ids,branch)
        results[name] = dict(training=training,evaluation=evaluation,
            checkpoint_sha256=digest(branch / "selected.pt"),common_sha256=common_hash,
            initial_prediction_exactly_preserved=True, residual_learned=True)
        save_json(branch / "result.json",results[name])
    save_json(destination / "result.json",dict(completed_at=time.strftime("%FT%T%z"),
        branches=results,deployment_approved=False,historical_validation_reuse=True))


if __name__ == "__main__": main()

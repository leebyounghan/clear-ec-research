#!/usr/bin/env python3
"""Fold-clean, matched metric35 initialization test; research only.

Both arms keep central native256 crops, slide-smoothed labels, mixed loss,
seed42, 35 terminal epochs and D8 evaluation. Only encoder initialization
differs. The historical six-member model is never modified.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path("experiments/phase2_metric35_ssl_20260928")
RESEARCH = Path("experiments/raw_ssl_next_20260913")
FROZEN = Path("experiments/crop_dino_screen_20260910/source/frozen")
SSL_DIR = Path("experiments/ssl_screen_20260911/source")
DATA = Path("datasets/clear_ec_official")
FOLDS = (2, 3)
INITIAL_SHA = "0f1cc3f7ecdd8a037a57c6c48d9d8921391be4cbce3fa9f13c3e3a2e1253c667"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path: Path) -> dict:
    return json.loads(Path(path).read_text())


def save(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def frozen_api():
    sys.path.insert(0, str(FROZEN))
    import train_specialize_screen as shared
    shared.K = 1
    return shared


class CentralViews:
    """Exact metric35 central training/evaluation views, plus SSL tile API."""

    def __init__(self, images):
        self.images = images
        self.height = self.width = 512
        self._origins = np.array([(0, 0), (0, 256), (256, 0), (256, 256),
                                  (128, 128), (0, 128), (128, 0), (128, 256)], dtype=np.int32)
        self.selected = np.broadcast_to(self._origins, (len(images), 8, 2))

    def tile(self, index: int, origin, variant: int):
        from final_candidate_core import normalize
        row, col = map(int, origin)
        image = normalize(self.images[int(index), row:row + 256, col:col + 256])
        image = np.rot90(image, int(variant) % 4)
        if int(variant) >= 4:
            image = image[:, ::-1]
        return np.stack([np.ascontiguousarray(image)] * 3)

    def train(self, index: int, k: int = 1):
        from final_candidate_core import train_view
        return np.stack([train_view(self.images[int(index)]) for _ in range(k)])

    def evaluate(self, index: int, d8: int):
        from final_candidate_core import eval_views
        if d8 != 1:
            raise ValueError("Only exact 40-view metric35 evaluation is supported")
        return eval_views(self.images[int(index)])


def slide_targets(ids: np.ndarray, raw: np.ndarray, train: np.ndarray,
                  validation: np.ndarray) -> np.ndarray:
    with (DATA / "final_train_ids.csv").open(newline="") as stream:
        mapping = {Path(str(row["ID"])).stem: str(row["slide_id"]) for row in csv.DictReader(stream)}
    slides = np.array([mapping[Path(x).stem] for x in ids])
    if set(slides[train]) & set(slides[validation]):
        raise ValueError("slide overlap between train and validation")
    target = raw.copy()
    for slide in np.unique(slides[train]):
        members = train[slides[train] == slide]
        if len(members) > 1:
            target[members] = raw[members].mean(axis=0)
    return target


def prepare() -> None:
    if ROOT.exists() and (ROOT / "manifest.json").exists():
        raise FileExistsError(ROOT / "manifest.json")
    old = read(RESEARCH / "manifest.json")
    frozen_files = [ROOT / "source/phase2_metric35_ssl.py",
                    FROZEN / "train_specialize_screen.py",
                    FROZEN / "final_candidate_core.py",
                    FROZEN / "method_training_core.py",
                    SSL_DIR / "ssl_native_core.py"]
    inputs = [DATA / "cache_native_meta_512.npz", DATA / "img_native_512.dat",
              DATA / "final_train_ids.csv", Path("cpsam_models/cpsam_v2"),
              RESEARCH / "manifest.json"]
    if sha(inputs[3]) != INITIAL_SHA:
        raise ValueError("foundation checksum changed")
    with np.load(inputs[0], allow_pickle=True) as d:
        ids, y = d["ids"].astype(str), d["Y"].astype(np.float32)
    if len(ids) != 8954 or len(np.unique(ids)) != len(ids):
        raise ValueError("unexpected dataset")
    images = np.memmap(DATA / "img_native_512.dat", mode="r", dtype=np.uint8,
                       shape=(8954, 512, 512))
    view = CentralViews(images)
    sys.path.insert(0, str(FROZEN))
    from final_candidate_core import eval_views
    for index in (0, 4477, 8953):
        if not np.array_equal(view.evaluate(index, 1), eval_views(images[index])):
            raise ValueError("central D8 parity failure")
        if not np.array_equal(view.tile(index, (0, 0), 0), eval_views(images[index])[0]):
            raise ValueError("central tile parity failure")
    folds = {}
    for fold in FOLDS:
        prior = old["folds"][str(fold)]
        split_path = Path(prior["split_path"])
        if sha(split_path) != prior["split_sha256"] or prior["slide_and_exact_duplicate_overlap"] != 0:
            raise ValueError(f"unclean or changed split {fold}")
        with np.load(split_path) as d:
            tr, va = d["train"].copy(), d["validation"].copy()
        if len(tr) != prior["train_rows"] or len(va) != prior["validation_rows"]:
            raise ValueError(f"split size mismatch {fold}")
        if np.intersect1d(tr, va).size or not np.isfinite(y[tr]).all() or not (y[tr] > 0).all():
            raise ValueError(f"invalid fold {fold}")
        slide_targets(ids, y, tr, va)
        reference = Path(prior["reference_root"]) / "preflight/reference.npz"
        with np.load(reference) as d:
            mask = d["folds"] == fold
            np.testing.assert_array_equal(d["indices"][mask], va)
            np.testing.assert_array_equal(d["ids"][mask].astype(str), ids[va])
            np.testing.assert_array_equal(d["target"][mask], y[va])
            primary = int(d["primary"][mask].sum())
        if primary != prior["primary_rows"]:
            raise ValueError(f"primary subset mismatch {fold}")
        inputs += [split_path, reference]
        folds[str(fold)] = dict(split_path=str(split_path), split_sha256=prior["split_sha256"],
                               train_rows=len(tr), validation_rows=len(va), primary_rows=primary,
                               updates=35 * (len(tr) // 48), reference=str(reference))
    manifest = dict(created_at=time.strftime("%FT%T%z"), folds=folds,
                    source_sha256={str(p): sha(p) for p in frozen_files},
                    input_sha256={str(p): sha(p) for p in inputs},
                    comparison="Original CPSAM versus train-fold-clean 6000-step native masked-EMA initialization; all other metric35 conditions matched",
                    primary="Six-member fixed replacement, primary duplicate-clean validation subset",
                    gate="Both folds improve and pooled mean MAPE improves >=0.01; confirmation on third fold before any promotion",
                    crop="central512 random native256 train, central C5 x D8 evaluation",
                    target="slide-smoothed on clean train rows, original labels on validation",
                    seed=42, k_crops=1, batch=48, objective="mixed_mape",
                    ssl_steps=6000, ssl_batch=32, checkpoint="fixed terminal",
                    max_concurrent_gpu=2, automatic_promotion=False, official_release_untouched=True,
                    caveat="Historical reused folds; only fold-clean training/SSL, not independent challenge score")
    save(ROOT / "manifest.json", manifest)
    print(json.dumps({"prepared": True, "folds": folds, "source_count": len(frozen_files)}, indent=2))


def verify() -> dict:
    manifest = read(ROOT / "manifest.json")
    for group in ("source_sha256", "input_sha256"):
        for path, digest in manifest[group].items():
            if sha(Path(path)) != digest:
                raise ValueError(f"changed {group}: {path}")
    return manifest


def metrics(prediction: np.ndarray, target: np.ndarray) -> list[float]:
    values = []
    for metric in range(3):
        valid = (target[:, metric] > 0) & np.isfinite(target[:, metric])
        if not valid.any() or not np.isfinite(prediction[:, metric]).all():
            raise ValueError(f"invalid metric {metric}")
        values.append(float((np.abs(prediction[valid, metric] - target[valid, metric]) /
                             target[valid, metric]).mean() * 100))
    return values


def build_net(shared, encoder_checkpoint: Path | None):
    import torch
    import torch.utils.checkpoint as checkpoint
    shared.seed(42)
    net = shared.build_model("cpsam_models/cpsam_v2")
    if encoder_checkpoint is not None:
        ck = torch.load(encoder_checkpoint, map_location="cpu", weights_only=True)
        if ck["config"]["validation_images_used"] != 0:
            raise ValueError("validation images entered SSL")
        net.encoder.load_state_dict(ck["teacher_encoder"], strict=True)
    for block in net.encoder.blocks:
        original = block.forward
        block.forward = lambda x, fn=original: checkpoint.checkpoint(fn, x, use_reentrant=False)
    return net


def fit_arm(shared, view, tr, va, ids, raw, smoothed, fold: int,
            arm: str, steps: int, encoder: Path | None, smoke: bool) -> dict:
    import torch
    out = ROOT / ("smoke" if smoke else "runs") / f"fold{fold}" / arm
    out.mkdir(parents=True, exist_ok=False)
    net = build_net(shared, encoder)
    config = dict(fold=fold, arm=arm, seed=42, smoke=smoke, updates=steps,
                  split_sha256=read(ROOT / "manifest.json")["folds"][str(fold)]["split_sha256"],
                  ssl_checkpoint_sha256=sha(encoder) if encoder else None,
                  target="slide-smoothed", region="central512", k_crops=1,
                  objective="mixed_mape", validation_selection=False)
    save(out / "config.json", config)
    np.savez_compressed(out / "split_indices.npz", train=tr, validation=va)
    z = torch.tensor(((np.log(smoothed.clip(1e-6)) - shared.LM) / shared.LS).astype(np.float32), device="cuda")
    timing = shared.train(net, view, tr, z, None, steps, out, 1e-5, 42)
    torch.save(dict(model=net.state_dict(), config=config), out / "selected.pt")
    saved = torch.load(out / "selected.pt", map_location="cpu", weights_only=True)
    net.load_state_dict(saved["model"], strict=True)
    evaluation = shared.evaluate(net, view, va[:4] if smoke else va, raw, ids, out)
    result = dict(training=timing, evaluation=evaluation,
                  checkpoint_sha256=sha(out / "selected.pt"), strict_reload=True,
                  completed_at=time.strftime("%FT%T%z"))
    save(out / "result.json", result)
    return result


def worker(fold: int, smoke: bool) -> None:
    import gc
    import cv2
    import torch
    manifest = verify()
    shared = frozen_api()
    torch.set_num_threads(4); cv2.setNumThreads(1)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    if torch.cuda.device_count() != 1:
        raise RuntimeError("one isolated GPU required")
    prior = manifest["folds"][str(fold)]
    with np.load(prior["split_path"]) as d:
        tr, va = d["train"].copy(), d["validation"].copy()
    with np.load(DATA / "cache_native_meta_512.npz", allow_pickle=True) as d:
        ids, raw = d["ids"].astype(str), d["Y"].astype(np.float32)
    smoothed = slide_targets(ids, raw, tr, va)
    images = np.memmap(DATA / "img_native_512.dat", mode="r", dtype=np.uint8,
                       shape=(8954, 512, 512))
    view = CentralViews(images)
    steps = 3 if smoke else prior["updates"]
    control = fit_arm(shared, view, tr, va, ids, raw, smoothed, fold, "control", steps, None, smoke)
    gc.collect(); torch.cuda.empty_cache()
    pre = ROOT / ("smoke" if smoke else "runs") / f"fold{fold}" / "pretrain"
    pre.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(SSL_DIR))
    from ssl_native_core import pretrain
    shared.seed(42)
    net = shared.build_model("cpsam_models/cpsam_v2")
    ssl = pretrain(net.encoder, view, tr, pre, 4 if smoke else manifest["ssl_steps"], shared)
    del net
    gc.collect(); torch.cuda.empty_cache()
    trained = fit_arm(shared, view, tr, va, ids, raw, smoothed, fold, "ssl", steps,
                      Path(ssl["final_checkpoint"]), smoke)
    if control["training"]["first_three_batches_sha256"] != trained["training"]["first_three_batches_sha256"]:
        raise ValueError("matched batches/crops differ between arms")
    save(ROOT / ("smoke" if smoke else "runs") / f"fold{fold}" / "complete.json",
         dict(fold=fold, smoke=smoke, control_sha256=control["checkpoint_sha256"],
              ssl_sha256=trained["checkpoint_sha256"], ssl_encoder_sha256=ssl["checkpoint_sha256"],
              matched_first_three_batches=True, completed_at=time.strftime("%FT%T%z")))


def audit() -> None:
    manifest = verify()
    groups = []
    for fold in FOLDS:
        prior = manifest["folds"][str(fold)]
        base = ROOT / "runs" / f"fold{fold}"
        complete = read(base / "complete.json")
        if complete["smoke"] or not complete["matched_first_three_batches"]:
            raise ValueError(f"fold {fold} incomplete")
        runs = {arm: np.load(base / arm / "oof.npz", allow_pickle=True) for arm in ("control", "ssl")}
        with np.load(prior["reference"]) as ref:
            fold_mask = ref["folds"] == fold
            primary = ref["primary"][fold_mask].astype(bool)
            for arm, data in runs.items():
                for key in ("indices", "ids", "target"):
                    np.testing.assert_array_equal(data[key].astype(str) if key == "ids" else data[key],
                                                  ref[key][fold_mask].astype(str) if key == "ids" else ref[key][fold_mask])
                if sha(base / arm / "selected.pt") != read(base / arm / "result.json")["checkpoint_sha256"]:
                    raise ValueError(f"changed checkpoint {fold}/{arm}")
            old_metric = np.load(Path("experiments/methods_20260904/runs") /
                                 f"metric35_f{fold}_seed42_c256k1_mean_multitask_slidesmooth/best_oof.npz", allow_pickle=True)
            np.testing.assert_array_equal(old_metric["indices"], ref["indices"][fold_mask])
            np.testing.assert_array_equal(old_metric["ids"].astype(str), ref["ids"][fold_mask].astype(str))
            np.testing.assert_array_equal(old_metric["target"], ref["target"][fold_mask])
            six = np.load(RESEARCH / f"ssl_runs/fold{fold}/standard/oof.npz", allow_pickle=True)
            np.testing.assert_array_equal(six["indices"], ref["indices"][fold_mask])
            np.testing.assert_array_equal(six["ids"].astype(str), ref["ids"][fold_mask].astype(str))
            np.testing.assert_array_equal(six["target"], ref["target"][fold_mask])
            current5 = ref["current5"][fold_mask].astype(float)
            predicted = {arm: runs[arm]["pred"].astype(float) for arm in runs}
            output = {arm: (5 * current5 - old_metric["pred"].astype(float) + predicted[arm] + six["pred"].astype(float)) / 6
                      for arm in runs}
            groups.append(dict(fold=fold, target=ref["target"][fold_mask][primary].astype(float),
                               control=predicted["control"][primary], ssl=predicted["ssl"][primary],
                               control_six=output["control"][primary], ssl_six=output["ssl"][primary],
                               previous_six=(5 * current5 + six["pred"].astype(float))[primary] / 6))
        for data in runs.values():
            data.close()
    y = np.concatenate([g["target"] for g in groups])
    scores = {}
    for key in ("control", "ssl", "control_six", "ssl_six", "previous_six"):
        pred = np.concatenate([g[key] for g in groups])
        scores[key] = dict(overall=float(metrics(pred, y).mean()), per_metric=metrics(pred, y).tolist(),
                           per_fold={str(g["fold"]): float(metrics(g[key], g["target"]).mean()) for g in groups})
    delta = scores["ssl_six"]["overall"] - scores["control_six"]["overall"]
    gate = delta <= -0.01 and all(scores["ssl_six"]["per_fold"][str(f)] < scores["control_six"]["per_fold"][str(f)] for f in FOLDS)
    report = dict(n=len(y), folds=list(FOLDS), scores=scores, delta_ssl_vs_control_six=delta,
                  screening_gate_passed=gate, automatic_promotion=False,
                  caveat="Reused folds; fixed remaining members include historical validation-selected checkpoints. Not independent Phase II prediction.",
                  completed_at=time.strftime("%FT%T%z"))
    save(ROOT / "audit.json", report)
    print(json.dumps(report, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "verify", "worker", "audit"))
    parser.add_argument("--fold", type=int, choices=FOLDS)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.command == "prepare":
        prepare()
    elif args.command == "verify":
        print(json.dumps({"verified": True, "manifest": verify()["created_at"]}))
    elif args.command == "worker":
        if args.fold is None:
            parser.error("worker needs --fold")
        worker(args.fold, args.smoke)
    else:
        audit()


if __name__ == "__main__":
    main()

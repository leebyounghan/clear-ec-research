"""Isolated all-data adapter over the frozen, corrected deployment trainer.

Keeps optimizer/update schedule, single head construction, loss and checkpoint
logic unchanged. Explicit substitutions are saved in each run configuration.
"""
import ast
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

FROZEN = Path("experiments/region_search_20260905/source/code")
sys.path.insert(0, str(FROZEN))
from native_region_views import NativeRegionViews, diverse_origins
from fullframe_views import focus_score

STEPS = {"anchor_quality": 4470, "full_quality": 4917}


class MappedImages:
    def __init__(self, original, mapping, extras):
        self.original, self.mapping, self.extras = original, mapping, extras
        self.shape = (len(mapping), *original.shape[1:])

    def __getitem__(self, key):
        index, *slices = key if isinstance(key, tuple) else (key,)
        image = self.extras[index] if self.mapping[index] < 0 else self.original[self.mapping[index]]
        return image[tuple(slices)] if slices else image


def build_regions(policy):
    import SimpleITK as sitk
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(1)
    data = Path("datasets/clear_ec_final9000_20260904")
    raw = Path("datasets/clear_ec_fullframe_20260904")
    old_meta = np.load("datasets/clear_ec_official/cache_native_meta_512.npz", allow_pickle=True)
    old_ids = old_meta["ids"].astype(str)
    old_central = np.memmap("datasets/clear_ec_official/img_native_512.dat", dtype="uint8", mode="r", shape=(8954, 512, 512))
    views = NativeRegionViews(old_central, old_ids, policy)
    meta = np.load(data / "meta.npz")
    ids = meta["ids"].astype(str)
    assert len(ids) == 9000 and len(set(ids)) == 9000
    assert len(meta["train_indices"]) == 8998
    lookup = {value: index for index, value in enumerate(old_ids)}
    mapping = np.asarray([lookup.get(value, -1) for value in ids])
    assert (mapping >= 0).sum() == 8954
    extras = {}
    scores = np.empty((9000, len(views.origins)), dtype=views.scores.dtype)
    central = np.memmap(data / "img_native_512.dat", dtype="uint8", mode="r", shape=(9000, 512, 512))
    for i, image_id in enumerate(ids):
        j = mapping[i]
        if j >= 0:
            image = views.images[j]
            scores[i] = views.scores[j]
            np.testing.assert_allclose(meta["Y"][i], old_meta["Y"][j], rtol=1e-5)
        else:
            image = sitk.GetArrayFromImage(sitk.ReadImage(str(raw / "train_mha" / (image_id + ".mha"))))
            assert image.shape == (972, 1296) and image.dtype == np.uint8
            extras[i] = image
            scores[i] = [focus_score(image[r:r+256, c:c+256]) for r, c in views.origins]
        r, c = views.central_origin
        assert np.array_equal(image[r:r+512, c:c+512], central[i]), image_id
    original = views.images
    original_selected = views.selected
    views.images = MappedImages(original, mapping, extras)
    views.scores = scores
    views.selected = np.stack([diverse_origins(views.origins, s, 8) for s in scores])
    np.testing.assert_array_equal(views.selected[mapping >= 0], original_selected[mapping[mapping >= 0]])
    # Exact D8 pixels and normalization for a fixed shared image, before fitting.
    reference = NativeRegionViews(old_central, old_ids, policy)
    ix = int(np.flatnonzero(mapping >= 0)[0])
    np.testing.assert_array_equal(views.evaluate(ix, 1), reference.evaluate(int(mapping[ix]), 1))
    print("raw ID/label/central pixel alignment 9000; shared quality origins 8954; fixed D8 view equality passed", flush=True)
    return views


def check_head_construction():
    import torch
    from torch import nn
    from final_candidate_core import CandidateNet
    tree = ast.parse((FROZEN / "train_region_method.py").read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "ScalarNet")
    ns = dict(nn=nn, torch=torch, BAG_POOL="mean")
    exec(compile(ast.Module(body=[cls], type_ignores=[]), "frozen_scalar_class", "exec"), ns)
    torch.manual_seed(77)
    reference = ns["ScalarNet"](nn.Identity(), "mean")
    torch.manual_seed(77)
    candidate = CandidateNet(nn.Identity())
    assert reference.state_dict().keys() == candidate.state_dict().keys()
    for key, value in reference.state_dict().items():
        assert torch.equal(value, candidate.state_dict()[key]), key
    print("single head initialization equals frozen OOF class", flush=True)


def main():
    policy = sys.argv[sys.argv.index("--recipe") + 1]
    assert policy in STEPS
    check_head_construction()
    original = FROZEN / "train_deploy_methods.py"
    source = original.read_text()
    assert 'initialization="single_head_construction_v2"' in source
    replacements = {
        "seed = 42": "seed = 77",
        'images = np.memmap(DATA / "img_native_512.dat", dtype=np.uint8, mode="r", shape=(9000, 512, 512))':
            "images = build_regions(args.recipe)",
        'initialization="single_head_construction_v2",':
            'initialization="single_head_construction_v2", region_policy=args.recipe, adapter_sha256=ADAPTER_HASH, adapter_substitutions=SUBSTITUTIONS,',
    }
    for before, after in replacements.items():
        assert source.count(before) == 1, before
        source = source.replace(before, after)
    ns = dict(__name__="isolated_regional_deployment", __file__=str(original),
              build_regions=build_regions, ADAPTER_HASH=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              SUBSTITUTIONS=replacements)
    exec(compile(source, str(original), "exec"), ns)
    ns["RECIPES"] = {p: dict(selected_steps=steps, k_crops=2, objective="mixed_mape") for p, steps in STEPS.items()}
    ns["training_array"] = lambda views, indices, k: np.stack([views.train(int(i), k) for i in indices])
    ns["main"]()


if __name__ == "__main__":
    main()

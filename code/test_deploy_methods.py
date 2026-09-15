import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import random
import copy
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from final_candidate_core import CandidateNet, LM, LS, build_model
from train_deploy_methods import PredictionBagNet, RECIPES, build_recipe_model, capture_training_state, restore_training_state
from audit_hybrid_method import comparison_gate
from types import SimpleNamespace


class Encoder(nn.Module):
    def forward(self, x):
        return nn.functional.adaptive_avg_pool2d(x.mean(1, keepdim=True), (2, 2)).expand(-1, 256, -1, -1)


class Tests(unittest.TestCase):
    def test_exact_training_resume(self):
        torch.manual_seed(42); np.random.seed(42); random.seed(42)
        net = nn.Sequential(nn.Linear(3, 4), nn.Dropout(.2), nn.Linear(4, 1))
        opt = torch.optim.AdamW(net.parameters(), lr=.01)
        sched = torch.optim.lr_scheduler.StepLR(opt, 1, .9)
        def update():
            x = torch.rand(8, 3) + float(np.random.rand()) + random.random()
            opt.zero_grad(); loss = net(x).square().mean(); loss.backward(); opt.step(); sched.step()
        update(); saved = copy.deepcopy(capture_training_state(net, opt, sched))
        update(); expected = copy.deepcopy(net.state_dict()); lr = sched.get_last_lr()
        restore_training_state(saved, net, opt, sched); update()
        for key in expected:
            torch.testing.assert_close(net.state_dict()[key], expected[key], rtol=0, atol=0)
        self.assertEqual(lr, sched.get_last_lr())

    def interface(self):
        source = Path(__file__).parent.parent / "submission_candidate_20260905" / "inference.py"
        spec = importlib.util.spec_from_file_location("deploy_interface", source)
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        return module

    def test_single_initialization_and_rng_match(self):
        import cellpose.models
        def fake_base(**kwargs):
            return SimpleNamespace(net=SimpleNamespace(encoder=Encoder()))
        with patch.object(cellpose.models, "CellposeModel", side_effect=fake_base):
            torch.manual_seed(42)
            ordinary = build_model("unused", device="cpu")
            state = torch.get_rng_state()
            torch.manual_seed(42)
            with patch("train_deploy_methods.build_model", side_effect=lambda path, model_class: build_model(path, device="cpu", model_class=model_class)):
                bag = build_recipe_model("predbag2_35")
            self.assertIsInstance(bag, PredictionBagNet)
            self.assertTrue(torch.equal(state, torch.get_rng_state()))
            for name, value in ordinary.state_dict().items():
                torch.testing.assert_close(value, bag.state_dict()[name], rtol=0, atol=0)

    def test_strict_metadata_rejects_wrong_normalization_and_feature_bag(self):
        module = self.interface()
        config = dict(pool="mean", k_crops=2, lm=LM.tolist(), ls=LS.tolist())
        net = CandidateNet(Encoder())
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "model.pt"
            torch.save(dict(config=config, model=net.state_dict()), p)
            module.load_strict(net, p)
            for change in (dict(lm=[0., 0., 0.]), dict(feature_agg=1), dict(bag_pool="meanstd")):
                torch.save(dict(config=dict(config, **change), model=net.state_dict()), p)
                with self.assertRaises((ValueError, AssertionError)):
                    module.load_strict(net, p)

    def test_candidate_configuration_fails_closed(self):
        module = self.interface()
        config = dict(members=module.MEMBERS["candidate5.json"], view_batch=8, calibration=[1.,1.,1.])
        module.validate_config(config, "candidate5.json")
        for change in (dict(members=["seed42.pt"]*5), dict(calibration=[float("nan"),1.,1.]), dict(calibration=[1.]), dict(view_batch=40)):
            with self.assertRaises(ValueError):
                module.validate_config(dict(config, **change), "candidate5.json")

    def test_hybrid_cannot_pass_against_only_old_baseline(self):
        def score(x):
            return dict(raw_overall=x, exploratory_crossfit_overall=x-.03, per_fold_raw=[x]*5)
        self.assertFalse(comparison_gate(dict(blend=score(9.04),current4=score(9.045),current5=score(9.035))))
        self.assertTrue(comparison_gate(dict(blend=score(9.02),current4=score(9.045),current5=score(9.035))))

    def test_recipe_steps_are_frozen(self):
        self.assertEqual(RECIPES["metric35"], {"selected_steps": 4470, "k_crops": 1, "objective": "mixed_mape"})
        self.assertEqual(RECIPES["predbag2_35"], {"selected_steps": 3874, "k_crops": 2, "objective": "log_l1"})

    def test_bag_checkpoint_keys_and_prediction_average(self):
        ordinary = CandidateNet(Encoder()).eval()
        bag = PredictionBagNet(Encoder()).eval()
        for parameter in ordinary.parameters():
            nn.init.uniform_(parameter, -0.1, 0.1)
        bag.load_state_dict(ordinary.state_dict(), strict=True)
        self.assertEqual(set(ordinary.state_dict()), set(bag.state_dict()))
        inputs = torch.rand(2, 2, 3, 16, 16)
        expected = ordinary(inputs.flatten(0, 1)).reshape(2, 2, 3).mean(1)
        torch.testing.assert_close(bag(inputs), expected)

    def test_interface_config_selection(self):
        source = Path(__file__).parent.parent / "submission_candidate_20260905" / "inference.py"
        spec = importlib.util.spec_from_file_location("deploy_interface", source)
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        import SimpleITK as sitk
        with tempfile.TemporaryDirectory() as name:
            root = Path(name); input_root = root / "input"
            image_dir = input_root / "images/corneal-specular-microscopy-image"
            image_dir.mkdir(parents=True)
            (input_root / "inputs.json").write_text(json.dumps([{"socket": {"slug": "corneal-specular-microscopy-image"}}]))
            sitk.WriteImage(sitk.GetImageFromArray(np.zeros((972, 1296), np.uint8)), str(image_dir / "a.mha"))
            model = root / "model"; model.mkdir()
            (model / "candidate5.json").write_text(json.dumps({"members": [1,2,3,4,5]}))
            argv = ["inference.py", "--input", str(input_root), "--output", str(root / "output"),
                    "--model", str(model), "--config", "candidate5.json"]
            with patch.object(sys, "argv", argv), patch.object(module, "predict", return_value=np.array([2700., 0.3, 0.6])):
                module.main()
            self.assertEqual(len(list((root / "output").glob("*.json"))), 3)


if __name__ == "__main__":
    unittest.main()

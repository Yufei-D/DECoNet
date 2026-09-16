from copy import deepcopy
from pathlib import Path
import subprocess
from types import SimpleNamespace
import unittest

import torch


ROOT = Path(__file__).resolve().parents[1]
MODEL_CFG = ROOT / "configs" / "models" / "yolo12n-deconet.yaml"


class DECoNetTests(unittest.TestCase):
    def test_deconet_builds_and_uses_training_only_auxiliary_branch(self):
        from ultralytics import YOLO
        from ultralytics.nn.modules.head import DetectAuxTGFA
        from ultralytics.nn.modules.tgfa import TGFABlock
        from ultralytics.nn.modules.wscdown import WSCDownDual

        model = YOLO(str(MODEL_CFG)).model
        if isinstance(model.args, dict):
            model.args = SimpleNamespace(**model.args)
        self.assertIsInstance(model.model[-1], DetectAuxTGFA)
        self.assertEqual(sum(isinstance(module, WSCDownDual) for module in model.modules()), 5)

        criterion = model.init_criterion()
        self.assertEqual(criterion.c1_burn_in, 5000)
        self.assertEqual(criterion.c1_tau_scheduler.tau_start, 0.45)
        self.assertEqual(criterion.c1_tau_scheduler.tau_end, 0.30)
        self.assertEqual(criterion.c1_tau_m, 0.30)
        self.assertEqual(criterion.c2_high_scheduler.tau_start, 0.50)
        self.assertEqual(criterion.c2_high_scheduler.tau_end, 0.40)
        self.assertEqual(criterion.c2_low_scheduler.tau_start, 0.35)
        self.assertEqual(criterion.c2_low_scheduler.tau_end, 0.25)
        self.assertEqual(criterion.c2_m_thresh, 0.25)
        self.assertEqual(criterion.c2_weight_high, 0.75)
        self.assertEqual(criterion.c2_weight_low, 0.50)
        self.assertEqual(criterion.c3_alpha, 0.10)
        self.assertEqual(criterion.aux_weight, 0.25)

        head = model.model[-1]
        training_only_parameters = sum(parameter.numel() for parameter in head.tgfa.parameters())
        training_only_parameters += sum(parameter.numel() for parameter in head.aux_cv2.parameters())
        training_only_parameters += sum(parameter.numel() for parameter in head.aux_cv3.parameters())
        total_parameters = sum(parameter.numel() for parameter in model.parameters())
        self.assertEqual(total_parameters - training_only_parameters, 2_276_977)

        from ultralytics.utils.torch_utils import get_flops

        self.assertAlmostEqual(get_flops(model.eval(), 640), 5.669888, places=5)

        image = torch.zeros(1, 3, 128, 128)
        batch = {
            "img": image,
            "batch_idx": torch.tensor([0]),
            "cls": torch.tensor([[0.0]]),
            "bboxes": torch.tensor([[0.5, 0.5, 0.25, 0.25]]),
            "im_file": ["synthetic.jpg"],
        }
        object.__setattr__(model, "teacher_model", deepcopy(model).eval())
        criterion.pre_forward_teacher_inject(image, batch)
        model.train()
        train_output = model(image)
        self.assertEqual(set(train_output), {"main", "aux"})
        self.assertEqual(len(train_output["main"]), 3)
        self.assertEqual(len(train_output["aux"]), 3)
        loss, loss_items = criterion(train_output, batch)
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(loss_items.shape, (3,))

        def fail_if_called(*_args, **_kwargs):
            raise AssertionError("TGFA must not run in the inference graph")

        for module in model.modules():
            if isinstance(module, TGFABlock):
                module.forward = fail_if_called

        model.eval()
        prediction, features = model(torch.zeros(1, 3, 128, 128))
        self.assertEqual(prediction.shape[0], 1)
        self.assertEqual(len(features), 3)

    def test_git_tracks_no_model_or_data_artifacts(self):
        tracked = subprocess.run(
            ["git", "ls-files"], cwd=ROOT, check=True, capture_output=True, text=True
        ).stdout.splitlines()
        required_sources = {
            "ultralytics/data/m_heatmap_utils.py",
            "ultralytics/nn/modules/tgfa.py",
            "ultralytics/nn/modules/wscdown.py",
            "ultralytics/utils/deconet_loss.py",
        }
        self.assertTrue(required_sources.issubset(tracked))
        forbidden_suffixes = {
            ".pt",
            ".pth",
            ".ckpt",
            ".safetensors",
            ".onnx",
            ".engine",
            ".npy",
            ".npz",
        }
        offenders = [path for path in tracked if Path(path).suffix.lower() in forbidden_suffixes]
        self.assertEqual(offenders, [])


if __name__ == "__main__":
    unittest.main()

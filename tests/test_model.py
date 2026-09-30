"""Fast invariants for V0 self/memory paths and detached distillation."""

import unittest

import torch
import torch.nn.functional as F

from model import DualAxisModel
from train import memory_to_self_kl, source_ce_loss


class ModelTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(4)
        self.x = torch.randn(3, 1, 48, 13, 13)
        self.y = torch.tensor([0, 1, 1])

    def test_ablation_shapes(self):
        for arm in ("a0", "a1", "a2", "a3"):
            model = DualAxisModel(variant=arm, channels=16, dim=32)
            result = model(self.x)
            self.assertEqual(result["logits_self"].shape, (3, 7))
            self.assertEqual(result["z_cls_self"].shape, (3, 32))
            if arm != "a0":
                self.assertEqual(result["spatial_attn"].shape, (3, 4, 169))
            if arm in ("a2", "a3"):
                self.assertEqual(result["spectral_attn"].shape, (3, 4, 12))

    def test_memory_slot_mask_and_gradient(self):
        model = DualAxisModel(variant="a3", channels=16, dim=32)
        initial = model(self.x, use_memory=False)
        self.assertTrue(torch.equal(initial["logits_self"], initial["logits_memory"]))
        model.update_memory(initial["spatial_tokens"], initial["spectral_tokens"], self.y)
        before = model.memory_spatial.clone()
        result = model(self.x)
        self.assertTrue(torch.equal(before, model.memory_spatial))
        self.assertEqual(result["logits_memory"].shape, (3, 7))
        weights = result["memory_spatial_attn"]
        self.assertEqual(weights.shape, (3, 4, 7))
        self.assertTrue(torch.all(weights[:, :, 2:] == 0))
        self.assertTrue(torch.allclose(weights.sum(-1), torch.ones(3, 4)))
        (F.cross_entropy(result["logits_self"], self.y) +
         F.cross_entropy(result["logits_memory"], self.y)).backward()
        self.assertIsNotNone(model.alpha_spatial.grad)
        self.assertIsNotNone(model.spatial_memory_reader.in_proj_weight.grad)

    def test_a2_lite_class_query_reads_both_axes_without_fusion(self):
        model = DualAxisModel(variant="a2_lite", channels=16, dim=32)
        self.assertFalse(hasattr(model, "spatial_fusion"))
        self.assertFalse(hasattr(model, "spectral_fusion"))
        result = model(self.x)
        self.assertEqual(result["spatial_tokens"].shape, (3, 4, 32))
        self.assertEqual(result["spectral_tokens"].shape, (3, 4, 32))
        self.assertEqual(result["class_attn_self"].shape, (3, 8))
        self.assertTrue(torch.allclose(result["class_attn_self"].sum(1), torch.ones(3)))
        F.cross_entropy(result["logits_self"], self.y).backward()
        self.assertIsNotNone(model.spectral_queries.grad)
        self.assertGreater(model.spectral_queries.grad.abs().sum().item(), 0)

    def test_distillation_detaches_memory_teacher(self):
        student = torch.randn(3, 7, requires_grad=True)
        teacher = torch.randn(3, 7, requires_grad=True)
        memory_to_self_kl(student, teacher).backward()
        self.assertIsNotNone(student.grad)
        self.assertIsNone(teacher.grad)

    def test_equal_source_branches_match_single_ce_gradient(self):
        logits = torch.randn(3, 7, requires_grad=True)
        single = F.cross_entropy(logits, self.y)
        single_gradient = torch.autograd.grad(single, logits, retain_graph=True)[0]
        averaged, self_ce, memory_ce = source_ce_loss(logits, self.y, logits)
        averaged_gradient = torch.autograd.grad(averaged, logits)[0]
        self.assertTrue(torch.allclose(averaged_gradient, single_gradient))
        self.assertTrue(torch.equal(self_ce, memory_ce))


if __name__ == "__main__":
    unittest.main()

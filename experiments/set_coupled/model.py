"""A1 backbone with a training-only set-level bidirectional coupled branch."""

import sys
from pathlib import Path

import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.v0.model import DualAxisModel


class SetCoupledA1(nn.Module):
    """Each sample's four queries retrieve from every token in the other batch."""

    def __init__(self, bands=48, classes=7, patch_size=13, dim=128,
                 heads=4, alpha=1.0, learnable_alpha=False):
        super().__init__()
        self.backbone = DualAxisModel(bands=bands, classes=classes,
                                      patch_size=patch_size, variant="a1", dim=dim,
                                      heads=heads)
        self.query_norm = nn.LayerNorm(dim)
        self.context_norm = nn.LayerNorm(dim)
        self.cross_attention = nn.MultiheadAttention(dim, heads, batch_first=True)
        alpha_tensor = torch.tensor(float(alpha))
        if learnable_alpha:
            self.alpha = nn.Parameter(alpha_tensor)
        else:
            self.register_buffer("alpha", alpha_tensor)

    def self_forward(self, images):
        return self.backbone(images)

    def _cross(self, queries, context):
        """Cross attention to all B*K context tokens, shared by every sample."""
        batch, slots, dim = queries.shape
        context_set = context.reshape(1, -1, dim).expand(batch, -1, -1)
        read, attention = self.cross_attention(
            self.query_norm(queries), self.context_norm(context_set),
            self.context_norm(context_set), need_weights=True)
        return queries + self.alpha * read, attention

    def forward(self, source_images, target_images):
        source = self.backbone(source_images)
        target = self.backbone(target_images)
        source_cross_tokens, source_to_target_attention = self._cross(
            source["spatial_tokens"], target["spatial_tokens"])
        target_cross_tokens, target_to_source_attention = self._cross(
            target["spatial_tokens"], source["spatial_tokens"])
        source_cross_logits, source_cross_z, source_cross_class_attention = \
            self.backbone._classify(source_cross_tokens, None)
        target_cross_logits, target_cross_z, target_cross_class_attention = \
            self.backbone._classify(target_cross_tokens, None)
        return {
            "source_self_logits": source["logits_self"],
            "source_cross_logits": source_cross_logits,
            "target_self_logits": target["logits_self"],
            "target_cross_logits": target_cross_logits,
            "source_self_z": source["z_cls_self"],
            "source_cross_z": source_cross_z,
            "target_self_z": target["z_cls_self"],
            "target_cross_z": target_cross_z,
            "source_tokens": source["spatial_tokens"],
            "target_tokens": target["spatial_tokens"],
            "source_cross_tokens": source_cross_tokens,
            "target_cross_tokens": target_cross_tokens,
            "source_to_target_attention": source_to_target_attention,
            "target_to_source_attention": target_to_source_attention,
            "source_cross_class_attention": source_cross_class_attention,
            "target_cross_class_attention": target_cross_class_attention,
            "alpha": self.alpha,
        }

"""Dual-axis semantic memory network and its A0--A3/A2-lite ablations."""

import math

import torch
from torch import nn


def sinusoidal_positions(length, dim, device, dtype):
    position = torch.arange(length, device=device, dtype=torch.float32)[:, None]
    frequency = torch.exp(torch.arange(0, dim, 2, device=device, dtype=torch.float32) * (-math.log(10000.0) / dim))
    encoding = torch.zeros(length, dim, device=device, dtype=torch.float32)
    encoding[:, 0::2] = torch.sin(position * frequency)
    encoding[:, 1::2] = torch.cos(position * frequency[:encoding[:, 1::2].shape[1]])
    return encoding.to(dtype)


class CrossBlock(nn.Module):
    def __init__(self, dim, heads=4):
        super().__init__()
        self.q_norm = nn.LayerNorm(dim)
        self.kv_norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.ff_norm = nn.LayerNorm(dim)
        self.ff = nn.Sequential(nn.Linear(dim, 2 * dim), nn.GELU(), nn.Linear(2 * dim, dim))

    def forward(self, query, context):
        delta, weights = self.attn(self.q_norm(query), self.kv_norm(context), self.kv_norm(context), need_weights=True)
        result = query + delta
        return result + self.ff(self.ff_norm(result)), weights


class DualAxisModel(nn.Module):
    def __init__(self, bands=48, classes=7, patch_size=13, variant="a3", channels=64,
                 dim=128, spectral_bins=12, spatial_slots=4, spectral_slots=4,
                 heads=4, memory_momentum=0.9):
        super().__init__()
        if variant not in {"a0", "a1", "a2_lite", "a2", "a3"}:
            raise ValueError(f"Unknown ablation: {variant}")
        if bands < 4 or patch_size < 1 or spectral_bins < 1 or dim % heads:
            raise ValueError("Invalid input dimensions or attention head count")
        self.variant = variant
        self.classes = classes
        self.memory_momentum = memory_momentum
        self.spectral_bins = spectral_bins
        self.stem = nn.Sequential(
            nn.Conv3d(1, 32, (5, 1, 1), padding=(2, 0, 0)), nn.GroupNorm(8, 32), nn.GELU(),
            nn.Conv3d(32, 64, (1, 3, 3), padding=(0, 1, 1)), nn.GroupNorm(8, 64), nn.GELU(),
            nn.Conv3d(64, channels, (5, 1, 1), padding=(2, 0, 0)), nn.GroupNorm(8, channels), nn.GELU(),
        )
        self.spectral_downsample = nn.AdaptiveAvgPool3d((spectral_bins, patch_size, patch_size))
        if variant == "a0":
            self.pool_head = nn.Sequential(nn.Linear(channels, dim), nn.LayerNorm(dim), nn.GELU())
        else:
            self.spatial_projection = nn.Conv3d(channels, dim, (spectral_bins, 1, 1))
            self.spatial_queries = nn.Parameter(torch.randn(spatial_slots, dim) * 0.02)
            self.spatial_reader = CrossBlock(dim, heads)
            if variant in {"a2_lite", "a2", "a3"}:
                self.spectral_projection = nn.Linear(channels, dim)
                self.spectral_queries = nn.Parameter(torch.randn(spectral_slots, dim) * 0.02)
                self.spectral_reader = CrossBlock(dim, heads)
                if variant in {"a2", "a3"}:
                    self.spatial_fusion = CrossBlock(dim, heads)
                    self.spectral_fusion = CrossBlock(dim, heads)
            self.class_query = nn.Parameter(torch.randn(1, dim) * 0.02)
            self.class_reader = CrossBlock(dim, heads)
        self.classifier = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, classes))
        if variant == "a3":
            self.spatial_memory_reader = nn.MultiheadAttention(dim, heads, batch_first=True)
            self.spectral_memory_reader = nn.MultiheadAttention(dim, heads, batch_first=True)
            self.alpha_spatial = nn.Parameter(torch.tensor(0.01))
            self.alpha_spectral = nn.Parameter(torch.tensor(0.01))
            self.register_buffer("memory_spatial", torch.zeros(classes, spatial_slots, dim))
            self.register_buffer("memory_spectral", torch.zeros(classes, spectral_slots, dim))
            self.register_buffer("memory_seen", torch.zeros(classes, dtype=torch.bool))

    def _classify(self, spatial, spectral):
        tokens = spatial if spectral is None else torch.cat((spatial, spectral), dim=1)
        z_cls, class_attn = self.class_reader(self.class_query.unsqueeze(0).expand(tokens.shape[0], -1, -1), tokens)
        z_cls = z_cls[:, 0]
        return self.classifier(z_cls), z_cls, class_attn[:, 0]

    def _memory_read(self, tokens, bank, reader):
        batch, slots, dim = tokens.shape
        # Each slot attends only to the same slot across valid source classes.
        queries = tokens.transpose(0, 1).reshape(slots * batch, 1, dim)
        keys = bank.transpose(0, 1)[:, None].expand(slots, batch, self.classes, dim)
        keys = keys.reshape(slots * batch, self.classes, dim)
        mask = (~self.memory_seen)[None].expand(slots * batch, -1)
        read, attention = reader(queries, keys, keys, key_padding_mask=mask, need_weights=True)
        read = read.reshape(slots, batch, dim).transpose(0, 1)
        attention = attention.reshape(slots, batch, self.classes).transpose(0, 1)
        return read, attention

    def forward(self, x, use_memory=True):
        if x.ndim != 5 or x.shape[1] != 1:
            raise ValueError("Expected HSI patches [N, 1, B, H, W]")
        cube = self.stem(x)
        cube = self.spectral_downsample(cube)
        output = {key: None for key in ("spatial_tokens", "spectral_tokens", "spatial_attn",
                  "spectral_attn", "memory_spatial_attn", "memory_spectral_attn",
                  "alpha_spatial", "alpha_spectral")}
        if self.variant == "a0":
            z_cls = self.pool_head(cube.mean(dim=(2, 3, 4)))
            logits = self.classifier(z_cls)
            output.update(logits_self=logits, logits_memory=None, z_cls_self=z_cls,
                          z_cls_memory=None, class_attn_self=None,
                          class_attn_memory=None, logits=logits, z_cls=z_cls)
            return output

        batch, _, _, height, width = cube.shape
        spatial_pixels = self.spatial_projection(cube).squeeze(2).flatten(2).transpose(1, 2)
        half = spatial_pixels.shape[-1] // 2
        row = sinusoidal_positions(height, half, x.device, x.dtype)
        col = sinusoidal_positions(width, spatial_pixels.shape[-1] - half, x.device, x.dtype)
        position_2d = torch.cat((row[:, None].expand(-1, width, -1),
                                 col[None].expand(height, -1, -1)), dim=-1)
        spatial_pixels = spatial_pixels + position_2d.reshape(1, height * width, -1)
        spatial, spatial_attn = self.spatial_reader(self.spatial_queries[None].expand(batch, -1, -1), spatial_pixels)
        spectral = None
        if self.variant in {"a2_lite", "a2", "a3"}:
            spectral_pixels = self.spectral_projection(cube.mean(dim=(3, 4)).transpose(1, 2))
            spectral_pixels = spectral_pixels + sinusoidal_positions(self.spectral_bins, spectral_pixels.shape[-1], x.device, x.dtype)
            spectral, spectral_attn = self.spectral_reader(self.spectral_queries[None].expand(batch, -1, -1), spectral_pixels)
            if self.variant in {"a2", "a3"}:
                # Both fusion directions read the same pre-fusion tokens.
                spatial_new, _ = self.spatial_fusion(spatial, spectral)
                spectral_new, _ = self.spectral_fusion(spectral, spatial)
                spatial, spectral = spatial_new, spectral_new
            output["spectral_attn"] = spectral_attn
        output["spatial_attn"] = spatial_attn
        output["spatial_tokens"] = spatial
        output["spectral_tokens"] = spectral
        if self.variant == "a3":
            output["alpha_spatial"] = self.alpha_spatial
            output["alpha_spectral"] = self.alpha_spectral
        logits_self, z_cls_self, class_attn_self = self._classify(spatial, spectral)
        output["logits_self"] = logits_self
        output["z_cls_self"] = z_cls_self
        output["class_attn_self"] = class_attn_self
        output["logits"] = logits_self
        output["z_cls"] = z_cls_self
        output["logits_memory"] = None
        output["z_cls_memory"] = None
        output["class_attn_memory"] = None
        if self.variant == "a3":
            if use_memory and bool(self.memory_seen.any()):
                spatial_read, spatial_weights = self._memory_read(spatial, self.memory_spatial, self.spatial_memory_reader)
                spectral_read, spectral_weights = self._memory_read(spectral, self.memory_spectral, self.spectral_memory_reader)
                spatial_memory = spatial + self.alpha_spatial * spatial_read
                spectral_memory = spectral + self.alpha_spectral * spectral_read
                output["memory_spatial_attn"] = spatial_weights
                output["memory_spectral_attn"] = spectral_weights
                (output["logits_memory"], output["z_cls_memory"],
                 output["class_attn_memory"]) = self._classify(spatial_memory, spectral_memory)
            else:
                # Before the first source update, the two branches coincide.
                output["logits_memory"] = logits_self
                output["z_cls_memory"] = z_cls_self
                output["class_attn_memory"] = class_attn_self
        return output

    @torch.no_grad()
    def update_memory(self, spatial, spectral, labels):
        if self.variant != "a3":
            raise RuntimeError("Only A3 has semantic memory")
        for class_id in labels.unique().tolist():
            selected = labels == class_id
            mean_spatial = spatial[selected].detach().mean(0)
            mean_spectral = spectral[selected].detach().mean(0)
            if self.memory_seen[class_id]:
                self.memory_spatial[class_id].lerp_(mean_spatial, 1 - self.memory_momentum)
                self.memory_spectral[class_id].lerp_(mean_spectral, 1 - self.memory_momentum)
            else:
                self.memory_spatial[class_id].copy_(mean_spatial)
                self.memory_spectral[class_id].copy_(mean_spectral)
                self.memory_seen[class_id] = True

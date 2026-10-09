"""Minimal changes to the frozen native DCRN_02 forward; no head/loss changes."""
import inspect
import textwrap
from types import MethodType

import torch
from torch import nn


ACTIVE_BN = tuple('bn%d' % i for i in range(1, 8))


class RelationPool(nn.Module):
    def __init__(self, bands=48):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(bands, 32), nn.ReLU(), nn.Linear(32, 4))
        self.uniform = False

    def weights(self, patch):
        assert patch.ndim == 4 and patch.shape[1:] == (48, 7, 7)
        anchor = patch[..., 3:4, 3:4]
        # Same population-standard-deviation formula as the Houston relation arm.
        relation = (patch - anchor) / (patch.std((-2, -1), correction=0, keepdim=True) + 1e-6)
        return self.mlp(relation.permute(0, 2, 3, 1)).permute(0, 3, 1, 2).flatten(2).softmax(-1)

    def forward(self, values, patch):
        assert values.shape[1:] == (288, 7, 7)
        assert values.shape[0] == patch.shape[0]
        if self.uniform:
            # Verification-only: arithmetic reference for spatially uniform weights.
            weights = values.new_full((values.shape[0], 4, 49), 1 / 49)
        else:
            weights = self.weights(patch)
        return torch.einsum('bkn,bcn->bkc', weights, values.flatten(2)).mean(1)


def install(model, joint=False, relation=False, seed=0):
    feature = model.feature_layers
    if not joint and not relation:
        return model
    source = textwrap.dedent(inspect.getsource(type(feature).forward))
    source = source.replace('def forward(self, x,y):', 'def forward(self, x,y):\n    raw_x, raw_y = x, y')
    if joint:
        # CNN branches run once on a same-view S/T concatenation. Split BEFORE
        # CA/SA and dropout cross-attention, preserving their native call order.
        start = source.index('    y = y.unsqueeze(1)')
        end = source.index('    x = torch.cat((x1, x2), 1)', start)
        source = source[:start] + '    x1, y1 = x1.split((raw_x.shape[0], raw_y.shape[0]), dim=0)\n    x2, y2 = x2.split((raw_x.shape[0], raw_y.shape[0]), dim=0)\n' + source[end:]
        source = source.replace('    x = x.unsqueeze(1)', '    x = torch.cat((raw_x, raw_y), dim=0).unsqueeze(1)', 1)
        for name in ACTIVE_BN:
            assert getattr(feature, name).momentum == .1
            getattr(feature, name).momentum = .19
    if relation:
        # Construct only after the complete native model, with an independent RNG.
        with torch.random.fork_rng(devices=[]):
            torch.set_rng_state(torch.Generator().manual_seed(seed + 41000).get_state())
            feature.relation_pool = RelationPool()
        source = source.replace('self.avgpool(x)', 'self.relation_pool(x, raw_x)')
        source = source.replace('self.avgpool(y)', 'self.relation_pool(y, raw_y)')
    namespace = {'torch': torch}
    exec(compile(source, '<audited-native-feature-forward>', 'exec'), namespace)
    feature.forward = MethodType(namespace['forward'], feature)
    feature._derived_source = source
    return model


def bn_counters(model):
    return {name: int(layer.num_batches_tracked) for name, layer in
            model.feature_layers.named_modules() if isinstance(layer, nn.BatchNorm3d)}

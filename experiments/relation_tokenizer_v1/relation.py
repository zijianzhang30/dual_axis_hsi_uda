"""Replace only BiDA's spatial weighting; preserve CNN token values."""
from types import MethodType
import torch
from torch import nn

EPS = 1e-6


class RelationScores(nn.Module):
    def __init__(self, arm, bands=48, hidden=32, tokens=4):
        super().__init__()
        if arm not in ('B', 'C'):
            raise ValueError(arm)
        self.arm = arm
        self.mlp = nn.Sequential(nn.Linear(bands, hidden), nn.ReLU(), nn.Linear(hidden, tokens))

    def relations(self, patch):
        if patch.ndim != 5 or patch.shape[1:3] != (1, 48):
            raise ValueError(f'Expected B×1×48×H×W: {patch.shape}')
        x = patch[:, 0]
        std = x.std(dim=(-2, -1), correction=0, keepdim=True)
        anchor = x.mean(dim=(-2, -1), keepdim=True) if self.arm == 'B' else x[..., x.shape[-2]//2:x.shape[-2]//2+1, x.shape[-1]//2:x.shape[-1]//2+1]
        return (x-anchor)/(std+EPS)

    def forward(self, patch):
        relation = self.relations(patch).permute(0, 2, 3, 1)
        scores = self.mlp(relation).permute(0, 3, 1, 2)
        return scores.flatten(2).softmax(dim=-1)


def semantic_tokens(self, features):
    if not self._relation_pending:
        raise RuntimeError('Relation tokenizer must run through paired model.forward')
    patch = self._relation_pending.pop(0)
    if features.shape[0] != patch.shape[0] or features.shape[-2:] != patch.shape[-2:]:
        raise RuntimeError('CNN value grid and relation scores are not exactly aligned')
    weights = self.relation_scores(patch)
    return torch.einsum('bln,bcn->blc', weights, features.flatten(2))


def forward(self, source, target, *args, **kwargs):
    if self._relation_pending:
        raise RuntimeError('Nested relation-tokenizer forward')
    self._relation_pending = [source, target]
    try:
        result = self._relation_forward_func(self, source, target, *args, **kwargs)
        if self._relation_pending:
            raise RuntimeError('Both domain patches must be consumed')
        return result
    finally:
        self._relation_pending = []


def install(model, arm, seed):
    """Call after joint.install; isolated MLP RNG preserves original streams."""
    before = {k: v.clone() for k, v in model.state_dict().items()}
    with torch.random.fork_rng(devices=[]):
        torch.set_rng_state(torch.Generator().manual_seed(seed+41000).get_state())
        scores = RelationScores(arm)
    del model.conv_a
    model.relation_scores = scores
    for key, value in before.items():
        if not key.startswith('conv_a.'):
            assert torch.equal(model.state_dict()[key], value), key
    model._relation_pending = []
    model._relation_forward_func = model.forward.__func__
    model.forward = MethodType(forward, model)
    model._forward_semantic_tokens = MethodType(semantic_tokens, model)

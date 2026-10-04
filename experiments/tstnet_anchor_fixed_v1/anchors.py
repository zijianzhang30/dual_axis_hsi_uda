"""Track native flip geometry without changing patches, labels or RNG draws."""
import ast
from pathlib import Path
from types import MethodType

import numpy as np
import torch

NATIVE = Path('/nas1/zhangzj26/dual_axis_hsi_uda/results/tstnet_hyrank_native_v1/run_snapshot/datasets.py')


def tracked_loader_classes():
    tree = ast.parse(NATIVE.read_text())
    dataset = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'HyperX')
    dataset.name = 'TrackedHyperX'
    for node in ast.walk(dataset):
        if isinstance(node, ast.Name) and node.id == 'HyperX':
            node.id = 'TrackedHyperX'
    getitem = next(n for n in dataset.body if isinstance(n, ast.FunctionDef) and n.name == '__getitem__')
    label_at = next(i for i, n in enumerate(getitem.body) if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name) and n.targets[0].id == 'label')
    marker = ast.parse('marker = np.zeros((self.patch_size, self.patch_size), dtype=np.uint8)\nmarker[self.patch_size//2, self.patch_size//2] = 1').body
    getitem.body[label_at + 1:label_at + 1] = marker
    flips = [n for n in ast.walk(getitem) if isinstance(n, ast.Assign) and isinstance(n.value, ast.Call) and isinstance(n.value.func, ast.Attribute) and n.value.func.attr == 'flip']
    assert len(flips) == 1
    flips[0].targets[0].elts.append(ast.Name(id='marker', ctx=ast.Store()))
    flips[0].value.args.append(ast.Name(id='marker', ctx=ast.Load()))
    returns = [n for n in ast.walk(getitem) if isinstance(n, ast.Return)]
    assert len(returns) == 1
    returns[0].value.elts.append(ast.parse('torch.from_numpy(np.argwhere(marker)[0].astype(np.int64))', mode='eval').body)
    prefetch = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'data_prefetcher')
    prefetch.name = 'TrackedPrefetcher'
    preload = next(n for n in prefetch.body if isinstance(n, ast.FunctionDef) and n.name == 'preload')
    fetches = [n for n in ast.walk(preload) if isinstance(n, ast.Assign) and isinstance(n.value, ast.Call) and isinstance(n.value.func, ast.Name) and n.value.func.id == 'next']
    assert len(fetches) == 1
    fetches[0].targets[0].elts.append(ast.Attribute(value=ast.Name(id='self', ctx=ast.Load()), attr='anchor', ctx=ast.Store()))
    transfer = next(n for n in preload.body if isinstance(n, ast.With))
    transfer.body += ast.parse('self.anchor = self.anchor.cuda(non_blocking=True)').body
    nextfn = next(n for n in prefetch.body if isinstance(n, ast.FunctionDef) and n.name == 'next')
    preload_at = next(i for i, n in enumerate(nextfn.body) if isinstance(n, ast.Expr) and isinstance(n.value, ast.Call) and isinstance(n.value.func, ast.Attribute) and n.value.func.attr == 'preload')
    nextfn.body[preload_at:preload_at] = ast.parse('anchor = self.anchor').body
    next(n for n in nextfn.body if isinstance(n, ast.Return)).value.elts.append(ast.Name(id='anchor', ctx=ast.Load()))
    module = ast.Module(body=[dataset, prefetch], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = dict(np=np, torch=torch)
    exec(compile(module, str(NATIVE), 'exec'), namespace)
    return namespace['TrackedHyperX'], namespace['TrackedPrefetcher']


def set_context(model, source_anchor, target_anchor):
    model._anchor_context = (source_anchor, target_anchor)


def relations(self, patch, anchors=None):
    if patch.ndim != 5 or patch.shape[1:3] != (1, 176):
        raise ValueError(f'Expected Bx1x176xHxW, got {patch.shape}')
    x = patch[:, 0]
    if anchors is None:
        if not self._anchor_allow_default:
            raise RuntimeError('Missing actual augmented anchor coordinates')
        anchors = torch.tensor([x.shape[-2]//2, x.shape[-1]//2], device=x.device).expand(len(x), 2)
    anchors = anchors.to(device=x.device, dtype=torch.long)
    if anchors.shape != (len(x), 2):
        raise ValueError('Anchor batch shape mismatch')
    if torch.any(anchors < 0) or torch.any(anchors[:, 0] >= x.shape[-2]) or torch.any(anchors[:, 1] >= x.shape[-1]):
        raise ValueError('Anchor outside patch')
    anchor = x.permute(0, 2, 3, 1)[torch.arange(len(x), device=x.device), anchors[:, 0], anchors[:, 1]][:, :, None, None]
    return (x - anchor) / (x.std(dim=(-2, -1), correction=0, keepdim=True) + 1e-6)


def scores(self, patch, anchors=None):
    relation = self.relations(patch, anchors).permute(0, 2, 3, 1)
    logits = self.mlp(relation).permute(0, 3, 1, 2)
    return logits.flatten(2).softmax(dim=-1)


def semantic_tokens(self, features):
    if not self._relation_pending or not self._anchor_pending:
        raise RuntimeError('Relation tokenizer metadata not supplied')
    patch = self._relation_pending.pop(0)
    anchors = self._anchor_pending.pop(0)
    if features.shape[0] != patch.shape[0] or features.shape[-2:] != patch.shape[-2:]:
        raise RuntimeError('Relation and CNN value grids differ')
    weights = self.relation_scores(patch, anchors)
    return torch.einsum('bln,bcn->blc', weights, features.flatten(2))


def forward(self, source, target, *args, **kwargs):
    if self._anchor_pending:
        raise RuntimeError('Nested anchor-aware forward')
    context = self._anchor_context
    if context is None:
        if not self._anchor_allow_default:
            raise RuntimeError('Missing loader anchor context')
        context = (None, None)  # Synthetic, unaugmented preflight only.
    self._anchor_pending = list(context)
    try:
        result = self._anchor_original_forward(self, source, target, *args, **kwargs)
        if self._anchor_pending:
            raise RuntimeError('Both domain anchors must be consumed')
        return result
    finally:
        self._anchor_pending = []
        self._anchor_context = None


def install_model(model, allow_default=False):
    before = {k: v.clone() for k, v in model.state_dict().items()}
    rng_cpu = torch.get_rng_state().clone()
    model._anchor_context = None
    model._anchor_pending = []
    model._anchor_allow_default = allow_default
    model.relation_scores._anchor_allow_default = allow_default
    model.relation_scores.relations = MethodType(relations, model.relation_scores)
    model.relation_scores.forward = MethodType(scores, model.relation_scores)
    model._forward_semantic_tokens = MethodType(semantic_tokens, model)
    model._anchor_original_forward = model.forward.__func__
    model.forward = MethodType(forward, model)
    assert torch.equal(rng_cpu, torch.get_rng_state())
    assert all(torch.equal(v, model.state_dict()[k]) for k, v in before.items())

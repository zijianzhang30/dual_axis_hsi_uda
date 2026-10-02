"""BiDA BN interventions, preserving the original downstream forward."""

import copy
from types import MethodType

import torch
from torch import nn


class DomainBN(nn.Module):
    """Independent native running buffers, exactly shared affine parameters."""

    def __init__(self, original):
        super().__init__()
        self.source = original
        self.target = copy.deepcopy(original)
        self.target.weight = self.source.weight
        self.target.bias = self.source.bias
        self.domain = "target"

    def forward(self, features):
        return getattr(self, self.domain)(features)


def set_domain(model, domain):
    if domain not in {"source", "target"}:
        raise ValueError(domain)
    model._eval_domain = domain
    for module in model.modules():
        if isinstance(module, DomainBN):
            module.domain = domain


def canonical_state(model):
    """State keys matching the original BiDA, using source buffers for DSBN."""
    state = {}
    for key, value in model.state_dict().items():
        if ".target." in key:
            continue
        state[key.replace(".source.", ".")] = value
    return state


def paired_tokens(model, source, target):
    if source.shape != target.shape:
        raise ValueError("Equal-shaped source/target batches required for 0.5 mixture")
    size = len(source)
    conv3d, bn3d, activation3d = model.conv3d_features
    cube = activation3d(bn3d(torch.cat((conv3d(source), conv3d(target)), dim=0)))
    batch, channels, bands, height, width = cube.shape
    cube = cube.reshape(batch, channels * bands, height, width)
    conv2d, bn2d, activation2d = model.conv2d_features
    source_cube, target_cube = cube.split(size, dim=0)
    features = activation2d(bn2d(torch.cat((conv2d(source_cube), conv2d(target_cube)), dim=0)))
    source_features, target_features = features.split(size, dim=0)
    return (model._forward_semantic_tokens(source_features),
            model._forward_semantic_tokens(target_features))


def tokenize_routed(self, images):
    if self._bn_arm == "fixed_mixture" and self.training:
        if not self._paired_cache:
            raise RuntimeError("Paired normalization must be invoked through model.forward")
        return self._paired_cache.pop(0)
    if self._bn_arm == "dsbn":
        domain = ("source" if self._tokenize_calls == 0 else "target") if self.training else self._eval_domain
        self._tokenize_calls += 1
        for module in self.modules():
            if isinstance(module, DomainBN):
                module.domain = domain
    return self._original_tokenize_func(self, images)


def forward_routed(self, source, target, *args, **kwargs):
    self._tokenize_calls = 0
    if self._bn_arm == "fixed_mixture" and self.training:
        self._paired_cache = list(paired_tokens(self, source, target))
    try:
        result = self._original_forward_func(self, source, target, *args, **kwargs)
        if self._bn_arm == "fixed_mixture" and self.training and self._paired_cache:
            raise RuntimeError("Original forward did not consume both domain tokens")
        return result
    finally:
        self._paired_cache = []


def install(model, arm):
    if arm not in {"dsbn", "fixed_mixture"}:
        raise ValueError(arm)
    before = {key: value.clone() for key, value in model.state_dict().items()}
    parameter_count = sum(value.numel() for value in model.parameters())
    if arm == "dsbn":
        model.conv3d_features[1] = DomainBN(model.conv3d_features[1])
        model.conv2d_features[1] = DomainBN(model.conv2d_features[1])
    else:
        model.conv3d_features[1].momentum = 0.19
        model.conv2d_features[1].momentum = 0.19
    after = canonical_state(model)
    if set(after) != set(before) or any(not torch.equal(value, after[key]) for key, value in before.items()):
        raise RuntimeError("Initial shared tensors changed")
    if sum(value.numel() for value in model.parameters()) != parameter_count:
        raise RuntimeError("Trainable parameter count changed")
    model._bn_arm = arm
    model._eval_domain = "target"
    model._tokenize_calls = 0
    model._paired_cache = []
    model._original_forward_func = model.forward.__func__
    model._original_tokenize_func = model._tokenize.__func__
    model.forward = MethodType(forward_routed, model)
    model._tokenize = MethodType(tokenize_routed, model)
    return len(before)

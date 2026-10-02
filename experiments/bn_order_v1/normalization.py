"""Reverse only stem BN update order; preserve attention/dropout order."""

from types import MethodType

import torch


def canonical_state(model):
    return model.state_dict()


def set_domain(model, domain):
    pass


def tokenize_cached(self, images):
    if self.training:
        if not self._reverse_cache:
            raise RuntimeError("Call paired forward to populate token cache")
        return self._reverse_cache.pop(0)
    return self._original_tokenize_func(self, images)


def forward_reverse(self, source, target, *args, **kwargs):
    if self.training:
        target_tokens = self._original_tokenize_func(self, target)
        source_tokens = self._original_tokenize_func(self, source)
        self._reverse_cache = [source_tokens, target_tokens]
    try:
        result = self._original_forward_func(self, source, target, *args, **kwargs)
        if self.training and self._reverse_cache:
            raise RuntimeError("Tokens not fully consumed")
        return result
    finally:
        self._reverse_cache = []


def install(model, arm):
    if arm != "reverse":
        raise ValueError(arm)
    before = {key: value.clone() for key, value in model.state_dict().items()}
    model._original_tokenize_func = model._tokenize.__func__
    model._original_forward_func = model.forward.__func__
    model._reverse_cache = []
    model._tokenize = MethodType(tokenize_cached, model)
    model.forward = MethodType(forward_reverse, model)
    assert all(torch.equal(value, model.state_dict()[key]) for key, value in before.items())
    return len(before)

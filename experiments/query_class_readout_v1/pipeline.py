"""Image-dependent A1 class-query readout without any BiDA refinement block."""
from types import MethodType
import torch
from torch import nn

def classify_tokens(self, tokens):
    tokens = self.dropout(tokens)
    cls, _ = self.class_reader(self.class_query[None].expand(len(tokens), -1, -1), tokens)
    z = self.norm(cls[:, 0])
    return self.nn1(z), z

def forward_query(self, source, target, inference_target_only=False, return_feat_prob=False):
    if self.training:
        source_tokens, target_tokens = self._tokenize(source), self._tokenize(target)
        source_logits, _ = self._classify_tokens(source_tokens)
        target_logits, _ = self._classify_tokens(target_tokens)
        return source_logits, target_logits, None, None
    target_tokens = self._tokenize(target)
    logits, z = self._classify_tokens(target_tokens)
    return (None, logits, None, z) if return_feat_prob else (None, logits, None)

def install_readout(model, a1):
    before = {key: value.clone() for key, value in model.state_dict().items()}
    model.blocks = nn.ModuleList()
    # Remove the now-unused BiDA-only parameters; no constant CLS is classified.
    for name in ('cls_token', 'pos_embedding', 'token_wA', 'token_wV'):
        delattr(model, name)
    model.class_query = nn.Parameter(torch.randn(1, 64) * .02)
    model.class_reader = a1.CrossBlock(64, heads=4)
    model._classify_tokens = MethodType(classify_tokens, model)
    model.forward = MethodType(forward_query, model)
    for key, value in model.state_dict().items():
        if not key.startswith(('class_query', 'class_reader.')):
            assert torch.equal(value, before[key]), key
    assert len(model.blocks) == 0
    return sum(p.numel() for p in model.parameters())

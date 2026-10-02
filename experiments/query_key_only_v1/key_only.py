"""Parameter-free split of spatial key and value inputs; class reader untouched."""
from types import MethodType
import torch

def key_only_forward(self, query, context):
    key_context, value_context = context
    delta, weights = self.attn(self.q_norm(query), self.kv_norm(key_context),
                               self.kv_norm(value_context), need_weights=True)
    result = query + delta
    return result + self.ff(self.ff_norm(result)), weights

def install_key_only(model, a1):
    before = {k: v.clone() for k, v in model.state_dict().items()}
    class_forward = model.class_reader.forward.__func__
    model.spatial_reader.forward = MethodType(key_only_forward, model.spatial_reader)
    def semantic_tokens(self, features):
        batch, channels, height, width = features.shape
        pixels = features.flatten(2).transpose(1, 2)
        half = channels // 2
        row = a1.sinusoidal_positions(height, half, features.device, features.dtype)
        col = a1.sinusoidal_positions(width, channels-half, features.device, features.dtype)
        position = torch.cat((row[:, None].expand(-1, width, -1),
                              col[None].expand(height, -1, -1)), dim=-1)
        key_context = pixels + position.reshape(1, height*width, channels)
        tokens, _ = self.spatial_reader(self.spatial_queries[None].expand(batch, -1, -1),
                                         (key_context, pixels))
        return tokens
    model._forward_semantic_tokens = MethodType(semantic_tokens, model)
    assert model.class_reader.forward.__func__ is class_forward
    after = model.state_dict()
    assert set(after) == set(before)
    assert all(torch.equal(v, after[k]) for k,v in before.items())

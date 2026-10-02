"""Hooks on actual self-inference residual states; no rewritten forward."""
from types import MethodType
import torch

class LayerCapture:
    def __init__(self, model):
        self.model, self.values, self.handles = model, {}, []
        self.semantic = model._forward_semantic_tokens
        def semantic(this, features):
            tokens = self.semantic(features)
            self.values['tokenizer_all'] = tokens.flatten(0, 1).detach()
            self.values['tokenizer_mean'] = tokens.mean(1).detach()
            return tokens
        model._forward_semantic_tokens = MethodType(semantic, model)
        for index, block in enumerate(model.blocks, 1):
            prefix = f'block{index}'
            def before(module, inputs, key=prefix):
                self.values[key + '_input_cls'] = inputs[1][:, 0].detach()
            def attention(module, inputs, outputs, key=prefix):
                self.values[key + '_attention_update'] = outputs[1][:, 0].detach()
            def residual(module, inputs, key=prefix):
                self.values[key + '_after_attention'] = inputs[0][:, 0].detach()
            def mlp(module, inputs, output, key=prefix):
                self.values[key + '_mlp_update'] = output[:, 0].detach()
            def after(module, inputs, outputs, key=prefix):
                self.values[key + '_after_mlp'] = outputs[1][:, 0].detach()
            self.handles.extend([block.register_forward_pre_hook(before),
                                 block.attn.register_forward_hook(attention),
                                 block.norm2.register_forward_pre_hook(residual),
                                 block.mlp.register_forward_hook(mlp),
                                 block.register_forward_hook(after)])
        def head(module, inputs, outputs):
            self.values['final_cls'] = inputs[0].detach()
            self.values['final_logits'] = outputs.detach()
        self.handles.append(model.nn1.register_forward_hook(head))
    def close(self):
        self.model._forward_semantic_tokens = self.semantic
        for handle in self.handles:
            handle.remove()

def report(values, model, final=False):
    vectors = torch.cat(values).double()
    norm = vectors.norm(dim=-1)
    unit = torch.nn.functional.normalize(vectors, dim=-1)
    weight = model.nn1.weight.detach().cpu().double()
    cosine = unit @ torch.nn.functional.normalize(weight, dim=-1).T
    with torch.no_grad():
        normalized = vectors.float().to(model.nn1.weight.device)
        if not final:
            normalized = model.norm(normalized)
        logits = model.nn1(normalized).cpu().double()
        margin = logits[:, 5] - torch.cat((logits[:, :5], logits[:, 6:]), 1).max(1).values
    return {'mean_norm': float(norm.mean()), 'direction_concentration': float(unit.mean(0).norm()),
            'cos_w6_mean': float(cosine[:, 5].mean()),
            'aux_final_norm_head_margin6': float(margin.mean()),
            'aux_final_norm_head_class6_share': float((logits.argmax(1) == 5).double().mean())}

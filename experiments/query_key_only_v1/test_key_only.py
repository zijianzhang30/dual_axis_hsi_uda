"""CPU numerical wiring and parameter/RNG invariance checks."""
import copy
import importlib.util
from pathlib import Path
import torch
from torch import nn
from key_only import install_key_only, key_only_forward

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('a1_test', ROOT/'model.py')
a1 = importlib.util.module_from_spec(spec); spec.loader.exec_module(a1)

def main():
    torch.set_num_threads(2); torch.manual_seed(17)
    block = a1.CrossBlock(64,4)
    q = torch.randn(2,4,64); x = torch.randn(2,9,64)
    original = block(q,x)
    split = key_only_forward(block,q,(x,x))
    assert torch.equal(original[0],split[0]) and torch.equal(original[1],split[1])
    model = nn.Module(); model.spatial_reader = copy.deepcopy(block)
    model.class_reader = a1.CrossBlock(64,4)
    model.spatial_queries = nn.Parameter(torch.randn(4,64)*.02)
    state = {k:v.clone() for k,v in model.state_dict().items()}
    rng = torch.get_rng_state().clone(); class_forward = model.class_reader.forward.__func__
    install_key_only(model,a1)
    assert torch.equal(rng,torch.get_rng_state())
    assert all(torch.equal(v,model.state_dict()[k]) for k,v in state.items())
    assert model.class_reader.forward.__func__ is class_forward
    features = torch.randn(2,64,3,3,requires_grad=True)
    capture = []
    handle = model.spatial_reader.attn.register_forward_pre_hook(lambda module,inputs:capture.append(inputs))
    tokens = model._forward_semantic_tokens(features); handle.remove()
    pixels = features.flatten(2).transpose(1,2)
    row = a1.sinusoidal_positions(3,32,features.device,features.dtype)
    pe = torch.cat((row[:,None].expand(-1,3,-1),row[None].expand(3,-1,-1)),dim=-1).reshape(1,9,64)
    assert torch.equal(capture[0][1],model.spatial_reader.kv_norm(pixels+pe))
    assert torch.equal(capture[0][2],model.spatial_reader.kv_norm(pixels))
    assert not torch.equal(capture[0][1],capture[0][2])
    tokens.square().sum().backward()
    assert features.grad is not None and torch.isfinite(features.grad).all() and features.grad.norm()>0
    assert model.spatial_queries.grad is not None and torch.isfinite(model.spatial_queries.grad).all()
    print('PASS: zero-PE equivalence; identical state/RNG; class reader unchanged; K/V wiring; finite input/query gradients')

if __name__ == '__main__': main()

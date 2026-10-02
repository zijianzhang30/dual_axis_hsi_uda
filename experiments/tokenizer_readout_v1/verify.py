"""Check shared initialization, RNG neutrality and native joint gradients."""
import copy
from types import SimpleNamespace
import torch
import train

def main():
    torch.set_num_threads(2);torch.manual_seed(19)
    base=train.get_model('BiDA','Houston13',13,SimpleNamespace(num_tokens=4,dim=64,depth=3))
    conv=copy.deepcopy(base.conv_a);semantic=base._forward_semantic_tokens.__func__
    train.transplant(base);train.install_readout(base,train.a1)
    xs=torch.randn(3,1,48,5,5);xt=torch.randn_like(xs)*2+3
    for arm in ('query_class','bida_class','query_mean','bida_mean'):
        model=copy.deepcopy(base)
        rng=torch.get_rng_state().clone()
        train.configure_cell(model,arm,copy.deepcopy(conv),semantic)
        assert torch.equal(rng,torch.get_rng_state())
        for key,value in model.state_dict().items():
            reference=conv.state_dict()[key[len('conv_a.'):]] if key.startswith('conv_a.') else base.state_dict()[key]
            assert torch.equal(value,reference),key
        train.joint.install(model,'fixed_mixture');model.eval()
        with torch.no_grad():
            logits=model(xs,xs)[1]
            tokens=model._tokenize(xs)
            if arm.endswith('_mean'):z=tokens.mean(1)
            else:z=model.class_reader(model.class_query[None].expand(3,-1,-1),tokens)[0][:,0]
            assert torch.equal(logits,model.nn1(model.norm(z)))
            assert not torch.equal(logits[0],logits[1])
        model.train();target=xt.clone().requires_grad_(True)
        model(xs,target)[0].square().sum().backward()
        assert target.grad is not None and target.grad.abs().sum()>0
        assert int(model.conv2d_features[1].num_batches_tracked)==1
        if arm.startswith('bida_'):assert not hasattr(model,'spatial_reader')
        if arm.endswith('_mean'):assert not hasattr(model,'class_reader')
    print('PASS: four cells preserve retained initialization/RNG; exact native readouts; image dependence; differentiable joint target moments; no unused reader modules')

if __name__=='__main__':main()

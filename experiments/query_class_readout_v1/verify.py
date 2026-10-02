"""Verify image-dependent readout and unchanged differentiable joint BN."""
from types import SimpleNamespace
import torch
import train

def main():
    torch.set_num_threads(2)
    torch.manual_seed(19)
    model = train.get_model('BiDA','Houston13',13,SimpleNamespace(num_tokens=4,dim=64,depth=3))
    train.transplant(model)
    train.install_readout(model,train.a1)
    train.joint.install(model,'fixed_mixture')
    assert not any(name.startswith('blocks.') for name in model.state_dict())
    xs = torch.randn(3,1,48,5,5)
    xt = torch.randn_like(xs)*2+3
    model.eval()
    with torch.no_grad():
        logits = model(xs,xs)[1]
        tokens = model._tokenize(xs)
        cls,_ = model.class_reader(model.class_query[None].expand(3,-1,-1),tokens)
        reference = model.nn1(model.norm(cls[:,0]))
        assert torch.equal(logits,reference)
        assert not torch.equal(logits[0],logits[1])
        assert not torch.equal(logits,model(xt,xt)[1])
    model.train()
    xt.requires_grad_(True)
    model(xs,xt)[0].square().sum().backward()
    assert xt.grad is not None and xt.grad.abs().sum()>0
    for parameter in (model.class_query, model.spatial_queries, model.class_reader.attn.in_proj_weight, model.spatial_reader.attn.in_proj_weight):
        assert parameter.grad is not None and parameter.grad.abs().sum()>0
    assert int(model.conv3d_features[1].num_batches_tracked)==1
    assert int(model.conv2d_features[1].num_batches_tracked)==1
    restored = train.get_model('BiDA','Houston13',13,SimpleNamespace(num_tokens=4,dim=64,depth=3))
    train.transplant(restored)
    train.install_readout(restored,train.a1)
    train.joint.install(restored,'fixed_mixture')
    restored.load_state_dict(model.state_dict(),strict=True)
    restored.eval()
    model.eval()
    with torch.no_grad():
        assert torch.equal(restored(xs,xs)[1],model(xs,xs)[1])
    print('PASS: no BiDA blocks; image-dependent A1 class-query readout; exact manual eval; source gradients reach both readers and target moments; one BN update; checkpoint reload exact')

if __name__=='__main__':
    main()

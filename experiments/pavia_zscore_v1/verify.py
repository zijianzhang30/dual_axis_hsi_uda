"""Confirm official normalization equivalence, frozen inputs and gradients."""
import json
import numpy as np
import torch
import torch.nn.functional as F
import common as c


def main():
    c.OUT.mkdir(parents=True,exist_ok=True)
    if (c.OUT/'implementation_checks.json').exists():raise RuntimeError('Do not overwrite preflight')
    torch.set_num_threads(2);device=torch.device('cuda:7');torch.cuda.set_device(device)
    normalization=[]
    for name in ('paviaU','pavia'):
        x,gt=c.load_scene(name);actual,original_gt=c.official_scene(name)
        error=float(np.max(np.abs(x-actual)))
        assert error==0 and np.array_equal(gt,original_gt.astype(int)-1)
        normalization.append({'scene':name,'shape':list(x.shape),'MLUDA_function_max_abs_error':error,
            'band_mean_max_abs':float(np.abs(x.mean((0,1))).max()),
            'band_std_max_deviation':float(np.abs(x.std((0,1))-1).max())})
        del x,actual,gt,original_gt
    c.pavia.seed_worker(2100)
    values=c.loaders(workers=0)
    audit=c.audit(values)
    c.pavia.seed_worker(2100)
    old=c.base.loaders('pavia',2100,0)
    for a,b in zip(values,old):assert np.array_equal(a.dataset.indices,b.dataset.indices)
    del old
    x,y=next(iter(values[0]));t,labels=next(iter(values[2]));assert (labels==0).all()
    x,y,t=x[:8].to(device),y[:8].to(device),t[:8].to(device)
    assert x.shape==(8,1,102,13,13) and t.shape==x.shape
    reports=[]
    for arm in ('B','C'):
        c.pavia.seed_worker(2100)
        model,initial=c.base.build('pavia',arm,2100)
        old_cfg=json.loads((c.reference(arm)/'config.json').read_text())
        assert initial==old_cfg['initial_model_sha256']
        assert audit['data_sha256']==old_cfg['data_sha256'] and audit['sample_counts']==old_cfg['sample_counts']
        model.to(device).train()
        target=t.detach().clone().requires_grad_(True)
        loss=F.cross_entropy(model(x,target)[0],y)
        assert torch.isfinite(loss);loss.backward()
        assert target.grad is not None and torch.isfinite(target.grad).all() and target.grad.abs().sum()>0
        for name,p in model.named_parameters():
            if p.grad is not None:assert torch.isfinite(p.grad).all(),name
        if arm=='C':
            rs=model.relation_scores
            expected=(x[:,0]-x[:,0,:,6:7,6:7])/(x[:,0].std((-2,-1),correction=0,keepdim=True)+1e-6)
            assert torch.equal(rs.relations(x),expected)
            weights=rs(x);assert weights.shape==(8,4,169)
            assert torch.isfinite(weights).all() and torch.allclose(weights.sum(-1),torch.ones(8,4,device=device),atol=1e-6)
            assert all(p.grad is not None and p.grad.abs().sum()>0 for p in rs.parameters())
        model.eval()
        with torch.no_grad():logits=model(x,t)[1]
        assert logits.shape==(8,7) and torch.isfinite(logits).all()
        reports.append({'arm':arm,'initial_sha256':initial,'parameters':sum(p.numel() for p in model.parameters()),
                        'target_input_gradient_l1':target.grad.abs().sum().item(),'finite_loss':loss.item(),'passed':True})
        del model,target,loss,logits
    result={'all_checks_passed':True,'normalization':normalization,'data_audit':audit,
            'full_indices_match_normband':True,'model_receives_only_102_zscore_bands':True,'runs':reports}
    c.dump(c.OUT/'implementation_checks.json',result)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()

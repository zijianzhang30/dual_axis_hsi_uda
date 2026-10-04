"""Read-only historical reuse audit and pre-training implementation gate."""
import gc
import json
from types import SimpleNamespace
import torch
import torch.nn.functional as F
import common as c


def main():
    c.OUT.mkdir(parents=True,exist_ok=True)
    if (c.OUT/'implementation_checks.json').exists(): raise RuntimeError('Do not overwrite checks')
    torch.set_num_threads(2)
    device=torch.device('cuda:7');torch.cuda.set_device(device)
    reports=[];reuse=[]
    for task in ('pavia','shanghai'):
        c.pavia.seed_worker(2100)
        values=c.loaders(task,2100,0)
        audit=c.inputs_audit(task,values)
        if task=='pavia':
            # Compare native historical construction, including full sample ordering.
            c.pavia.seed_worker(2100)
            original=c.pavia.make_loaders(SimpleNamespace(dataset_dir=c.DATA/'Pavia',seed=2100,num_workers=0))
            for i in range(4):
                assert torch.equal(torch.from_numpy(values[i].dataset.indices),torch.from_numpy(original[i].dataset.indices))
            del original
            for seed in (2100,2101,2102):
                c.pavia.seed_worker(seed)
                # Loader construction consumes native Python RNG, not torch RNG.
                base,initial=c.build(task,'B',seed)
                history=c.historical(task,seed,initial,values)
                assert len(history)==200 and history[-1]['epoch']==200
                assert (c.ROOT/'results/pavia_joint_v1'/('joint_%d'%seed)/'results.json').exists()
                reuse.append({'seed':seed,'initial':initial,'data_indices_and_frozen_code_match':True})
                del base
        x,y=next(iter(values[0]));t,dummy=next(iter(values[2]))
        assert (dummy==0).all()
        x,y,t=x[:8].to(device),y[:8].to(device),t[:8].to(device)
        bands,classes=c.dimensions(task)
        for arm in ('B','C'):
            c.pavia.seed_worker(2100)
            base,_=c.build(task,'B',2100)
            before={k:v.clone() for k,v in base.state_dict().items()}
            cpu=torch.get_rng_state().clone();cuda=torch.cuda.get_rng_state(device).clone()
            if arm=='C':
                c.relation(bands)[0].install(base,'C',2100)
                assert torch.equal(cpu,torch.get_rng_state()) and torch.equal(cuda,torch.cuda.get_rng_state(device))
                assert all(torch.equal(v,base.state_dict()[k]) for k,v in before.items() if not k.startswith('conv_a.'))
                base.to(device)
                rs=base.relation_scores
                expected=(x[:,0]-x[:,0,:,6:7,6:7])/(x[:,0].std((-2,-1),correction=0,keepdim=True)+1e-6)
                assert torch.equal(rs.relations(x),expected)
                constant=torch.full_like(x,.1)
                assert torch.isfinite(rs(constant)).all()
                weights=rs(x)
                assert weights.shape==(8,4,169) and torch.isfinite(weights).all()
                assert torch.allclose(weights.sum(-1),torch.ones(8,4,device=device),atol=1e-6)
            base.to(device).train()
            target=t.detach().clone().requires_grad_(True)
            # The original preflight CPU relation module is moved with the model.
            loss=F.cross_entropy(base(x,target)[0],y)
            assert torch.isfinite(loss)
            loss.backward()
            assert target.grad is not None and torch.isfinite(target.grad).all() and target.grad.abs().sum()>0
            for name,param in base.named_parameters():
                if param.grad is not None: assert torch.isfinite(param.grad).all(),name
            if arm=='C':
                assert all(p.grad is not None and p.grad.abs().sum()>0 for p in base.relation_scores.parameters())
            base.eval()
            with torch.no_grad(): logits=base(x,t)[1]
            assert logits.shape==(8,classes) and torch.isfinite(logits).all()
            reports.append({'task':task,'arm':arm,'bands':bands,'classes':classes,
                'parameters':sum(p.numel() for p in base.parameters()),'loss':loss.item(),
                'target_input_gradient_l1':target.grad.abs().sum().item(),'data':audit,'passed':True})
            del base,before,target,loss,logits
        del values,x,y,t
        gc.collect();torch.cuda.empty_cache()
    result={'all_checks_passed':True,'pavia_B_reuse':reuse,'runs':reports}
    c.dump(c.OUT/'implementation_checks.json',result)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()

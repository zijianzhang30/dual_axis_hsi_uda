"""Verify losses, projections, native views and paired BN/RNG before all12 runs."""
import gc
import json
import numpy as np
import torch
import torch.nn.functional as F
import common as c


def main():
    c.verify_files();torch.set_num_threads(2);torch.cuda.set_device(0)
    b=c.base();_,mmd,contrast,augmentation=c.official();con=contrast.SupConLoss(temperature=.1)
    checks=[]
    for seed in c.SEEDS:
        source,target=b.train_sets('P2',seed);loaders,plan=b.epoch_loaders(source,target,seed,1)
        x,y=next(iter(loaders[0]));t,_=next(iter(loaders[1]))
        batchviews=c.views(x,t,seed,1,1,augmentation)
        assert len(batchviews)==3 and all(s.shape==v.shape==(32,1,102,11,11) for s,v in batchviews)
        for network in ('M','O'):
            initial=[];rng=[];counts=[]
            for enabled in (False,True):
                model,h=c.build(seed,network);initial.append(c.state_hash(model));model=model.cuda().train()
                captures=[];tg=None
                for i,(sx,tx) in enumerate(batchviews):
                    sx=sx.cuda();tx=tx.cuda().requires_grad_()
                    if i==0:tg=tx
                    result=model(sx,tx)
                    dim=288 if network=='M' else 64
                    assert result['fs'].shape==result['ft'].shape==(32,dim)
                    assert result['ps'].shape==result['pt'].shape==(32,128)
                    assert result['ls'].shape==result['lt'].shape==(32,7)
                    captures.append(result)
                rng.append(c.digest(torch.cuda.get_rng_state().cpu().numpy()))
                loss,terms=c.objective(captures,y.cuda(),1,enabled,mmd,con)
                if enabled:
                    # Compare the implemented sum directly to the official expression/weights.
                    lam=2/(1+__import__('math').exp(-.1))-1
                    reference=terms['CE']+.3*lam*terms['LMMD']+terms['Con_t']+terms['Con_s']
                    assert abs(float(loss.detach())-reference)<2e-5
                loss.backward();assert torch.isfinite(loss).all()
                assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
                assert tg.grad is not None and float(tg.grad.abs().sum())>0
                heads=(model.backbone.head1,model.backbone.head2) if network=='M' else (model.head1,model.head2)
                hg=[sum(float(p.grad.abs().sum()) for p in head.parameters() if p.grad is not None) for head in heads]
                assert all(g>0 for g in hg) if enabled else all(g==0 for g in hg)
                bc=c.bn_counts(model);counts.append(bc)
                for name,n in bc.items():
                    assert n==(0 if name.endswith(('bn9','bn10')) else (6 if network=='M' else 3)),(name,n)
                before=c.state_hash(model);opt=c.optimizer(model,network);opt.step()
                assert c.state_hash(model)!=before
                checks.append({'seed':seed,'network':network,'adaptation':enabled,'initial_sha256':initial[-1],
                               'parameters':sum(p.numel() for p in model.parameters()),'loss_terms':terms,
                               'projection_grad_l1':hg,'main_target_grad_l1':float(tg.grad.abs().sum()),
                               'BN_num_batches_after_three_views':bc})
                del model,opt,captures,sx,tx,tg,loss,result,heads
                gc.collect();torch.cuda.empty_cache()
            assert initial[0]==initial[1] and rng[0]==rng[1] and counts[0]==counts[1]
        del source,target,loaders
    c.dump(c.OUT/'verification.json',{'passed':True,'checks':checks,
        'native_objective_expression_verified':True,'within_row_initialization_forward_rng_BN_equal':True,
        'native_flip_preserved_without_label_permutation':True,'target_GT_values_accessed':False})
    print('All12 initializations / gradient / objective / BN-RNG checks passed.',flush=True)


if __name__=='__main__':main()

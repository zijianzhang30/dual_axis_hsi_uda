"""Check exact CE/Full identity and each real removal gradient path before training."""
import gc
import json
import numpy as np
import torch
import common as c

def main():
    c.forbid_target_label_loading();c.verify_files();c.audit_reuse()
    torch.set_num_threads(2);torch.cuda.set_device(0)
    b=c.base();_,mmd,contrast,aug=c.official();con=contrast.SupConLoss(temperature=.1)
    checks=[]
    for seed in c.SEEDS:
        source,target=b.train_sets('P2',seed);loaders,plan=b.epoch_loaders(source,target,seed,1)
        x,y=next(iter(loaders[0]));t,_=next(iter(loaders[1]));vv=c.views(x,t,seed,1,1,aug)
        starts=[];randoms=[];bn=[];logit_hash=None
        for arm in c.ARMS:
            model,initial=c.build(seed,'O');start=c.state_hash(model);starts.append(start)
            old=json.loads((c.run_folder('Full',seed)/'config.json').read_text())
            assert initial==old['initial_base_model_sha256'] and start==old['initial_complete_model_sha256']
            assert sum(p.numel() for p in model.parameters())==old['parameters']==645211
            model=model.cuda().train();records=[]
            for i,(sx,tx) in enumerate(vv):
                tx=tx.cuda().requires_grad_();r=model(sx.cuda(),tx)
                if i==0:tg=tx
                assert r['fs'].shape==r['ft'].shape==(32,64) and r['ps'].shape==r['pt'].shape==(32,128)
                records.append(r)
            logits=c.digest(records[0]['ls'].detach().cpu().numpy())
            if logit_hash is None:logit_hash=logits
            assert logits==logit_hash
            randoms.append(c.digest(torch.cuda.get_rng_state().cpu().numpy()))
            count=c.bn_counts(model);bn.append(count);assert all(n==3 for n in count.values())
            loss,terms=c.objective(records,y.cuda(),1,arm,mmd,con)
            if arm in ('CE','Full'):
                oldloss,_=c.legacy.objective(records,y.cuda(),1,arm=='Full',mmd,con)
                assert torch.equal(loss,oldloss)
                params=list(model.parameters())
                ga=torch.autograd.grad(loss,params,allow_unused=True,retain_graph=True)
                gb=torch.autograd.grad(oldloss,params,allow_unused=True,retain_graph=True)
                assert all((a is None and b is None) or (a is not None and b is not None and torch.equal(a,b)) for a,b in zip(ga,gb))
                del oldloss,ga,gb,params
            else:
                on=c.flags(arm);lam=terms['lambda']
                reference=terms['CE']
                if on['LMMD']:reference+=.3*lam*terms['LMMD']
                if on['Con_t']:reference+=terms['Con_t']
                if on['Con_s']:reference+=terms['Con_s']
                assert abs(float(loss.detach())-reference)<2e-5
            loss.backward()
            assert tg.grad is not None and float(tg.grad.abs().sum())>0
            assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
            hg=[sum(float(p.grad.abs().sum()) for p in head.parameters() if p.grad is not None) for head in (model.head1,model.head2)]
            expected=c.flags(arm)
            assert (hg[0]>0)==expected['Con_s'] and (hg[1]>0)==expected['Con_t'],(arm,hg)
            for head,key in ((model.head1,'Con_s'),(model.head2,'Con_t')):
                if not expected[key]:assert all(p.grad is None for p in head.parameters())
            head_before=[c.state_hash(h) for h in (model.head1,model.head2)]
            before=c.state_hash(model);opt=c.optimizer(model,'O');opt.step();assert c.state_hash(model)!=before
            for j,head in enumerate((model.head1,model.head2)):
                assert (c.state_hash(head)!=head_before[j])==expected[('Con_s','Con_t')[j]]
            checks.append({'seed':seed,'arm':arm,'initial_sha256':start,'parameters':645211,
                'logits_sha256':logits,'projection_grad_l1':hg,'main_target_grad_l1':float(tg.grad.abs().sum()),
                'objective_terms':terms,'BN_num_batches':count,'disabled_heads_grad_None_and_no_weight_decay_update':True})
            del model,opt,records,r,tx,tg,loss,head
            gc.collect();torch.cuda.empty_cache()
        assert len(set(starts))==len(set(randoms))==1 and all(v==bn[0] for v in bn)
        del source,target,loaders
    c.dump(c.OUT/'verification.json',{'passed':True,'checks':checks,'target_GT_values_accessed':False,
         'CE_Full_loss_and_gradients_bit_identical_to_reused_code':True,
         'same_init_logits_forward_RNG_BN_all_five_arms':True})
    print('All15 preflights passed; CE/Full reused,9 removal runs ready.',flush=True)

if __name__=='__main__':main()

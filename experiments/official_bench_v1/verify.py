"""Pre-launch native training, dimension, numerical and joint-gradient gate."""
import json
import torch
import torch.nn.functional as F
import common as c


def main():
    torch.set_num_threads(2);device=torch.device('cuda:7');torch.cuda.set_device(device)
    args=c.options();args.num_workers=0
    c.seed_worker(2100);loaders,audit=c.loaders(args)
    (x,y),(t,_)=next(iter(loaders[0])),next(iter(loaders[2]))
    x,y,t=x[:16].to(device),y[:16].to(device),t[:16].to(device)
    rows=[]
    for seed in (2100,2101,2102):
        initial=[]
        for arm in ('A','B','C'):
            c.seed_worker(seed);args.seed=seed
            model,ema,h=c.build(args,arm);initial.append(h)
            model.to(device).train();ti=t.detach().clone().requires_grad_(True)
            outputs=model(x,ti)
            assert len(outputs)==4 and all(v.shape==(16,12) and torch.isfinite(v).all() for v in outputs)
            loss=F.cross_entropy(outputs[0],y);loss.backward()
            assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
            target_grad=0. if ti.grad is None else ti.grad.abs().sum().item()
            if arm!='A':assert target_grad>0
            if arm=='C':
                assert model.relation_scores.mlp[0].in_features==176
                assert all(p.grad is not None and p.grad.abs().sum()>0 for p in model.relation_scores.parameters())
            model.eval()
            with torch.no_grad():assert torch.isfinite(model(x,t)[1]).all()
            rows.append({'seed':seed,'arm':arm,'initial_common_hash':h,'parameters':sum(p.numel() for p in model.parameters()),
                         'source_only_CE_target_input_grad_l1':target_grad,'passed':True})
            del model,ema,ti
        assert len(set(initial))==1
    # Execute the compiled native loop for ONE diagnostic update, with actual
    # source/target patches but all-zero target labels. No target evaluation.
    c.seed_worker(2100);args.seed=2100;args.epoch=1
    model,ema,_=c.build(args,'A');model.to(device);ema.to(device)
    opt,sched=c.load_scheduler('BiDA',model,args);criterion,_=c.make_loss(args,num_classes=12)
    calls=[]
    def callback(epoch,network,teacher,optimizer,global_step):calls.append((epoch,global_step))
    native,diff=c.native_train()
    native(model,ema,opt,criterion,12,[(x,y)],[],[(t,torch.zeros(16,dtype=torch.long))],[],args,'unused',device,sched,callback)
    assert calls==[(1,1)] and all(torch.isfinite(p).all() for p in model.parameters())
    result={'all_passed':True,'runs':rows,'dataset_protocol':audit,
            'native_full_A_one_step':'passed','native_loss_expression_order_preserved':True,
            'joint_target_gradient_in_B_C':'nonzero','new_target_metrics_inspected':False}
    (c.OUT/'implementation_checks.json').write_text(json.dumps(result,indent=2)+'\n')
    _,derived,relation_diff=c.relation_adapter(176)
    (c.OUT/'relation_176_bands.py').write_text(derived)
    (c.OUT/'relation_band_adaptation.patch').write_text(relation_diff)
    (c.OUT/'official_loop_diff.patch').write_text(diff)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()

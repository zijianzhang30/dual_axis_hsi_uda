"""Ensure the added cell delegates to native sequential BN unchanged."""
import json
import torch
import torch.nn.functional as F
import train


def main():
    torch.set_num_threads(2)
    device=torch.device('cuda:7');torch.cuda.set_device(device)
    args=type('Args',(),{'seed':2100,'num_workers':0,'dataset_dir':'/home/zhangzj26/TGRS_MLUDA-2024/datasets/Houston/'})()
    train.base.seed_worker(2100)
    source,_,target,_,_,_=train.base.make_loaders(args)
    x,y=next(iter(source));t,_=next(iter(target))
    x,y,t=x[:16].to(device),y[:16].to(device),t[:16].to(device)
    reports=[]
    for seed in (2100,2101,2102):
        train.base.seed_worker(seed)
        model,initial=train.build(seed)
        reference=json.loads((train.ROOT/f'results/relation_tokenizer_v1/C_{seed}/config.json').read_text())
        assert initial==reference['initial_original_model_sha256']
        assert train.base.helper.state_hash(model)==reference['initial_full_model_sha256']
        assert train.base.helper.state_hash(model.relation_scores)==reference['initial_relation_mlp_sha256']
        model.to(device).train()
        order=[];handles=[]
        for name,module in (('bn3',model.conv3d_features[1]),('bn2',model.conv2d_features[1])):
            handles.append(module.register_forward_hook(lambda m,i,o,n=name:order.append(n)))
        counts=[v.num_batches_tracked.item() for v in (model.conv3d_features[1],model.conv2d_features[1])]
        ti=t.detach().clone().requires_grad_(True)
        loss=F.cross_entropy(model(x,ti)[0],y);assert torch.isfinite(loss)
        loss.backward()
        for handle in handles:handle.remove()
        assert order==['bn3','bn2','bn3','bn2']
        assert [v.num_batches_tracked.item()-c for v,c in zip((model.conv3d_features[1],model.conv2d_features[1]),counts)]==[2,2]
        assert all(v.grad is not None and torch.isfinite(v.grad).all() and v.grad.abs().sum()>0 for v in model.relation_scores.parameters())
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        assert ti.grad is None or ti.grad.abs().sum()==0
        model.eval()
        with torch.no_grad():assert torch.isfinite(model(x,t)[1]).all()
        reports.append({'seed':seed,'native_forward_preserved':True,'bn_order':order,'native_counter_deltas':[2,2],
                        'initial_state_exact_with_cell4':True,'loss':loss.item(),'all_checks_passed':True})
    dest=train.ROOT/'results/relation_bn_2x2_v1/implementation_checks.json'
    if dest.exists():raise RuntimeError('Refuse overwrite')
    dest.write_text(json.dumps(reports,indent=2)+'\n')
    print(json.dumps(reports,indent=2))


if __name__=='__main__':main()

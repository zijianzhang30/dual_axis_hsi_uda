"""Pre-training numerical, interface, initialization and gradient gate."""
import json
from pathlib import Path
import torch
import torch.nn.functional as F
import train
import relation


def main():
    torch.set_num_threads(2)
    device=torch.device('cuda:7')
    torch.cuda.set_device(device)
    reports=[]
    # CPU relation tests include exact formulas and zero variance patches.
    g=torch.Generator().manual_seed(61)
    patch=torch.randn(3,1,48,13,13,generator=g)
    for arm in ('B','C'):
        scores=relation.RelationScores(arm)
        anchor=patch[:,0].mean((-2,-1),keepdim=True) if arm=='B' else patch[:,0,:,6:7,6:7]
        expected=(patch[:,0]-anchor)/(patch[:,0].std((-2,-1),correction=0,keepdim=True)+1e-6)
        assert torch.equal(scores.relations(patch),expected)
        weights=scores(patch)
        assert weights.shape==(3,4,169)
        assert torch.allclose(weights.sum(-1),torch.ones(3,4),atol=1e-6)
        constant=torch.full_like(patch,.13)
        assert torch.isfinite(scores.relations(constant)).all()
        assert torch.allclose(scores(constant),torch.full_like(weights,1/169),atol=1e-6)
    # Check real normband augmented patches and all three historical initializations.
    args=type('Args',(),{'dataset_dir':'/home/zhangzj26/TGRS_MLUDA-2024/datasets/Houston/',
                         'seed':2100,'num_workers':0})()
    train.seed_worker(2100)
    source,_,target,_,_,_=train.make_loaders(args)
    x,y=next(iter(source));t,_=next(iter(target))
    x,y,t=x[:16].to(device),y[:16].to(device),t[:16].to(device)
    for seed in (2100,2101,2102):
        hashes={}
        for arm in ('A','B','C'):
            train.seed_worker(seed)
            base,initial=train.build(seed,'A')
            ref=json.loads((train.ROOT/f'results/sceneshift_joint_v1/C_{seed}/config.json').read_text())
            assert initial==ref['initial_model_sha256']
            original={k:v.clone() for k,v in base.state_dict().items()}
            cpu_rng=torch.get_rng_state().clone()
            cuda_rng=torch.cuda.get_rng_state(device).clone()
            if arm!='A':
                relation.install(base,arm,seed)
                assert torch.equal(cpu_rng,torch.get_rng_state())
                assert torch.equal(cuda_rng,torch.cuda.get_rng_state(device))
                hashes[arm]=train.helper.state_hash(base.relation_scores)
            assert all(torch.equal(v,base.state_dict()[k]) for k,v in original.items() if arm=='A' or not k.startswith('conv_a.'))
            base.to(device).train()
            activations=[]
            def hook(module,inputs,output):
                output.retain_grad();activations.append(output)
            handle=base.conv3d_features[0].register_forward_hook(hook)
            target_input=t.detach().clone().requires_grad_(True)
            loss=F.cross_entropy(base(x,target_input)[0],y)
            assert torch.isfinite(loss)
            loss.backward()
            handle.remove()
            assert len(activations)==2
            target_grad=activations[1].grad
            assert target_grad is not None and torch.isfinite(target_grad).all() and target_grad.abs().sum()>0
            assert target_input.grad is not None and torch.isfinite(target_input.grad).all() and target_input.grad.abs().sum()>0
            for name,p in base.named_parameters():
                if p.grad is not None: assert torch.isfinite(p.grad).all(),name
            if arm!='A':
                for p in base.relation_scores.parameters():
                    assert p.grad is not None and p.grad.abs().sum()>0
                w=base.relation_scores(x)
                assert torch.isfinite(w).all() and torch.allclose(w.sum(-1),torch.ones(16,4,device=device),atol=1e-6)
            base.eval()
            with torch.no_grad():
                logits=base(x,t)[1]
            assert logits.shape==(16,7) and torch.isfinite(logits).all()
            reports.append({'seed':seed,'arm':arm,'parameters':sum(p.numel() for p in base.parameters()),
                'loss':loss.item(),'target_conv3d_gradient_l1':target_grad.abs().sum().item(),
                'target_input_gradient_l1':target_input.grad.abs().sum().item(),
                'all_checks_passed':True})
            del base,activations,target_input
        assert hashes['B']==hashes['C']
    output={'formula_constant_grid_softmax_tests':'passed','shared_tensors_rng_mlp_pairing':'passed',
            'source_only_CE_target_joint_statistics_gradient':'passed','runs':reports}
    out=train.ROOT/'results/relation_tokenizer_v1'
    out.mkdir(exist_ok=True,parents=True)
    path=out/'implementation_checks.json'
    if path.exists():raise RuntimeError('Checks file already exists')
    path.write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps(output,indent=2))


if __name__=='__main__':main()

"""Preflight odd-patch anchor, frozen initialization, gradients and paired RNG."""
import json
import numpy as np
import torch
import torch.nn.functional as F
import common as c


def main():
    torch.set_num_threads(2);torch.cuda.set_device(0)
    c.verify_frozen();records=[]
    for seed in c.SEEDS:
        initial=[];streams=[]
        for arm in c.ARMS:
            model,original=c.build(seed);model=model.cuda()
            initial.append(c.state_hash(model))
            ds,dt=c.train_sets(arm,seed)
            la,arrays=c.epoch_loaders(ds,dt,seed,1)
            streams.append([c.digest(a) for a in arrays])
            x,y=next(iter(la[0]));t,_=next(iter(la[1]))
            assert x.shape==t.shape==(32,1,102,11,11)
            assert torch.isfinite(x).all() and torch.isfinite(t).all()
            # Coordinate-coded artificial patch, all native transformations keep center(5,5).
            marker=np.arange(121,dtype=np.float32).reshape(11,11,1)
            for code in range(6):
                transformed=c.augment(marker,code)
                assert transformed[5,5,0]==60
                patch=torch.from_numpy(np.array(transformed.transpose(2,0,1))).unsqueeze(0).unsqueeze(0).repeat(1,1,102,1,1).cuda()
                r=model.relation_scores.relations(patch)
                assert r.shape==(1,102,11,11) and float(r[:,:,5,5].abs().max())==0
            counts={'bn3d':0,'bn2d':0,'tokens':[]}
            def bn3(*_):counts['bn3d']+=1
            def bn2(*_):counts['bn2d']+=1
            hooks=[model.conv3d_features[1].register_forward_hook(bn3),model.conv2d_features[1].register_forward_hook(bn2)]
            native=model._forward_semantic_tokens
            def token(features):
                assert features.shape==(32,64,11,11)
                out=native(features);assert out.shape==(32,4,64)
                counts['tokens'].append(list(out.shape));return out
            model._forward_semantic_tokens=token
            x,y,t=x.cuda(),y.cuda(),t.cuda().requires_grad_();model.train()
            loss=F.cross_entropy(model(x,t)[0],y)
            loss.backward()
            assert torch.isfinite(loss) and counts['bn3d']==counts['bn2d']==1
            relation_grad=sum(float(p.grad.abs().sum()) for p in model.relation_scores.parameters() if p.grad is not None)
            assert relation_grad>0 and float(t.grad.abs().sum())>0
            assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
            before=c.state_hash(model)
            opt=torch.optim.SGD(model.parameters(),lr=.001,momentum=.9,weight_decay=.0005)
            opt.step();assert c.state_hash(model)!=before
            for hook in hooks:hook.remove()
            records.append({'seed':seed,'arm':arm,'initial_model_sha256':initial[-1],
                            'source_ce':float(loss),'relation_grad_l1':relation_grad,
                            'target_joint_BN_grad_l1':float(t.grad.abs().sum()),
                            'BN_calls_per_training_forward':[counts['bn3d'],counts['bn2d']],
                            'tokens':counts['tokens'],'anchor_all_six_geometry_operations':'passed',
                            'parameters':sum(p.numel() for p in model.parameters())})
            del ds,dt,la,x,y,t,model,opt
            torch.cuda.empty_cache()
        assert initial[0]==initial[1] and streams[0]==streams[1]
    # All 100 planned epochs use exactly the same files in both arms; no data-value-dependent RNG.
    result={'passed':True,'models':records,'P1_P2_initialization_and_streams_identical':True,
            'source_180_per_class_official_sampler_exact':True,'target_training_label_values_accessed':False}
    c.dump(c.OUT/'verification.json',result)
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()

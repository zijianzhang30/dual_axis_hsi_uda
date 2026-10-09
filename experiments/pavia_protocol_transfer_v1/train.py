"""One fixed100 frozen-C protocol run. No target metrics during training."""
import argparse
import hashlib
import json
import time
import numpy as np
import torch
import torch.nn.functional as F
import common as c


def main():
    p=argparse.ArgumentParser();p.add_argument('--arm',choices=c.ARMS,required=True)
    p.add_argument('--seed',type=int,choices=c.SEEDS,required=True);a=p.parse_args()
    assert json.loads((c.OUT/'verification.json').read_text())['passed']
    c.verify_frozen()
    folder=c.OUT/('%s_%d'%(a.arm,a.seed));folder.mkdir(exist_ok=False);(folder/'checkpoints').mkdir()
    torch.set_num_threads(2);device=torch.device('cuda:0');torch.cuda.set_device(device)
    model,initial=c.build(a.seed);model.to(device)
    source,target=c.train_sets(a.arm,a.seed)
    cfg=c.config(a.seed,a.arm)
    cfg.update(initial_original_model_sha256=initial,initial_full_model_sha256=c.state_hash(model),
               initial_scorer_sha256=c.state_hash(model.relation_scores),parameters=sum(p.numel() for p in model.parameters()),
               source_selection_sha256=c.sha(c.OUT/'inputs'/('source_selection_%d.npz'%a.seed)),
               plan_sha256=c.sha(c.OUT/'plans'/('%d.npz'%a.seed)),
               source_cube_sha256=c.sha(c.OUT/'inputs'/('%s_source.npy'%a.arm)),
               target_cube_sha256=c.sha(c.OUT/'inputs'/('%s_target.npy'%a.arm)),
               provenance_sha256=c.sha(c.OUT/'provenance.json'))
    c.dump(folder/'config.json',cfg);print(json.dumps(cfg),flush=True)
    checkpoints=[];torch.cuda.synchronize();start=time.monotonic()
    with (folder/'history.jsonl').open('x') as history:
        for epoch in range(1,101):
            # Preserve the actual official MLUDA epoch boundary: momentum resets each epoch.
            opt=torch.optim.SGD(model.parameters(),lr=.001,momentum=.9,weight_decay=.0005)
            model.train();loaders,arrays=c.epoch_loaders(source,target,a.seed,epoch)
            orderhash=hashlib.sha256()
            for arr in arrays:orderhash.update(arr.tobytes())
            rnghash=hashlib.sha256();inputhash=hashlib.sha256();total=count=hits=steps=0
            for (x,y),(t,_) in zip(*loaders):
                assert x.shape==t.shape==(32,1,102,11,11)
                rnghash.update(torch.get_rng_state().numpy().tobytes())
                rnghash.update(torch.cuda.get_rng_state(device).numpy().tobytes())
                for value in (x,y,t):inputhash.update(value.numpy().tobytes())
                x,y,t=x.to(device),y.to(device),t.to(device)
                opt.zero_grad(set_to_none=True);logits=model(x,t)[0]
                loss=F.cross_entropy(logits,y);assert torch.isfinite(loss)
                loss.backward();opt.step()
                total+=loss.item()*len(y);count+=len(y);hits+=int((logits.argmax(1)==y).sum());steps+=1
            assert steps==38 and count==1216
            row={'epoch':epoch,'steps':steps,'source_ce':total/count,'source_train_OA':100*hits/count,
                 'order_and_augmentation_sha256':orderhash.hexdigest(),'model_rng_sha256':rnghash.hexdigest(),
                 'input_batch_sha256':inputhash.hexdigest(),'target_scoring':False}
            if epoch==1 or epoch%10==0:
                torch.cuda.synchronize();row['elapsed_seconds']=time.monotonic()-start
                path=folder/'checkpoints'/('epoch_%03d.pth'%epoch)
                torch.save({'model':model.state_dict(),'epoch':epoch,'config':cfg},path)
                checkpoints.append({'epoch':epoch,'path':str(path),'sha256':c.sha(path)})
                print(json.dumps(row),flush=True)
            history.write(json.dumps(row)+'\n');history.flush()
    torch.cuda.synchronize();seconds=time.monotonic()-start
    del source,target,loaders,opt
    # Freeze endpoint predictions without loading target label values.
    before=c.state_hash(model);model.eval();start=time.monotonic()
    dataset=c.PatchSet(a.arm,'target',np.load(c.OUT/'inputs/target_training_coords.npy'))
    loader=torch.utils.data.DataLoader(dataset,batch_size=128,shuffle=False,drop_last=False,num_workers=0,
              generator=torch.Generator().manual_seed(a.seed+83000))
    outputs=[]
    with torch.inference_mode():
        for x,_ in loader:
            x=x.to(device);outputs.append(model(x,x)[1].cpu().numpy())
    logits=np.concatenate(outputs);assert np.isfinite(logits).all() and c.state_hash(model)==before
    path=folder/'predictions_epoch100.npz'
    np.savez_compressed(path,logits=logits,predictions=logits.argmax(1))
    c.dump(folder/'frozen_manifest.json',{'config':cfg,'epoch':100,'train_seconds':seconds,
           'inference_seconds':time.monotonic()-start,'checkpoints':checkpoints,
           'prediction':{'path':str(path),'sha256':c.sha(path)},'no_target_scoring':True})
    print('Fixed epoch100 predictions frozen; target metrics deferred.',flush=True)


if __name__=='__main__':main()

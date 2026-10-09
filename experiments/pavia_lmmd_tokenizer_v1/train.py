"""100 predeclared epochs; paired three-view execution, only objective changes."""
import argparse
import hashlib
import json
import time
import numpy as np
import torch
import common as c


def main():
    p=argparse.ArgumentParser();p.add_argument('--arm',choices=c.NEW_ARMS,required=True)
    p.add_argument('--seed',type=int,choices=c.SEEDS,required=True);a=p.parse_args()
    c.forbid_target_label_loading();c.verify_files();assert json.loads((c.OUT/'verification.json').read_text())['passed']
    folder=c.OUT/('%s_%d'%(a.arm,a.seed));folder.mkdir(exist_ok=False);(folder/'checkpoints').mkdir()
    network='O';enabled=True
    reference_rows=[json.loads(s) for s in (c.run_folder('C_CE',a.seed)/'history.jsonl').read_text().splitlines()]
    torch.set_num_threads(2);torch.cuda.set_device(0);device=torch.device('cuda:0')
    b=c.base();_,mmd,contrast,augmentation=c.official()
    model,initial=c.build(a.seed,a.arm[0]);model=model.to(device)
    con=contrast.SupConLoss(temperature=.1).to(device)
    source,target=b.train_sets('P2',a.seed)
    cfg=json.loads((c.OUT/'frozen_config.json').read_text())
    cfg.update(arm=a.arm,seed=a.seed,network=network,adaptation_enabled=enabled,
               initial_base_model_sha256=initial,initial_complete_model_sha256=c.state_hash(model),
               parameters=sum(p.numel() for p in model.parameters()),provenance_sha256=c.sha(c.OUT/'provenance.json'),
               enabled_terms=c.flags(a.arm),tokenizer=a.arm[0],initial_shared_model_sha256=c.shared_hash(model))
    ref=json.loads((c.run_folder('C_CE',a.seed)/'config.json').read_text())
    if a.arm[0]=='C':
        for k in ('initial_base_model_sha256','initial_complete_model_sha256','parameters'):assert cfg[k]==ref[k]
    preflight=json.loads((c.OUT/'verification.json').read_text())
    check=next(v for v in preflight['checks'] if v['arm']==a.arm and v['seed']==a.seed)
    assert cfg['initial_shared_model_sha256']==check['shared_initial_sha256']
    assert cfg['initial_complete_model_sha256']==check['complete_initial_sha256']
    c.dump(folder/'config.json',cfg);print(json.dumps(cfg),flush=True)
    torch.cuda.synchronize();start=time.monotonic();checkpoints=[]
    with (folder/'history.jsonl').open('x') as history:
        for epoch in range(1,101):
            model.train();opt=c.optimizer(model,network)
            loaders,plan=b.epoch_loaders(source,target,a.seed,epoch)
            viewhash=hashlib.sha256();rnghash=hashlib.sha256();total=ce=count=hits=steps=0
            lmmd_sum=cs_sum=ct_sum=0.;pseudo_counts=np.zeros(7,dtype=np.int64)
            for (x,y),(t,_) in zip(*loaders):
                steps+=1;assert x.shape==t.shape==(32,1,102,11,11)
                last_source=x.clone()
                batchviews=c.views(x,t,a.seed,epoch,steps,augmentation)
                for sx,tx in batchviews:
                    viewhash.update(sx.numpy().tobytes());viewhash.update(tx.numpy().tobytes())
                viewhash.update(y.numpy().tobytes())
                rnghash.update(torch.get_rng_state().numpy().tobytes())
                rnghash.update(torch.cuda.get_rng_state().cpu().numpy().tobytes())
                opt.zero_grad(set_to_none=True)
                records=[model(sx.to(device),tx.to(device)) for sx,tx in batchviews]
                loss,terms=c.objective(records,y.to(device),epoch,a.arm,mmd,con)
                assert torch.isfinite(loss).all();loss.backward();opt.step()
                total+=float(loss.detach())*32;ce+=terms['CE']*32;count+=32
                hits+=int((records[0]['ls'].argmax(1).cpu()==y).sum())
                lmmd_sum+=terms['LMMD'];cs_sum+=terms['Con_s'];ct_sum+=terms['Con_t']
                if enabled:pseudo_counts+=terms['pseudo_counts']
                del records,loss,batchviews
            assert steps==38 and count==1216
            bc=c.bn_counts(model)
            for name,n in bc.items():
                expected=0 if name.endswith(('bn9','bn10')) else epoch*38*(6 if network=='M' else 3)
                assert n==expected,(name,n,expected)
            row={'epoch':epoch,'steps':steps,'loss':total/count,'source_CE':ce/count,'source_train_OA':100*hits/count,
                 'LMMD':lmmd_sum/steps,'Con_s':cs_sum/steps,'Con_t':ct_sum/steps,'pseudo_counts':pseudo_counts.tolist(),
                 'plan_sha256':c.trace_hash(plan),'all_input_views_sha256':viewhash.hexdigest(),
                 'model_rng_sha256':rnghash.hexdigest(),'BN_num_batches':bc,'target_scoring':False}
            for key in ('plan_sha256','all_input_views_sha256','model_rng_sha256','BN_num_batches'):
                assert row[key]==reference_rows[epoch-1][key],(a.arm,a.seed,epoch,key)
            row['optimized_terms']=c.flags(a.arm)
            row['contrastive_optimized_or_computed']=False
            if epoch==1 or epoch%10==0:
                torch.cuda.synchronize();row['elapsed_seconds']=time.monotonic()-start
                path=folder/'checkpoints'/('epoch_%03d.pth'%epoch)
                torch.save({'model':model.state_dict(),'epoch':epoch,'config':cfg},path)
                checkpoints.append({'epoch':epoch,'path':str(path),'sha256':c.sha(path)})
                print(json.dumps(row),flush=True)
            history.write(json.dumps(row)+'\n');history.flush()
    torch.cuda.synchronize();train_seconds=time.monotonic()-start
    np.save(folder/'last_source_reference.npy',last_source.numpy())
    del source,target,loaders,opt
    # Same exact pixel cohort/order in all cells. Native M cross-attention requires paired batch32.
    dataset=b.PatchSet('P2','target',np.load(c.P2/'inputs/target_training_coords.npy'))
    loader=torch.utils.data.DataLoader(dataset,batch_size=32,shuffle=False,drop_last=False,num_workers=0,
                  generator=torch.Generator().manual_seed(a.seed+85000))
    model.eval();before=c.state_hash(model);ref=last_source.to(device);outputs=[];start=time.monotonic()
    with torch.inference_mode():
        for x,_ in loader:
            n=len(x)
            if n<32:x=torch.cat((x,x[-1:].expand(32-n,-1,-1,-1,-1)),0)
            x=x.to(device);r=model(ref,x) if network=='M' else model(x,x)
            outputs.append(r['lt'][:n].cpu().numpy())
    logits=np.concatenate(outputs);assert len(logits)==39348 and np.isfinite(logits).all()
    assert c.state_hash(model)==before
    path=folder/'predictions_epoch100.npz';np.savez_compressed(path,logits=logits,predictions=logits.argmax(1))
    c.dump(folder/'frozen_manifest.json',{'config':cfg,'epoch':100,'training_seconds':train_seconds,
           'inference_seconds':time.monotonic()-start,'checkpoints':checkpoints,
           'prediction':{'path':str(path),'sha256':c.sha(path)},
           'source_reference_sha256':c.sha(folder/'last_source_reference.npy'),'no_target_scoring':True})
    print('Fixed epoch100 predictions frozen; no target-best or target scoring.',flush=True)


if __name__=='__main__':main()

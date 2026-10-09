"""No target class values used by training; strict paired streams against prior CE-only run."""
import argparse
import hashlib
import json
import math
import time
import numpy as np
import torch
import torch.nn.functional as F
import common as c

def preflight(dataset,seed):
    outputs=[]
    for arm in ('A','C'):
        model,loaders,helper,classes,audit=c.setup(dataset,arm,seed)
        before=torch.get_rng_state().clone()
        assert audit['complete_initial_hash']==c.state_hash(model)
        # source-only/target dummy labels; probe both streams and restore the entire model/RNG.
        x,y=next(iter(loaders[0]));t,_=next(iter(loaders[2]));assert x.shape==t.shape==(128,1,48 if dataset=='houston' else 176,13,13)
        model=model.cuda().train();state={k:v.clone() for k,v in model.state_dict().items()}
        cpu=torch.get_rng_state().clone();gpu=torch.cuda.get_rng_state().clone();capture=[]
        hook=model.norm.register_forward_hook(lambda mod,inp,out:capture.append(out[:,0]))
        r=model(x.cuda(),t.cuda());assert len(capture)==4
        ce=F.cross_entropy(r[0],y.cuda());m=c.lmmd();loss=m.lmmd(capture[0],capture[1],y.cuda(),r[1].softmax(1),CLASS_NUM=classes,BATCH_SIZE=128)
        assert torch.isfinite(loss).all()
        grads=torch.autograd.grad(loss,list(model.parameters()),retain_graph=True,allow_unused=True)
        norm=sum(float(g.square().sum()) for g in grads if g is not None)**.5;assert norm>0
        first=r[0].detach().clone();capture.clear();model.load_state_dict(state)
        torch.set_rng_state(cpu);torch.cuda.set_rng_state(gpu);rr=model(x.cuda(),t.cuda())
        error=float((rr[0]-first).abs().max());assert error<1e-6
        assert torch.equal(F.cross_entropy(rr[0],y.cuda()),ce.detach())
        if arm=='C':
            weights=model.relation_scores(x.cuda());rel=model.relation_scores.relations(x.cuda())
            assert weights.shape==(128,4,169) and float(rel[:,:,6,6].abs().max())==0
        hook.remove();model.load_state_dict(state);assert audit['complete_initial_hash']==c.state_hash(model)
        outputs.append({'dataset':dataset,'seed':seed,'arm':arm,**audit,'LMMD':float(loss.detach()),'LMMD_gradient_norm':norm,
                        'lambda0_logits_max_error':error,'source_target_label_values_not_used':True})
        del model,loaders,r,rr,grads,capture,x,t,y
    assert outputs[0]['shared_initial_hash']==outputs[1]['shared_initial_hash']
    c.dump(c.OUT/('preflight_%s_%d.json'%(dataset,seed)),{'passed':True,'checks':outputs})

def train(dataset,arm,seed):
    c.verify_files();assert json.loads((c.OUT/('preflight_%s_%d.json'%(dataset,seed))).read_text())['passed']
    folder=c.OUT/dataset/('%s_LMMD_%d'%(arm,seed));folder.mkdir(parents=True,exist_ok=False);(folder/'checkpoints').mkdir()
    torch.set_num_threads(2);torch.cuda.set_device(0)
    model,loaders,helper,classes,audit=c.setup(dataset,arm,seed);source,val,target,target_eval=loaders
    clean=c.clean_dataset(target_eval.dataset) if dataset=='hyrank' else None
    # Access to target class values is removed from dataset outputs before any training iterator.
    target.dataset.labels=np.zeros_like(target.dataset.labels)
    target.dataset.label=np.zeros_like(target.dataset.label)
    model=model.cuda();native=c.lmmd();captures=[]
    hook=model.norm.register_forward_hook(lambda mod,inp,out:captures.append(out[:,0]))
    optimizer=torch.optim.SGD(model.parameters(),lr=.01)
    cfg={**c.PLAN,'dataset':dataset,'arm':arm,'seed':seed,'classes':classes,**audit,
         'parameters':sum(p.numel() for p in model.parameters()),'CE_reference':str(c.baseline(dataset,arm,seed))}
    c.dump(folder/'config.json',cfg)
    reference=[json.loads(s) for s in (c.baseline(dataset,arm,seed)/'history.jsonl').read_text().splitlines()]
    checkpoints=[];start=time.monotonic()
    with (folder/'history.jsonl').open('x') as history:
      for epoch in range(1,201):
        model.train();batchhash=hashlib.sha256();rnghash=hashlib.sha256();count=hits=steps=0;ces=lms=totals=0.
        weight=.3*(2/(1+math.exp(-10*epoch/200))-1)
        for (x,y),(t,unused) in zip(source,target):
            if len(x)!=len(t):continue
            assert not unused.any()
            for v in (x,y,t):batchhash.update(v.numpy().tobytes())
            rnghash.update(torch.get_rng_state().numpy().tobytes());rnghash.update(torch.cuda.get_rng_state().numpy().tobytes())
            x,y,t=x.cuda(),y.cuda(),t.cuda();optimizer.zero_grad(set_to_none=True);captures.clear()
            r=model(x,t);assert len(captures)==4
            ce=F.cross_entropy(r[0],y);lmd=native.lmmd(captures[0],captures[1],y,r[1].softmax(1),CLASS_NUM=classes,BATCH_SIZE=len(y))
            loss=ce+weight*lmd;assert torch.isfinite(loss).all();loss.backward();optimizer.step()
            n=len(y);count+=n;steps+=1;hits+=int((r[0].argmax(1)==y).sum());ces+=float(ce.detach())*n;lms+=float(lmd.detach())*n;totals+=float(loss.detach())*n
        expected=18 if dataset=='houston' else 80;assert steps==expected,(steps,expected)
        row={'epoch':epoch,'steps':steps,'source_ce':ces/count,'LMMD':lms/count,'loss':totals/count,'LMMD_weight':weight,
             'source_train_OA':hits/count*100,'batch_stream_sha256':batchhash.hexdigest(),'model_rng_sha256':rnghash.hexdigest(),'target_scoring':False}
        rngkey='original_rng_stream_sha256' if dataset=='houston' else 'rng_stream_sha256'
        assert row['batch_stream_sha256']==reference[epoch-1]['batch_stream_sha256'],(dataset,arm,seed,epoch,'data order')
        assert row['model_rng_sha256']==reference[epoch-1][rngkey],(dataset,arm,seed,epoch,'model RNG')
        counts={k:int(v.num_batches_tracked) for k,v in model.named_modules() if isinstance(v,(torch.nn.BatchNorm2d,torch.nn.BatchNorm3d))}
        assert len(counts)==2 and all(n==epoch*steps for n in counts.values());row['BN_batches']=counts
        if epoch==1 or epoch%10==0:
            row['source_val']=helper.source_evaluate(model,val,torch.device('cuda:0'));row['elapsed_seconds']=time.monotonic()-start
            path=folder/'checkpoints'/('epoch_%03d.pth'%epoch);torch.save({'model':model.state_dict(),'epoch':epoch,'config':cfg},path)
            checkpoints.append({'epoch':epoch,'path':str(path),'sha256':c.sha(path)});print(json.dumps(row),flush=True)
        history.write(json.dumps(row)+'\n');history.flush()
    torch.cuda.synchronize();seconds=time.monotonic()-start;hook.remove();before=c.state_hash(model)
    # GT values removed from eval loader too; final predictions frozen before scoring barrier.
    indices=target_eval.dataset.indices.copy();target_eval.dataset.labels=np.zeros_like(target_eval.dataset.labels)
    target_eval.dataset.label=np.zeros_like(target_eval.dataset.label)
    logits,pred=helper.predict(model,target_eval,torch.device('cuda:0'));assert before==c.state_hash(model)
    path=folder/'predictions_epoch200.npz';np.savez_compressed(path,logits=logits,predictions=pred)
    np.save(folder/'evaluation_indices.npy',indices)
    clean_prediction=None
    if clean is not None:
        loader=torch.utils.data.DataLoader(clean,batch_size=128,shuffle=False,num_workers=0,generator=torch.Generator().manual_seed(seed+85000))
        cl,cp=helper.predict(model,loader,torch.device('cuda:0'));assert before==c.state_hash(model)
        pp=folder/'predictions_clean_epoch200.npz';np.savez_compressed(pp,logits=cl,predictions=cp)
        np.save(folder/'clean_evaluation_indices.npy',clean.indices)
        clean_prediction={'path':str(pp),'sha256':c.sha(pp),'n':len(cp)}
    c.dump(folder/'frozen_manifest.json',{'config':cfg,'checkpoints':checkpoints,'prediction':{'path':str(path),'sha256':c.sha(path)},
       'clean_prediction':clean_prediction,'train_seconds':seconds,'source_val_at_final':row['source_val'],'no_target_scoring':True})
    print('FIXED_EPOCH200_COMPLETE_NO_TARGET_SCORING',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--dataset',choices=['houston','hyrank'],required=True);p.add_argument('--arm',choices=['A','C']);p.add_argument('--seed',type=int,choices=c.SEEDS,required=True);p.add_argument('--preflight',action='store_true');a=p.parse_args()
    if a.preflight:torch.set_num_threads(2);preflight(a.dataset,a.seed)
    else:assert a.arm;train(a.dataset,a.arm,a.seed)

import argparse,hashlib,json,pickle,random,time
import numpy as np,torch
import torch.nn.functional as F
import study as s
def rng(g):return {'shared_loader':hashlib.sha256(g.get_state().numpy().tobytes()).hexdigest(),'torch_cpu':hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest(),'torch_cuda':hashlib.sha256(torch.cuda.get_rng_state().numpy().tobytes()).hexdigest(),'python':hashlib.sha256(pickle.dumps(random.getstate())).hexdigest(),'numpy':hashlib.sha256(pickle.dumps(np.random.get_state())).hexdigest()}
@torch.inference_mode()
def source_val(m,loader):
 m.eval();n=hit=0;ce=0.
 for x,y in loader:
  x=x.cuda();y=y.cuda();z=m(x,x)[1];n+=len(y);hit+=int((z.argmax(1)==y).sum());ce+=float(F.cross_entropy(z,y))*len(y)
 return {'OA':100*hit/n,'CE':ce/n}
@torch.inference_mode()
def freeze(m,ev,fullcoords,d,seed,epoch,rt,folder):
 m.eval();before=rt['hash'](m);cpu=torch.get_rng_state().clone();cuda=torch.cuda.get_rng_state().clone();rel=[]
 for batch in ev:
  x,y=batch[:2];assert not y.any();x=x.cuda();rel.append(m(x,x)[1].cpu().numpy())
 rel=np.concatenate(rel);tasks=[('native',ev.dataset.indices,rel)]
 if d=='hyrank':
  # Native subset and original-boundary extra pixels share the same normalized cube.
  ds=ev.dataset;full=[]
  for i in range(0,len(fullcoords),128):
   xy=fullcoords[i:i+128];x=np.stack([ds.data[a-6:a+7,b-6:b+7].transpose(2,0,1) for a,b in xy]).astype(np.float32);x=torch.from_numpy(x[:,None]).cuda();full.append(m(x,x)[1].cpu().numpy())
  tasks.append(('full',fullcoords,np.concatenate(full)))
 assert rt['hash'](m)==before and torch.equal(cpu,torch.get_rng_state()) and torch.equal(cuda,torch.cuda.get_rng_state());records=[]
 for scope,xy,logits in tasks:
  sr=s.spec_file(d,seed,epoch,scope);a=np.load(sr['path']);assert np.array_equal(xy,a['coordinates']);spec=a['R1_logits'];rr=torch.from_numpy(logits.astype(np.float64));ss=torch.from_numpy(spec.astype(np.float64));fusion=F.log_softmax(F.log_softmax(rr,1)+F.log_softmax(ss,1),1).numpy();p0=logits.argmax(1);p1=spec.argmax(1);p3=fusion.argmax(1);assert np.array_equal(p3,(rr+ss).numpy().argmax(1));path=folder/'predictions'/('%s_epoch_%03d.npz'%(scope,epoch));np.savez_compressed(path,coordinates=xy,relation_logits=logits,spectral_logits=spec,fusion_logits=fusion,relation_predictions=p0,spectral_predictions=p1,fusion_predictions=p3,agreement=p0==p1);records.append({'epoch':epoch,'scope':scope,'path':str(path),'sha256':s.sha(path),'immutable_spectral_prediction':sr})
 return records
def main(d,r,seed):
 s.verify();assert s.reused_relation(d,r,seed) is None;assert json.loads((s.OUT/('preflight_'+d+'.json')).read_text())['passed'];torch.set_num_threads(2);m,values,rt=s.setup(d,r,seed);source,val,target,ev=values;m.cuda();opt=torch.optim.SGD(m.parameters(),lr=.01);ref=[json.loads(x) for x in (s.reference(d,seed)/'history.jsonl').read_text().splitlines()]
 if d=='hyrank':
  xx,yy=np.nonzero(ev.dataset.label>=0);ok=(xx>=6)&(yy>=6)&(xx<ev.dataset.data.shape[0]-6)&(yy<ev.dataset.data.shape[1]-6);fullcoords=np.column_stack((xx[ok],yy[ok]));assert len(fullcoords)==10317;bs=(s.reference(d,seed)/'actual_batches.jsonl').open()
 else:fullcoords=ev.dataset.indices.copy();bs=None
 for ds in (target.dataset,ev.dataset):ds.label=np.zeros_like(ds.label);ds.labels=np.zeros_like(ds.labels)
 # Houston/Pavia historical target inference was offline, so use a private
 # iterator generator; HyRANK must retain its shared native evaluation iterator.
 inf=ev if d=='hyrank' else torch.utils.data.DataLoader(ev.dataset,batch_size=128,shuffle=False,num_workers=0,generator=torch.Generator().manual_seed(seed+185000))
 folder=s.OUT/d/('%s_%d'%(r,seed));folder.mkdir(parents=True,exist_ok=False);(folder/'checkpoints').mkdir();(folder/'predictions').mkdir();s.dump(folder/'config.json',{'dataset':d,'relation':r,'seed':seed,'plan':s.PLAN,'initial_hash':rt['hash'](m),'initial_original':rt['initial_original'],'parameters':sum(p.numel() for p in m.parameters()),'spectral_expert_reused_no_gradient':True,'protocol':{k:v for k,v in rt.items() if k in ('bands','classes','steps','indices','data_protocol')}});start=time.monotonic();records=[];cps=[]
 with (folder/'history.jsonl').open('x') as history:
  for epoch in range(1,201):
   m.train();bh=hashlib.sha256();rh=hashlib.sha256();ah=hashlib.sha256();ih=hashlib.sha256();steps=n=hit=0;ce=0.
   for sb,tb in zip(source,target):
    x,y=sb[:2];t,dummy=tb[:2]
    if len(x)!=len(t):continue
    assert not dummy.any()
    for v in (x,y,t):bh.update(v.numpy().tobytes())
    rh.update(torch.get_rng_state().numpy().tobytes());rh.update(torch.cuda.get_rng_state().numpy().tobytes())
    if d=='hyrank':
     current=rng(source.generator);ah.update(json.dumps(current,sort_keys=True).encode());si,ti=sb[2],tb[2];sx=source.dataset.indices[si.numpy()];tx=target.dataset.indices[ti.numpy()];ih.update(sx.tobytes());ih.update(tx.tobytes());expected=json.loads(next(bs));assert expected['epoch']==epoch and expected['step_in_epoch']==steps+1 and expected['source_dataset_indices']==si.tolist() and expected['target_dataset_indices']==ti.tolist() and expected['rng']==current
    x,y,t=x.cuda(),y.cuda(),t.cuda();opt.zero_grad(set_to_none=True);z=m(x,t)[0];loss=F.cross_entropy(z,y);assert torch.isfinite(loss);loss.backward();opt.step();steps+=1;n+=len(y);hit+=int((z.argmax(1)==y).sum());ce+=float(loss.detach())*len(y)
   rowref=ref[epoch-1];assert steps==rt['steps'] and bh.hexdigest()==rowref['batch_stream_sha256'];key='rng_stream_sha256' if d=='hyrank' else 'original_rng_stream_sha256';assert rh.hexdigest()==rowref[key],('RNG mismatch',d,r,seed,epoch)
   if d=='hyrank':assert ah.hexdigest()==rowref['all_rng_stream_sha256'] and ih.hexdigest()==rowref['actual_centers_sha256']
   counts={name:int(b.num_batches_tracked) for name,b in m.named_modules() if isinstance(b,(torch.nn.BatchNorm2d,torch.nn.BatchNorm3d))};assert all(v==epoch*steps*(1 if r.startswith('J') else 2) for v in counts.values());row={'dataset':d,'relation':r,'seed':seed,'epoch':epoch,'steps':steps,'source_CE':ce/n,'source_OA':hit/n*100,'batch_stream_sha256':bh.hexdigest(),'original_rng_stream_sha256':rh.hexdigest(),'all_rng_stream_sha256':ah.hexdigest() if d=='hyrank' else None,'BN_counts':counts,'elapsed_seconds':time.monotonic()-start}
   if d!='hyrank' and (epoch==1 or epoch%10==0):row['source_val']=source_val(m,val)
   if epoch%10==0:
    cp=folder/'checkpoints'/('epoch_%03d.pth'%epoch);torch.save({'model':m.state_dict(),'epoch':epoch,'config':json.loads((folder/'config.json').read_text())},cp);cps.append({'epoch':epoch,'path':str(cp),'sha256':s.sha(cp)});records+=freeze(m,inf,fullcoords,d,seed,epoch,rt,folder)
   history.write(json.dumps(row)+'\n');history.flush()
   if epoch==1 or epoch%10==0:print(json.dumps(row),flush=True)
 if bs:assert next(bs,None) is None;bs.close()
 s.verify();s.dump(folder/'TRAINING_PREDICTIONS_FROZEN.json',{'epochs':200,'source_CE_only':True,'all200_reference_data_and_RNG_match':True,'no_target_GT_scored':True,'checkpoints':cps,'predictions':records,'seconds':time.monotonic()-start});print('TRAIN_FREEZE_COMPLETE',d,r,seed,flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--dataset',choices=s.DATASETS,required=True);p.add_argument('--relation',choices=s.RELATIONS,required=True);p.add_argument('--seed',type=int,choices=s.SEEDS,required=True);a=p.parse_args();main(a.dataset,a.relation,a.seed)

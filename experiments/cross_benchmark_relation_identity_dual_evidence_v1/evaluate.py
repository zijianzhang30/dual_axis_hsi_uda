import argparse,json,time
import numpy as np,torch
import torch.nn.functional as F
import study as s
from expert import build

def main(dataset,seed):
 s.verify();torch.set_num_threads(2);m,values,r=s.setup(dataset,seed);e=build(r['bands'],r['classes'],seed).cuda().eval();m.cuda().eval();ds=values[-1].dataset;ds.label=np.zeros_like(ds.label);ds.labels=np.zeros_like(ds.labels);f=s.OUT/('%s_%d'%(dataset,seed));tm=json.loads((f/'SPECTRAL_TRAINING_COMPLETE.json').read_text());assert tm['epochs']==200;cs=np.load(f/'coordinates_frozen.npz');native,full=cs['native'],cs['full'];assert np.array_equal(native,ds.indices)
 centers=ds.data[native[:,0],native[:,1]].astype(np.float32);fullcenters=ds.data[full[:,0],full[:,1]].astype(np.float32);out=f/'predictions';out.mkdir(exist_ok=False);records=[];start=time.monotonic();loader=torch.utils.data.DataLoader(ds,batch_size=128,shuffle=False,num_workers=0,generator=torch.Generator().manual_seed(seed+185000));rr=s.baseline_records(dataset,seed)
 for rc,sc in zip(rr,tm['checkpoints']):
  assert rc['epoch']==sc['epoch'] and s.sha(sc['path'])==sc['sha256'];e.load_state_dict(torch.load(sc['path'],map_location='cpu',weights_only=True)['model']);ebefore=r['hash'](e)
  with torch.inference_mode():
   spec=np.concatenate([e(torch.from_numpy(centers[i:i+128]).cuda()).cpu().numpy() for i in range(0,len(centers),128)])
   if dataset=='hyrank':fspec=np.concatenate([e(torch.from_numpy(fullcenters[i:i+128]).cuda()).cpu().numpy() for i in range(0,len(fullcenters),128)])
  assert r['hash'](e)==ebefore
  if dataset=='pavia':
   assert s.sha(rc['checkpoint'])==rc['checkpoint_sha256'];m.load_state_dict(torch.load(rc['checkpoint'],map_location='cpu',weights_only=True)['model']);before=r['hash'](m)
   with torch.inference_mode():rel=np.concatenate([m(x.cuda(),x.cuda())[1].cpu().numpy() for x,dummy in loader])
   assert r['hash'](m)==before
   if rc['epoch']==200:assert np.array_equal(rel,np.load(s.baseline(dataset,seed)/'predictions_200.npz')['logits'])
   tasks=[('native',native,rel,spec)]
  else:
   rel=np.load(rc['prediction'])['logits'];frel=np.load(rc['full_prediction'])['logits'];tasks=[('native',native,rel,spec),('full',full,frel,fspec)]
  for scope,coords,rel,spec in tasks:
   a,b=torch.from_numpy(rel.astype(np.float64)),torch.from_numpy(spec.astype(np.float64));lf=F.log_softmax(F.log_softmax(a,1)+F.log_softmax(b,1),1).numpy();p0=rel.argmax(1);p1=spec.argmax(1);p3=lf.argmax(1);agree=p0==p1;assert np.array_equal(p3,(a+b).numpy().argmax(1));assert np.array_equal(p3[agree],p0[agree]);path=out/('%s_epoch_%03d.npz'%(scope,rc['epoch']));np.savez_compressed(path,coordinates=coords,R0_logits=rel,R1_logits=spec,R3_logits=lf,R0_predictions=p0,R1_predictions=p1,R3_predictions=p3,agreement=agree,disagreement=~agree);records.append({'epoch':rc['epoch'],'scope':scope,'path':str(path),'sha256':s.sha(path)})
  print('GRID_FROZEN',dataset,seed,rc['epoch'],flush=True)
 s.verify();s.dump(f/'ALL_PREDICTIONS_FROZEN.json',{'dataset':dataset,'seed':seed,'records':records,'all20grid_frozen':True,'R0_reused_immutable_formal_C':True,'Pavia_fixed200_R0_logits_bitwise_matches_baseline':dataset=='pavia','target_GT_not_scored':True,'seconds':time.monotonic()-start})
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--dataset',choices=s.DATASETS,required=True);p.add_argument('--seed',type=int,choices=s.SEEDS,required=True);a=p.parse_args();main(a.dataset,a.seed)

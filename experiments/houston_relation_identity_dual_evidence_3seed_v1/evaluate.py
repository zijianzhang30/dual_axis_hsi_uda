import argparse,json,time
import numpy as np,torch
import torch.nn.functional as F
import study as s
from expert import build

def main(seed):
 s.verify();torch.set_num_threads(2);c=s.environment();m,loaders,helper,n,audit=c.setup('houston','C',seed);e=build(seed);m.cuda().eval();e.cuda().eval();ds=loaders[-1].dataset;ds.label=np.zeros_like(ds.label);ds.labels=np.zeros_like(ds.labels);f=s.OUT/('seed_%d'%seed);mf=json.loads((f/'TRAINING_COMPLETE_FROZEN.json').read_text());assert mf['epochs']==200
 loader=torch.utils.data.DataLoader(ds,batch_size=128,shuffle=False,num_workers=0,generator=torch.Generator().manual_seed(seed+185000));out=f/'frozen_grid';out.mkdir(exist_ok=False);records=[];start=time.monotonic()
 for r in mf['checkpoints']:
  assert s.sha(r['path'])==r['sha256'];cp=torch.load(r['path'],map_location='cpu',weights_only=True);m.load_state_dict(cp['relation']);e.load_state_dict(cp['spectral']);before=(c.state_hash(m),c.state_hash(e));rel=[];spec=[]
  with torch.inference_mode():
   for x,dummy in loader:
    assert not dummy.any();x=x.cuda();rel.append(m(x,x)[1].cpu().numpy());spec.append(e(x[:,0,:,6,6]).cpu().numpy())
  assert before==(c.state_hash(m),c.state_hash(e));rel=np.concatenate(rel);spec=np.concatenate(spec);assert rel.shape==spec.shape==(52901,7)
  if r['epoch']==200:
   old=np.load(s.baseline(seed)/'predictions_200.npz');assert np.array_equal(rel,old['logits']) and np.array_equal(rel.argmax(1),old['predictions'])
  rr=torch.from_numpy(rel.astype(np.float64));ss=torch.from_numpy(spec.astype(np.float64));lr=F.log_softmax(rr,1);ls=F.log_softmax(ss,1);log2=(torch.logaddexp(lr,ls)-np.log(2)).numpy();log3=F.log_softmax(lr+ls,1).numpy();p0=rel.argmax(1);p1=spec.argmax(1);p2=log2.argmax(1);p3=log3.argmax(1);agree=p0==p1;assert np.array_equal(p3,(rr+ss).numpy().argmax(1));assert np.array_equal(p2[agree],p0[agree]) and np.array_equal(p3[agree],p0[agree]);path=out/('epoch_%03d.npz'%r['epoch']);np.savez_compressed(path,coordinates=ds.indices,R0_logits=rel,R1_logits=spec,R2_logits=log2,R3_logits=log3,R0_predictions=p0,R1_predictions=p1,R2_predictions=p2,R3_predictions=p3,agreement=agree,disagreement=~agree);records.append({'epoch':r['epoch'],'path':str(path),'sha256':s.sha(path)});print('GRID_FROZEN',seed,r['epoch'],flush=True)
 s.verify();s.dump(f/'ALL_GRID_PREDICTIONS_FROZEN.json',{'seed':seed,'records':records,'all20_grid_frozen':True,'no_target_GT_scored':True,'relation_fixed200_logits_bitwise_match_original_C':True,'BN_state_unchanged_in_eval':True,'fusion_implementation_same_as_prior':True,'seconds':time.monotonic()-start})
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--seed',type=int,choices=s.SEEDS,required=True);main(p.parse_args().seed)

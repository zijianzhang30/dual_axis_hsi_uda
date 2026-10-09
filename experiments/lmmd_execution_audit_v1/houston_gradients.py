"""Read-only fixed end-of-epoch checkpoint queries, NOT actual update gradients."""
import os,json,math,hashlib,importlib.util
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
B=Path('/nas1/zhangzj26/dual_axis_hsi_uda/results');OUT=B/'lmmd_houston_pavia_execution_audit_v1'
def dump(p,v):Path(p).write_text(json.dumps(v,indent=2)+'\n')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
spec=importlib.util.spec_from_file_location('audit_frozen_common',B/'cross_benchmark_lmmd_v1/snapshot/common.py');c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)
def stats(g,h,w):
 a=torch.cat([x.flatten() for x in g]);b=torch.cat([x.flatten() for x in h]);na=float(a.norm());nb=float(b.norm());dot=float(a@b)
 return {'CE_norm':na,'raw_LMMD_norm':nb,'weighted_LMMD_over_CE':w*nb/na if na>1e-12 else None,'cosine':dot/(na*nb) if na*nb>1e-12 else None,'CE_combined_descent_ratio':1+w*dot/(na*na) if na>1e-12 else None}
def main():
 OUT.mkdir(exist_ok=False);torch.set_num_threads(2);c.verify_files();rows=[];files={};phases=(1,10,50,100,200)
 dump(OUT/'frozen_plan.json',{'read_only':True,'no_optimizer_or_training':True,'seeds':[2100,2101,2102],'phases':list(phases),'batches':3,'batch_size':128,'queries':'first3 source/target paired augmented batches generated from frozen epoch1 loader seeds; identical inputs across checkpoint phases,private prescribed dropout seed','limits':'Frozen end-of-epoch model gradients,NOT actual saved training-step gradients; neither momentum nor SGD updates included. Not directly matched to Pavia existing exact-history query sample design.','Pavia_existing_diagnosis_reused':True,'no_new_variant':True})
 for seed in (2100,2101,2102):
  model,loaders,helper,classes,audit=c.setup('houston','C',seed);source,_,target,_=loaders
  target.dataset.labels=np.zeros_like(target.dataset.labels);target.dataset.label=np.zeros_like(target.dataset.label)
  batches=[]
  for (x,y),(t,dummy) in zip(source,target):
   assert not dummy.any();batches.append((x,y,t))
   if len(batches)==3:break
  hashes=[hashlib.sha256(b''.join(v.numpy().tobytes() for v in b)).hexdigest() for b in batches]
  model=model.cuda();lm=c.lmmd();cap=[];hook=model.norm.register_forward_hook(lambda mod,inp,out:cap.append(out[:,0]));params=list(model.named_parameters())
  groups={}
  for i,(name,p) in enumerate(params):
   key='CNN' if name.startswith(('conv3d_features.','conv2d_features.')) else 'tokenizer' if name.startswith('relation_scores.') else 'Transformer' if name.startswith('blocks.') else 'classifier' if name.startswith('nn1.') else 'other'
   groups.setdefault(key,[]).append(i)
  for epoch in phases:
   path=B/'cross_benchmark_lmmd_v1/houston'/('C_LMMD_%d/checkpoints/epoch_%03d.pth'%(seed,epoch));files[str(path)]=sha(path)
   state=torch.load(path,map_location='cpu',weights_only=True)['model'];model.load_state_dict(state);before=c.state_hash(model);w=.3*(2/(1+math.exp(-10*epoch/200))-1)
   for bi,(x,y,t) in enumerate(batches):
    model.load_state_dict(state);model.train();torch.manual_seed(seed+epoch*10000+bi+97000);torch.cuda.manual_seed_all(seed+epoch*10000+bi+97000);cap.clear()
    x,y,t=x.cuda(),y.cuda(),t.cuda();r=model(x,t);assert len(cap)==4
    ce=F.cross_entropy(r[0],y);loss=lm.lmmd(cap[0],cap[1],y,r[1].softmax(1),CLASS_NUM=7,BATCH_SIZE=128);assert torch.isfinite(ce+loss).all()
    gs=torch.autograd.grad(ce,[p for _,p in params],retain_graph=True,allow_unused=True);hs=torch.autograd.grad(loss,[p for _,p in params],allow_unused=True)
    gs=[torch.zeros_like(p) if g is None else g for (_,p),g in zip(params,gs)];hs=[torch.zeros_like(p) if h is None else h for (_,p),h in zip(params,hs)]
    row={'seed':seed,'epoch':epoch,'batch':bi,'input_sha256':hashes[bi],'LMMD_weight':w,'CE':float(ce.detach()),'LMMD':float(loss.detach()),'alignment_shape':list(cap[0].shape),'all_parameters':stats(gs,hs,w),'by_group':{name:stats([gs[i] for i in ids],[hs[i] for i in ids],w) for name,ids in groups.items()}}
    rows.append(row);dump(OUT/'houston_gradient_queries.json',rows)
   model.load_state_dict(state);assert c.state_hash(model)==before;assert sha(path)==files[str(path)]
  hook.remove();del model,loaders,batches,cap,gs,hs;torch.cuda.empty_cache();print('FROZEN_QUERIES_COMPLETE',seed,flush=True)
 assert all(sha(p)==h for p,h in files.items());dump(OUT/'queried_checkpoint_hashes.json',files)
 aggregate=[]
 for epoch in phases:
  rs=[r for r in rows if r['epoch']==epoch];values=[r['all_parameters'] for r in rs]
  aggregate.append({'epoch':epoch,'queries':len(rs),'mean_weighted_LMMD_over_CE':float(np.mean([v['weighted_LMMD_over_CE'] for v in values])),'median_weighted_LMMD_over_CE':float(np.median([v['weighted_LMMD_over_CE'] for v in values])),'mean_cosine':float(np.mean([v['cosine'] for v in values])),'negative_cosine_fraction':float(np.mean([v['cosine']<0 for v in values])),'CE_combined_first_order_uphill_fraction':float(np.mean([v['CE_combined_descent_ratio']<0 for v in values]))})
 dump(OUT/'houston_gradient_summary.json',aggregate);dump(OUT/'completed.json',{'read_only_queries_completed':True,'checkpoints_unchanged':True,'no_training_started':True});print(json.dumps(aggregate,indent=2),flush=True)
if __name__=='__main__':main()

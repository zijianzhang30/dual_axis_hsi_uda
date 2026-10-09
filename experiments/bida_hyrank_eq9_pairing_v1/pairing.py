"""Selector and cache have no target-label interface. No model/loss modifications."""
import copy, hashlib, json
import numpy as np
import torch

class Pairing:
 def __init__(self,seed,n,classes=12,enabled=True):
  self.probs=np.zeros((n,classes),np.float64);self.valid=np.zeros(n,bool)
  self.threshold=.5*np.log(classes);self.pick=np.random.RandomState(seed+90219)
  self.aug=np.random.RandomState(seed+90220);self.enabled=enabled;self.epoch=1
  self.rows=[];self.cache_forwards=0;self.reset_counts()
 def reset_counts(self):self.rows=[];self.cache_forwards=0
 def pools(self):
  p=self.probs;entropy=-(p*np.log(np.maximum(p,1e-300))).sum(1);pseudo=p.argmax(1)
  keep=self.valid&(entropy<=self.threshold)
  return [np.flatnonzero(keep&(pseudo==c)) for c in range(p.shape[1])],pseudo,entropy,keep
 def select(self,source_labels,candidate_ids):
  ids=np.asarray(candidate_ids).copy();matched=np.zeros(len(ids),bool)
  pools,pseudo,entropy,keep=self.pools()
  if self.enabled and self.epoch>1:
   for c in np.unique(source_labels):
    slots=np.flatnonzero(source_labels==c)
    if len(pools[c]):ids[slots]=self.pick.choice(pools[c],size=len(slots),replace=True);matched[slots]=True
  assert np.all(pseudo[ids[matched]]==source_labels[matched]) and np.all(keep[ids[matched]])
  return ids,matched,pseudo[ids].copy(),entropy[ids].copy()
 def observe(self,ids,logits):
  z=np.asarray(logits,dtype=np.float64);z=z-z.max(1,keepdims=True);p=np.exp(z);p/=p.sum(1,keepdims=True)
  unique,inverse=np.unique(ids,return_inverse=True);sums=np.zeros((len(unique),p.shape[1]),np.float64)
  np.add.at(sums,inverse,p);counts=np.bincount(inverse)
  self.probs[unique]=sums/counts[:,None];self.valid[unique]=True
 def augment(self,ds,index):
  # Native augmentation code, private RNG; never leak draws into baseline streams.
  state=np.random.get_state();np.random.set_state(self.aug.get_state())
  try:x,ignored,index=ds[int(index)];self.aug.set_state(np.random.get_state())
  finally:np.random.set_state(state)
  return x
 def pre(self,model,inputs):
  t=self.trace;assert t.phase in ('train','validation')
  if t.phase!='train':return
  y=t.source[1].numpy();candidate=t.target[1].numpy()
  selected,matched,pseudo,entropy=self.select(y,candidate)
  self.current=selected
  row={'source':y.copy(),'candidate':candidate.copy(),'selected':selected.copy(),'matched':matched.copy(),'pseudo_at_selection':pseudo,'entropy_at_selection':entropy}
  self.rows.append(row)
  if not np.any(selected!=candidate):return
  batch=t.target[0].clone()
  for slot in np.flatnonzero(selected!=candidate):batch[slot]=self.augment(self.dataset,selected[slot])
  inputs[1].copy_(batch.to(inputs[1].device))
  # Native tensor is shared with the subsequent EMA call.
 def post(self,model,inputs,outputs):
  if self.trace.phase=='train':self.observe(self.current,outputs[1].detach().cpu().numpy())
 @torch.inference_mode()
 def refresh(self,model,eval_ds,seed,reuse=None):
  modes={m:m.training for m in model.modules()};cpu=torch.get_rng_state().clone();cuda=torch.cuda.get_rng_state().clone()
  before={k:v.detach().clone() for k,v in model.state_dict().items()}
  model.eval()
  try:
   if reuse is not None:logits=reuse
   else:
    ds=copy.copy(eval_ds);ds.label=np.zeros_like(ds.label);ds.labels=np.zeros_like(ds.labels)
    gen=torch.Generator().manual_seed(seed+99281)
    loader=torch.utils.data.DataLoader(ds,batch_size=128,shuffle=False,num_workers=0,generator=gen)
    outputs=[]
    for x,dummy,index in loader:
     assert not dummy.any();x=x.cuda();outputs.append(model.forward(x,x)[1].cpu().numpy());self.cache_forwards+=1
    logits=np.concatenate(outputs)
   assert len(logits)==len(self.valid)
   self.observe(np.arange(len(logits)),logits)
   assert all(torch.equal(v,before[k]) for k,v in model.state_dict().items())
   assert torch.equal(cpu,torch.get_rng_state()) and torch.equal(cuda,torch.cuda.get_rng_state())
  finally:
   for m,mode in modes.items():m.training=mode
 def save_epoch(self,path):
  data={k:np.concatenate([r[k] for r in self.rows]) for k in self.rows[0]}
  np.savez_compressed(path,**data)
  pools,pseudo,entropy,keep=self.pools();matched=data['matched'];source=data['source']
  return {'pairing_coverage':float(matched.mean()),'matched_rows':int(matched.sum()),'total_source_rows':len(source),
  'source_CE_class_counts':np.bincount(source,minlength=12).tolist(),
  'matched_source_class_counts':np.bincount(source[matched],minlength=12).tolist(),
  'pseudo_class_counts':np.bincount(pseudo[self.valid],minlength=12).tolist(),
  'eligible_pseudo_class_counts':[len(p) for p in pools],
  'cache_valid_count':int(self.valid.sum()),'cache_accept_rate':float(keep.mean()),
  'extra_eval_only_cache_forwards':self.cache_forwards,'policy':'native fallback for empty candidates, no CE rows removed'}

import argparse,json,math,time,torch,numpy as np
import torch.nn.functional as F
import common as s

def norm(g):return sum(float(v.square().sum()) for v in g if v is not None)**.5
def compare(a,b):
 na,nb=norm(a),norm(b);dot=sum(float(x.flatten()@y.flatten()) for x,y in zip(a,b) if x is not None and y is not None);return {'norm_a':na,'norm_b':nb,'cosine':dot/(na*nb) if na*nb>1e-15 else None}
def main(seed):
 s.verify();torch.set_num_threads(2);c=s.module();m,ls,helper,classes,audit=c.setup('houston','C',seed);source,val,target,ev=ls
 for ds in (target.dataset,ev.dataset):ds.label=np.zeros_like(ds.label);ds.labels=np.zeros_like(ds.labels)
 ids=np.load(s.OUT/'holdout_target_indices.npy');xt,_=s.tensor_inputs(ev.dataset,ids);xs,ys=s.tensor_inputs(val.dataset,range(len(val.dataset)));m=m.cuda();lm=c.lmmd();folder=s.OUT/('seed_%d'%seed);folder.mkdir();(folder/'predictions').mkdir();data=np.load(s.OUT/('inputs_%d.npz'%seed));names=[n for n,p in m.named_parameters()];ps=list(m.parameters());cap=[];hook=m.norm.register_forward_hook(lambda mod,inp,out:cap.append(out[:,0]) if m.training else None);rows=[];start=time.monotonic()
 # SGD direction/one-step implementation preflight;no checkpoint optimized.
 p=torch.nn.Parameter(torch.tensor([.2,-.3],device='cuda'));g=torch.tensor([.4,-.7],device='cuda');before=p.detach().clone();opt=torch.optim.SGD([p],lr=.01);p.grad=g.clone();opt.step();assert torch.allclose(p,before-.01*g,atol=1e-8,rtol=1e-6);del p,opt
 for epoch in s.PHASES:
  cp=s.B/'tokenizer_mechanism_v2/houston'/('C_%d/checkpoints/epoch_%03d.pth'%(seed,epoch));state=torch.load(cp,map_location='cpu',weights_only=True)['model'];m.load_state_dict(state);checkpoint_hash=c.state_hash(m);weight=.3*(2/(1+math.exp(-10*epoch/200))-1)
  for bi in range(3):
   m.load_state_dict(state);m.train();torch.manual_seed(seed+epoch*10000+bi+97000);torch.cuda.manual_seed_all(seed+epoch*10000+bi+97000);cap.clear();x=torch.from_numpy(data['source'][bi]).cuda();y=torch.from_numpy(data['source_labels'][bi]).cuda();t=torch.from_numpy(data['target'][bi]).cuda();truth=torch.from_numpy(data['oracle_target_labels'][bi]).cuda();out=m(x,t);assert len(cap)==4
   ce=F.cross_entropy(out[0],y);prob=out[1].softmax(1);oracle=F.one_hot(truth,7).float();predloss=lm.lmmd(cap[0],cap[1],y,prob,CLASS_NUM=7,BATCH_SIZE=128);oracleloss=lm.lmmd(cap[0],cap[1],y,oracle,CLASS_NUM=7,BATCH_SIZE=128);assert torch.isfinite(ce+predloss+oracleloss)
   gc=torch.autograd.grad(ce,ps,retain_graph=True,allow_unused=True);gb=torch.autograd.grad(predloss,ps,retain_graph=True,allow_unused=True);go=torch.autograd.grad(oracleloss,ps,allow_unused=True);assert all(torch.isfinite(g).all() for gs in (gc,gb,go) for g in gs if g is not None)
   nb,no=norm(gb),norm(go);factor=nb/no if no>1e-15 else None;assert nb>0 and no>0;post={n:b.clone() for n,b in m.named_buffers()};buffers_before={n:state[n].cuda() for n,b in m.named_buffers()};modelstate={n:p.detach().clone() for n,p in m.named_parameters()};bncounts={n:int(b.num_batches_tracked) for n,b in m.named_modules() if isinstance(b,(torch.nn.BatchNorm2d,torch.nn.BatchNorm3d))};groups={}
   for prefix in ('conv3d_features.','conv2d_features.','relation_scores.','blocks.','nn1.'):
    selected=[i for i,n in enumerate(names) if n.startswith(prefix)];groups[prefix]={'CE_norm':norm([gc[i] for i in selected]),'pred_LMMD_norm':norm([gb[i] for i in selected]),'oracle_LMMD_norm':norm([go[i] for i in selected])}
   assert groups['nn1.']['pred_LMMD_norm']==groups['nn1.']['oracle_LMMD_norm']==0
   states=[('base',None,0.),('A',None,0.),('B',gb,weight),('C',go,weight),('C_normmatch',go,weight*factor)]
   for kind,ga,scale in states:
    m.load_state_dict(state)
    with torch.no_grad():
     for n,b in m.named_buffers():b.copy_(post[n])
    if kind!='base':
     opt=torch.optim.SGD(ps,lr=.01)
     for p,a,b in zip(ps,gc,ga if ga is not None else [None]*len(ps)):p.grad=(torch.zeros_like(p) if a is None else a.detach().clone())+(scale*b.detach() if b is not None else 0)
     opt.step();opt.zero_grad(set_to_none=True);del opt
    assert all(torch.equal(b,dict(m.named_buffers())[n]) for n,b in post.items());zt=s.predict(m,xt);zs=s.predict(m,xs);np.savez_compressed(folder/'predictions'/('%03d_%d_%s.npz'%(epoch,bi,kind)),target_logits=zt,source_logits=zs);assert all(torch.equal(b,dict(m.named_buffers())[n]) for n,b in post.items())
   active_source=set(y.cpu().tolist());predicted=set(prob.detach().argmax(1).cpu().tolist());actual=set(truth.cpu().tolist());row={'seed':seed,'epoch':epoch,'batch':bi,'CE':float(ce.detach()),'pred_LMMD':float(predloss.detach()),'oracle_LMMD':float(oracleloss.detach()),'weight':weight,'CE_gradient_norm':norm(gc),'pred_LMMD_gradient_norm':nb,'oracle_LMMD_gradient_norm':no,'weighted_pred_over_CE':weight*nb/max(norm(gc),1e-15),'weighted_oracle_over_CE':weight*no/max(norm(gc),1e-15),'pred_oracle_gradient':compare(gb,go),'CE_pred_gradient':compare(gc,gb),'CE_oracle_gradient':compare(gc,go),'oracle_normmatch_factor':factor,'prediction_batch_accuracy':float((prob.detach().argmax(1)==truth).float().mean()*100),'predicted_active_classes':sorted(active_source&predicted),'oracle_active_classes':sorted(active_source&actual),'update_target_class_counts':torch.bincount(truth,minlength=7).cpu().tolist(),'gradient_by_module':groups,'shared_BN_num_batches':bncounts,'single_main_train_forward':True,'eval_BN_buffers_identical_all_variants':True,'holdout_GT_in_gradient':False,'Oracle_training_method':False}
   rows.append(row);s.dump(folder/'gradient_queries.json',rows);m.load_state_dict(state);assert c.state_hash(m)==checkpoint_hash and all(p.grad is None for p in ps);print('ORACLE_QUERY',seed,epoch,bi,'done',flush=True);del gc,gb,go,out,prob,oracle,post,modelstate,buffers_before,zt,zs
  assert s.sha(cp)==json.load(open(s.OUT/'immutable_manifest.json'))[str(cp)]
 hook.remove();s.verify();s.dump(folder/'frozen_manifest.json',{'seed':seed,'queries':len(rows),'all_temporary_models_optimizers_discarded':True,'original_state_and_checkpoint_hashes_unchanged':True,'updated_weights_saved':False,'seconds':time.monotonic()-start,'prediction_files':{str(p):s.sha(p) for p in (folder/'predictions').glob('*.npz')}});print('SEED_COUNTERFACTUAL_COMPLETE',seed,flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--seed',type=int,choices=s.SEEDS,required=True);main(p.parse_args().seed)

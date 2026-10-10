import sys,json,math,time,hashlib,argparse,importlib.util
from pathlib import Path
import numpy as np,torch
import torch.nn.functional as F
ROOT=Path('/home/zhangzj26/dual_axis_hsi_uda')
sys.path.insert(0,str(ROOT/'experiments/houston_oracle_lmmd_counterfactual_v1'))
import common as s
OUT=s.B/'houston_lmmd_scale_stage_diagnostic_v1'
PHASES=(10,100,200)
def norm(gs):return sum(float(g.square().sum()) for g in gs if g is not None)**.5
def verify_new():
 s.verify()
 for p,h in json.loads((OUT/'immutable_manifest.json').read_text()).items():assert s.sha(p)==h,p

def main(seed):
 verify_new();torch.set_num_threads(2);c=s.module();m,ls,helper,classes,audit=c.setup('houston','C',seed);source,val,target,ev=ls
 for ds in (target.dataset,ev.dataset):ds.label=np.zeros_like(ds.label);ds.labels=np.zeros_like(ds.labels)
 xt,_=s.tensor_inputs(ev.dataset,np.load(s.OUT/'holdout_target_indices.npy'));xs,ys=s.tensor_inputs(val.dataset,range(len(val.dataset)))
 m=m.cuda();lm=c.lmmd();folder=OUT/('seed_%d'%seed);folder.mkdir();(folder/'predictions').mkdir();data=np.load(s.OUT/('inputs_%d.npz'%seed));ps=list(m.parameters());names=[n for n,p in m.named_parameters()];cap=[]
 hook=m.norm.register_forward_hook(lambda mod,inp,out:cap.append(out[:,0]) if m.training else None)
 old={(r['epoch'],r['batch']):r for r in json.loads((s.OUT/('seed_%d/gradient_queries.json'%seed)).read_text())};rows=[];start=time.monotonic()
 for epoch in PHASES:
  cp=s.B/'tokenizer_mechanism_v2/houston'/('C_%d/checkpoints/epoch_%03d.pth'%(seed,epoch));state=torch.load(cp,map_location='cpu',weights_only=True)['model'];m.load_state_dict(state);initial=c.state_hash(m);weight=.3*(2/(1+math.exp(-10*epoch/200))-1)
  for bi in range(3):
   m.load_state_dict(state);m.train();torch.manual_seed(seed+epoch*10000+bi+97000);torch.cuda.manual_seed_all(seed+epoch*10000+bi+97000);cap.clear()
   # Do not access archived oracle_target_labels. Target class values are erased above.
   x=torch.from_numpy(data['source'][bi]).cuda();y=torch.from_numpy(data['source_labels'][bi]).cuda();t=torch.from_numpy(data['target'][bi]).cuda();out=m(x,t);assert len(cap)==4
   ce=F.cross_entropy(out[0],y);align=lm.lmmd(cap[0],cap[1],y,out[1].softmax(1),CLASS_NUM=7,BATCH_SIZE=128);assert torch.isfinite(ce+align)
   gc=torch.autograd.grad(ce,ps,retain_graph=True,allow_unused=True);ga=torch.autograd.grad(align,ps,allow_unused=True);assert all(torch.isfinite(g).all() for gs in (gc,ga) for g in gs if g is not None)
   nc,na=norm(gc),norm(ga);assert nc>0 and na>0
   dot=sum(float(a.flatten()@b.flatten()) for a,b in zip(gc,ga) if a is not None and b is not None);cos=dot/(nc*na)
   oldrow=old[(epoch,bi)];errors={}
   for key,new in [('CE',float(ce.detach())),('pred_LMMD',float(align.detach())),('CE_gradient_norm',nc),('pred_LMMD_gradient_norm',na)]:
    errors[key]=abs(new-oldrow[key]);assert np.isclose(new,oldrow[key],atol=2e-6,rtol=2e-5),(key,new,oldrow[key])
   post={n:b.clone() for n,b in m.named_buffers()};bn={n:int(b.num_batches_tracked) for n,b in m.named_modules() if isinstance(b,(torch.nn.BatchNorm2d,torch.nn.BatchNorm3d))};assert bn==oldrow['shared_BN_num_batches']
   scales={'A':0.,'B':weight,'C_cap03':min(weight,.3*nc/na),'C_cap10':min(weight,nc/na),'D_fixed03':.3};updates={};saved={}
   for kind,scale in scales.items():
    m.load_state_dict(state)
    with torch.no_grad():
     for n,b in m.named_buffers():b.copy_(post[n])
    opt=torch.optim.SGD(ps,lr=.01);gg=[]
    for p,a,b in zip(ps,gc,ga):
     p.grad=(torch.zeros_like(p) if a is None else a.detach().clone())+(scale*b.detach() if b is not None else 0);gg.append(p.grad)
    totalnorm=norm(gg);opt.step();opt.zero_grad(set_to_none=True);del opt,gg
    actual=sum(float((p.detach()-state[n].to(p.device)).square().sum()) for n,p in m.named_parameters())**.5
    assert np.isclose(actual,.01*totalnorm,rtol=1e-3,atol=1e-7),(actual,totalnorm)
    updates[kind]={'lambda_effective':scale,'weighted_LMMD_over_CE':scale*na/nc,'CE_gradient_norm':nc,'LMMD_gradient_norm':na,'combined_gradient_norm':totalnorm,'actual_parameter_update_norm':actual,'CE_update_norm':.01*nc,'alignment_update_norm':.01*scale*na,'native_alignment_retained_fraction':scale/weight,'cap_active':scale<weight-1e-12,'CE_LMMD_gradient_cosine':cos}
    assert all(torch.equal(b,dict(m.named_buffers())[n]) for n,b in post.items())
    if kind in ('A','B'):
     oldpath=s.OUT/('seed_%d/predictions/%03d_%d_%s.npz'%(seed,epoch,bi,kind))
     # Verify historical reuse by reproducing both arms at every state/batch.
     zt=s.predict(m,xt);zs=s.predict(m,xs);arch=np.load(oldpath)
     err=max(float(np.max(np.abs(zt-arch['target_logits']))),float(np.max(np.abs(zs-arch['source_logits']))));assert err<1e-6,err
     saved[kind]={'path':str(oldpath),'sha256':s.sha(oldpath),'historical_reproduction_max_error':err}
    else:
     zt=s.predict(m,xt);zs=s.predict(m,xs);p=folder/'predictions'/('%03d_%d_%s.npz'%(epoch,bi,kind));np.savez_compressed(p,target_logits=zt,source_logits=zs);saved[kind]={'path':str(p),'sha256':s.sha(p)}
    assert all(torch.equal(b,dict(m.named_buffers())[n]) for n,b in post.items())
   basepath=s.OUT/('seed_%d/predictions/%03d_%d_base.npz'%(seed,epoch,bi));saved['base']={'path':str(basepath),'sha256':s.sha(basepath)}
   groups={}
   for prefix in ('conv3d_features.','conv2d_features.','relation_scores.','blocks.','nn1.'):
    ii=[i for i,n in enumerate(names) if n.startswith(prefix)];groups[prefix]={'CE_norm':norm([gc[i] for i in ii]),'LMMD_norm':norm([ga[i] for i in ii])}
   assert groups['nn1.']['LMMD_norm']==0
   rows.append({'seed':seed,'epoch':epoch,'batch':bi,'CE':float(ce.detach()),'LMMD':float(align.detach()),'gradient_cosine':cos,'CE_gradient_norm':nc,'LMMD_gradient_norm':na,'native_lambda':weight,'arms':updates,'predictions':saved,'historical_gradient_reproduction_error':errors,'shared_postforward_BN_counts':bn,'gradient_by_module':groups,'target_GT_accessed_for_update':False,'single_main_train_forward':True})
   s.dump(folder/'queries.json',rows);m.load_state_dict(state);assert c.state_hash(m)==initial and all(p.grad is None for p in ps)
   print('SCALE_STAGE_QUERY',seed,epoch,bi,'done',flush=True);del gc,ga,out,post,zt,zs
  assert s.sha(cp)==json.loads((s.OUT/'immutable_manifest.json').read_text())[str(cp)]
 hook.remove();verify_new();s.dump(folder/'frozen_manifest.json',{'seed':seed,'queries':len(rows),'seconds':time.monotonic()-start,'checkpoint_unchanged':True,'temporary_models_optimizers_discarded':True,'updated_model_saved':False,'target_GT_accessed_for_gradient':False,'source_val_labels':ys.numpy().tolist(),'files':{v['path']:v['sha256'] for r in rows for v in r['predictions'].values()}});print('SCALE_STAGE_SEED_COMPLETE',seed,flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--seed',type=int,choices=s.SEEDS,required=True);main(p.parse_args().seed)

"""Local hypothetical SGD updates only. No trained model or buffer is persisted."""
import os,json,math,hashlib,copy,importlib.util,time
from pathlib import Path
import numpy as np,torch
import torch.nn.functional as F
B=Path('/nas1/zhangzj26/dual_axis_hsi_uda/results');OUT=B/'relation_class_risk_diagnostic_v1';SEEDS=(2100,2101,2102);PHASES=(1,10,50,100,200)
def dump(p,v):Path(p).write_text(json.dumps(v,indent=2)+'\n')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def module(p):
 spec=importlib.util.spec_from_file_location('risk_frozen_common',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
c=module(B/'cross_benchmark_lmmd_v1/snapshot/common.py')
def tensors(ds,ids):
 ds=copy.copy(ds)
 for key in ('data_aug','flip_augmentation','radiation_augmentation','mixture_augmentation'):setattr(ds,key,False)
 samples=[ds[int(i)] for i in ids];return torch.stack([x for x,y in samples]),torch.stack([y for x,y in samples])
def perturb(x,strength,condition):
 # Fixed smooth per-band changes; patch-shared,positive gains. No post-affine renormalization.
 n=x.shape[2];w=torch.linspace(0,2*math.pi,n,device=x.device).view(1,1,n,1,1);phase=condition*math.pi/2
 return x*(1+strength*torch.sin(w+phase))+strength*.1*torch.cos(w+phase)
@torch.no_grad()
def predict(m,x):
 m.eval();before={n:b.clone() for n,b in m.named_buffers()};z=[]
 for t in x.split(128):z.append(m(t.cuda(),t.cuda())[1].cpu())
 assert all(torch.equal(v,dict(m.named_buffers())[n]) for n,v in before.items());return torch.cat(z)
def metrics(z,y):
 ce=F.cross_entropy(z,y,reduction='none');true=z.gather(1,y[:,None]).squeeze(1);other=z.clone();other.scatter_(1,y[:,None],float('-inf'));margin=true-other.max(1).values
 return {'class_CE':[float(ce[y==k].mean()) for k in range(7)],'class_margin':[float(margin[y==k].mean()) for k in range(7)],'OA':float((z.argmax(1)==y).float().mean()*100)}
def score_delta(a,b):
 d=np.asarray(b['class_CE'])-a['class_CE'];dm=np.asarray(a['class_margin'])-b['class_margin'];return {'max_class_CE_increase':float(d.max()),'mean_positive_class_CE_increase':float(np.maximum(d,0).mean()),'mean_class_CE_increase':float(d.mean()),'max_class_margin_damage':float(dm.max()),'class_CE_delta':d.tolist()}
def main():
 OUT.mkdir(exist_ok=False);(OUT/'predictions').mkdir();torch.set_num_threads(2);c.verify_files();files={};rows=[];start=time.monotonic()
 plan={'no_training':True,'no_projection':True,'dataset':'Houston','seeds':list(SEEDS),'phases':list(PHASES),'batches_per_phase':3,'input_batches':'same frozen epoch1 batches/private dropout seeds as existing LMMD execution audit','checks':'source existing validation127; calibration16/class from original source training pool; no additional labels','perturbation_candidates':[.025,.05],'perturbation':'post-normalization patch-shared smooth per-band gain 1+s*sin; shift .1*s*cos,4 fixed phases,no renormalization','source_selection':'largest candidate with calibration accuracy drop<=5pp,mean CE rise<=.25,relation relative-error p95<=.005,W max-error p95<=.001; if none STOP','risk_score_primary':'max class/condition CE increase,0 threshold predefined; no target threshold fitting','controls':['clean source max class CE increase','weighted alignment update norm'],'target_subset':'2048 uniform private-index sample from existing native candidate mask,not class-balanced','GT_barrier':'all temporary-state logits and source risk scores saved before any target GT scoring','update':'one main train forward at theta; CE and weighted LMMD gradients same state; actual original SGD .01/no momentum/no decay; eval both clones with identical post-forward BN buffers','limits':'local frozen-checkpoint finite-step diagnostic,not saved historical update replay or target guarantee; source validation was already monitored in historical training'}
 dump(OUT/'frozen_plan.json',plan);dump(OUT/'status.json',{'status':'source_calibration_then_local_queries'})
 old=json.loads((B/'lmmd_houston_pavia_execution_audit_v1/houston_gradient_queries.json').read_text())
 for seed in SEEDS:
  m,ls,helper,classes,audit=c.setup('houston','C',seed);source,val,target,ev=ls
  for d in (target.dataset,ev.dataset):d.labels=np.zeros_like(d.labels);d.label=np.zeros_like(d.label)
  batches=[]
  for (x,y),(t,dummy) in zip(source,target):
   assert not dummy.any();batches.append((x,y,t))
   if len(batches)==3:break
  xcheck,ycheck=tensors(val.dataset,range(len(val.dataset)));assert len(xcheck)==127
  ids=np.random.default_rng(99123).choice(len(ev.dataset),2048,replace=False);xt,_=tensors(ev.dataset,ids)
  np.save(OUT/('target_ids_%d.npy'%seed),ids);np.save(OUT/('source_check_indices_%d.npy'%seed),val.dataset.indices)
  if seed==2100:
   sy=np.asarray(source.dataset.labels);calids=np.concatenate([np.flatnonzero(sy==k)[:16] for k in range(7)]);xc,yc=tensors(source.dataset,calids);assert len(xc)==112
   path=B/'tokenizer_mechanism_v2/houston/C_2100/checkpoints/epoch_200.pth';files[str(path)]=sha(path);m.load_state_dict(torch.load(path,map_location='cpu',weights_only=True)['model']);m=m.cuda().eval();base=predict(m,xc);bm=metrics(base,yc);cal=[]
   for strength in plan['perturbation_candidates']:
    for a in range(4):
     xp=perturb(xc,strength,a);pm=metrics(predict(m,xp),yc)
     with torch.no_grad():
      r=m.relation_scores.relations(xc.cuda());rp=m.relation_scores.relations(xp.cuda());rel=(rp-r).flatten(1).norm(dim=1)/(r.flatten(1).norm(dim=1)+1e-12);werr=(m.relation_scores(xp.cuda())-m.relation_scores(xc.cuda())).flatten(1).abs().max(1).values
     cal.append({'strength':strength,'condition':a,'source_OA_drop':bm['OA']-pm['OA'],'mean_class_CE_rise':float(np.mean(pm['class_CE'])-np.mean(bm['class_CE'])),'relation_relative_error_p95':float(torch.quantile(rel,.95)),'W_max_error_p95':float(torch.quantile(werr,.95))})
   acceptable=[s for s in plan['perturbation_candidates'] if all(r['source_OA_drop']<=5 and r['mean_class_CE_rise']<=.25 and r['relation_relative_error_p95']<=.005 and r['W_max_error_p95']<=.001 for r in cal if r['strength']==s)]
   dump(OUT/'source_only_calibration.json',{'records':cal,'source_indices':source.dataset.indices[calids].tolist(),'source_check_labels_count':np.bincount(ycheck.numpy(),minlength=7).tolist(),'target_GT_used':False,'accepted':acceptable})
   if not acceptable:dump(OUT/'status.json',{'status':'NoGo_source_calibration_failed_no_updates'});return
   strength=max(acceptable);dump(OUT/'locked_rule.json',{'strength':strength,'primary_score':'max class/condition CE delta','warning_threshold':0.,'target_GT_used':False})
  m=m.cuda();lm=c.lmmd();cap=[];hook=m.norm.register_forward_hook(lambda mod,inp,out:cap.append(out[:,0]) if m.training else None);params=list(m.parameters())
  for epoch in PHASES:
   path=B/'cross_benchmark_lmmd_v1/houston'/('C_LMMD_%d/checkpoints/epoch_%03d.pth'%(seed,epoch));files[str(path)]=sha(path);state=torch.load(path,map_location='cpu',weights_only=True)['model'];m.load_state_dict(state);checkpoint_hash=c.state_hash(m);weight=.3*(2/(1+math.exp(-10*epoch/200))-1)
   for bi,(x,y,t) in enumerate(batches):
    inputhash=hashlib.sha256(b''.join(v.numpy().tobytes() for v in (x,y,t))).hexdigest();reference=next(r for r in old if r['seed']==seed and r['epoch']==epoch and r['batch']==bi);assert inputhash==reference['input_sha256']
    m.load_state_dict(state);m.train();torch.manual_seed(seed+epoch*10000+bi+97000);torch.cuda.manual_seed_all(seed+epoch*10000+bi+97000);cap.clear();out=m(x.cuda(),t.cuda());assert len(cap)==4
    ce=F.cross_entropy(out[0],y.cuda());align=lm.lmmd(cap[0],cap[1],y.cuda(),out[1].softmax(1),CLASS_NUM=7,BATCH_SIZE=128);g=torch.autograd.grad(ce,params,retain_graph=True,allow_unused=True);h=torch.autograd.grad(align,params,allow_unused=True);assert torch.isfinite(ce+align)
    post={n:b.clone() for n,b in m.named_buffers()};variants={}
    for kind in ('CE','candidate'):
     with torch.no_grad():
      for (name,p),gg,hh in zip(m.named_parameters(),g,h):p.copy_(state[name].cuda()-.01*((torch.zeros_like(p) if gg is None else gg)+(weight*(torch.zeros_like(p) if hh is None else hh) if kind=='candidate' else 0)))
      for n,b in m.named_buffers():b.copy_(post[n])
     assert all(torch.equal(dict(m.named_buffers())[n],b) for n,b in post.items());zs=[predict(m,xcheck)]+[predict(m,perturb(xcheck,strength,a)) for a in range(4)];variants[kind]=[metrics(z,ycheck) for z in zs];zt=predict(m,xt)
     np.savez_compressed(OUT/'predictions'/('%d_%03d_%d_%s.npz'%(seed,epoch,bi,kind)),logits=zt.numpy())
    clean=score_delta(variants['CE'][0],variants['candidate'][0]);aug=[score_delta(a,b) for a,b in zip(variants['CE'][1:],variants['candidate'][1:])];nr=(sum(float(v.square().sum()) for v in h if v is not None)**.5)*weight*.01
    row={'seed':seed,'epoch':epoch,'batch':bi,'input_sha256':inputhash,'weight':weight,'CE':float(ce.detach()),'LMMD':float(align.detach()),'alignment_update_norm':nr,'clean':clean,'perturbed':aug,'perturbed_primary_score':max(r['max_class_CE_increase'] for r in aug),'clean_primary_score':clean['max_class_CE_increase'],'source_metrics':variants}
    rows.append(row);dump(OUT/'queries.json',rows);m.load_state_dict(state);assert c.state_hash(m)==checkpoint_hash;print('QUERY',seed,epoch,bi,'scores',row['clean_primary_score'],row['perturbed_primary_score'],flush=True);del g,h,out,post,variants,zt,zs
   assert sha(path)==files[str(path)]
  hook.remove();del m,ls,batches,cap;torch.cuda.empty_cache()
 assert all(sha(p)==h for p,h in files.items());dump(OUT/'checkpoint_hashes.json',files);dump(OUT/'GT_SCORING_BARRIER.json',{'all_predictions_frozen':True,'all_source_scores_locked':True,'queries':len(rows),'prediction_sha256':{str(p):sha(p) for p in (OUT/'predictions').glob('*.npz')},'no_training_or_projection':True});dump(OUT/'status.json',{'status':'queries_complete_pending_offline_target_scoring','seconds':time.monotonic()-start})
if __name__=='__main__':main()

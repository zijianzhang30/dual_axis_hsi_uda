import argparse,json,time,warnings
import numpy as np,torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import roc_auc_score
from common import OUT,BASE,SEEDS,frozen,dump,sha,hasharray
from metrics import separability

class Patches(torch.utils.data.Dataset):
 def __init__(self,domain):self.cube=np.load(OUT/'data'/f'{domain}_cube.npy',mmap_mode='r');self.coords=np.load(OUT/'data'/f'{domain}_coords.npy')
 def __len__(self):return len(self.coords)
 def __getitem__(self,index):
  x,y=self.coords[index];return torch.from_numpy(np.array(self.cube[x-6:x+7,y-6:y+7],np.float32).transpose(2,0,1)).unsqueeze(0)

@torch.inference_mode()
def extract(model,domain,folder):
 dataset=Patches(domain);loader=torch.utils.data.DataLoader(dataset,batch_size=128,num_workers=2,shuffle=False);latest={};layers={'conv3d':model.conv3d_features[0],'bn3d':model.conv3d_features[1],'F3D':model.conv3d_features[2],'conv2d':model.conv2d_features[0],'bn2d':model.conv2d_features[1],'V':model.conv2d_features[2]};hooks=[];moments={k:[] for k in layers};norms={k:[] for k in layers}
 def hook(key):
  def f(m,args,out):latest[key]=out.detach()
  return f
 for key,module in layers.items():hooks.append(module.register_forward_hook(hook(key)))
 hooks.append(model.relation_scores.register_forward_hook(hook('W')));orig=model._forward_semantic_tokens
 def token(v):
  t=orig(v);latest['T']=t.detach();return t
 model._forward_semantic_tokens=token;arrays={};pos=0;shapes={};max_w_error=max_t_error=max_logit_error=0.
 oldw=np.load(BASE/'houston'/f'C_{model._diag_seed}'/'weights_200.npy',mmap_mode='r') if domain=='target' else None;oldt=np.load(BASE/'houston'/f'C_{model._diag_seed}'/'tokens_200.npy',mmap_mode='r') if domain=='target' else None;oldlogits=np.load(BASE/'houston'/f'C_{model._diag_seed}'/'predictions_200.npz')['logits'] if domain=='target' else None
 try:
  for x in loader:
   x=x.cuda();out=model(x,x,return_feat_prob=True);latest['Z']=out[3].detach();batch={}
   for key in layers:
    value=latest[key];shapes[key]=list(value.shape[1:]);axes=(0,*range(2,value.ndim));mu=value.mean(axes).cpu().numpy().astype(float);var=value.var(axes,correction=0).cpu().numpy().astype(float);moments[key].append((mu,var,int(value.numel()/value.shape[1])));norms[key].append(value.flatten(1).norm(dim=1).cpu().numpy());center=value[...,6,6].reshape(len(x),-1);batch[key]=center.cpu().numpy()
   assert latest['F3D'].shape[1:]==(8,48,13,13) and latest['V'].shape[1:]==(64,13,13)
   batch['F3D_mean']=latest['F3D'].mean((-2,-1)).reshape(len(x),-1).cpu().numpy();batch['V_mean']=latest['V'].mean((-2,-1)).cpu().numpy();batch['T']=latest['T'].flatten(1).cpu().numpy();batch['Z']=latest['Z'].cpu().numpy();batch['logits']=out[1].cpu().numpy();batch['W']=latest['W'].cpu().numpy()
   v=F.normalize(latest['V'].flatten(2),dim=1,eps=1e-12);batch['neighbor_cosine']=(v*v[:,:,84:85]).sum(1).cpu().numpy()
   for key,a in batch.items():
    if key not in arrays:arrays[key]=np.lib.format.open_memmap(folder/f'{domain}_{key}.npy',mode='w+',dtype=np.float32,shape=(len(dataset),*a.shape[1:]))
    arrays[key][pos:pos+len(x)]=a
   if domain=='target':
    max_w_error=max(max_w_error,float(np.max(np.abs(batch['W']-oldw[pos:pos+len(x)]))));max_t_error=max(max_t_error,float(np.max(np.abs(batch['T']-oldt[pos:pos+len(x)].reshape(len(x),-1)))));max_logit_error=max(max_logit_error,float(np.max(np.abs(batch['logits']-oldlogits[pos:pos+len(x)]))))
   pos+=len(x)
 finally:
  model._forward_semantic_tokens=orig
  for h in hooks:h.remove()
 assert max(max_w_error,max_t_error,max_logit_error)<1e-6
 for a in arrays.values():a.flush()
 stats={}
 for key,values in moments.items():
  count=sum(a[2] for a in values);mu=sum(a[0]*a[2] for a in values)/count;var=sum((a[1]+a[0]**2)*a[2] for a in values)/count-mu**2;n=np.concatenate(norms[key]);dimension=int(np.prod(shapes[key]));stats[key]={'shape':shapes[key],'full_map_norm_mean':float(n.mean()),'full_map_norm_std':float(n.std()),'norm_per_sqrt_dimension_mean':float(n.mean()/np.sqrt(dimension)),'channel_mean':mu.tolist(),'channel_variance':var.tolist(),'channel_variance_mean':float(var.mean()),'channel_variance_quantiles':np.quantile(var,[0,.1,.5,.9,1]).tolist()}
 return stats,{'native_W_max_error':max_w_error,'native_T_max_error':max_t_error,'native_logits_max_error':max_logit_error}

def domain_stats(s,t):
 s=np.asarray(s,np.float64);t=np.asarray(t,np.float64);mu=s.mean(0);mt=t.mean(0);cs=np.cov(s,rowvar=False);ct=np.cov(t,rowvar=False);dim=s.shape[1];gap=np.linalg.norm(mu-mt);scale=np.sqrt(.5*(np.trace(cs)+np.trace(ct)));rng=np.random.default_rng(98000);a=s[rng.choice(len(s),512,replace=False)];b=t[rng.choice(len(t),512,replace=False)];x=torch.from_numpy(np.concatenate((a,b)));square=(x*x).sum(1);dist=(square[:,None]+square[None,:]-2*x@x.T).clamp_min(0);bandwidth=float(dist[torch.triu_indices(1024,1024,offset=1).unbind()].median());bandwidth=max(bandwidth,1e-12);kernel=sum(torch.exp(-dist/(bandwidth*c)) for c in [.5,1,2])/3;k=kernel.numpy();mmd=((k[:512,:512].sum()-np.trace(k[:512,:512]))+(k[512:,512:].sum()-np.trace(k[512:,512:])))/(512*511)-2*k[:512,512:].mean()
 return {'descriptor_dimension':dim,'mean_shift_L2':float(gap),'relative_mean_shift':float(gap/(scale+1e-30)),'CORAL':float(np.linalg.norm(cs-ct,'fro')**2/(4*dim*dim)),'relative_covariance_gap':float(np.linalg.norm(cs-ct,'fro')/(np.linalg.norm(cs,'fro')+np.linalg.norm(ct,'fro')+1e-30)),'MMD_unbiased_multiscale_RBF':float(mmd),'MMD_median_squared_distance':bandwidth}

def neighbor_stats(folder):
 coords=np.load(OUT/'data/target_coords.npy');gt=np.load(OUT/'data/target_gt.npy');shape=json.loads((OUT/'fixed_inputs.json').read_text())['domains']['target']['shape'];labels=np.stack([gt[x-6:x+7,y-6:y+7].reshape(-1) for x,y in coords]);center=labels[:,84];known=labels>=0;same=labels==center[:,None];known[:,84]=False;same[:,84]=False;diff=known&~same;c=np.load(folder/'target_neighbor_cosine.npy',mmap_mode='r');w=np.load(folder/'target_W.npy',mmap_mode='r');central=labels.reshape(-1,13,13)[:,5:8,5:8].reshape(-1,9);boundary=((central>=0)&(central!=center[:,None])).any(1);interior=(central==center[:,None]).all(1);orig=coords-6;safe=(orig[:,0]>=6)&(orig[:,0]<shape[0]-6)&(orig[:,1]>=6)&(orig[:,1]<shape[1]-6);scores=np.array(w);scores[:,:,84]=-np.inf;top=np.argsort(scores,axis=-1)[:,:,-10:];topknown=np.take_along_axis(known[:,None,:],top,axis=-1);topsame=np.take_along_axis(same[:,None,:],top,axis=-1);topcos=np.take_along_axis(c[:,None,:],top,axis=-1);sc=known.sum(1);ss=same.sum(1);ds=diff.sum(1);smean=np.divide((c*same).sum(1),ss,out=np.full(len(c),np.nan),where=ss>0);dmean=np.divide((c*diff).sum(1),ds,out=np.full(len(c),np.nan),where=ds>0);result={}
 def desc(a):
  a=np.asarray(a,float);a=a[np.isfinite(a)]
  return {'n':len(a),'mean':float(a.mean()) if len(a) else None,'variance':float(a.var()) if len(a) else None}
 for name,mask in [('all_nonreflected',safe),('boundary3x3',safe&boundary),('interior3x3_fully_labeled',safe&interior)]:
  sv=np.asarray(c[mask][same[mask]]);dv=np.asarray(c[mask][diff[mask]]);rng=np.random.default_rng(99000);sv_sample=sv[rng.choice(len(sv),min(50000,len(sv)),replace=False)] if len(sv) else [];dv_sample=dv[rng.choice(len(dv),min(50000,len(dv)),replace=False)] if len(dv) else [];auc=float(roc_auc_score(np.r_[np.ones(len(sv_sample)),np.zeros(len(dv_sample))],np.r_[sv_sample,dv_sample])) if len(sv_sample) and len(dv_sample) else None;tk=topknown[mask];ts=topsame[mask];tc=topcos[mask];paired=mask&np.isfinite(smean)&np.isfinite(dmean);perclass={}
  for cls in range(7):
   cm=mask&(center==cls);both=cm&np.isfinite(smean)&np.isfinite(dmean);kt=topknown[cm];st=topsame[cm]
   perclass[str(cls+1)]={'patches':int(cm.sum()),'paired_patch_cosine_gap':desc(smean[both]-dmean[both]),'top10_known_same_class_ratio':float((st&kt).sum()/kt.sum()) if kt.sum() else None}
  result[name]={'patches':int(mask.sum()),'same_cosine':desc(sv),'different_cosine':desc(dv),'mean_cosine_gap':float(sv.mean()-dv.mean()) if len(sv) and len(dv) else None,'paired_patch_cosine_gap':desc(smean[paired]-dmean[paired]),'balanced_sample_cosine_AUROC':auc,'top10_known_same_class_ratio':float((ts&tk).sum()/tk.sum()) if tk.sum() else None,'all_known_neighbors_same_class_ratio':float(same[mask].sum()/known[mask].sum()) if known[mask].sum() else None,'top10_known_coverage':float(tk.mean()) if len(tk) else None,'top10_cosine_all':desc(tc),'top10_cosine_known_same':desc(tc[tk&ts]),'top10_cosine_known_different':desc(tc[tk&~ts]),'top10_same_minus_different_cosine':float(tc[tk&ts].mean()-tc[tk&~ts].mean()) if (tk&ts).sum() and (tk&~ts).sum() else None,'classbalanced_paired_patch_gap_mean':float(np.mean([v['paired_patch_cosine_gap']['mean'] for v in perclass.values() if v['paired_patch_cosine_gap']['n']])) if any(v['paired_patch_cosine_gap']['n'] for v in perclass.values()) else None,'per_class':perclass}
 return result

def main():
 p=argparse.ArgumentParser();p.add_argument('--seed',type=int,choices=SEEDS,required=True);seed=p.parse_args().seed;torch.set_num_threads(2);env=frozen.environment();env[4](seed);folder=OUT/f'seed_{seed}';folder.mkdir(exist_ok=False);cp=BASE/'houston'/f'C_{seed}/checkpoints/epoch_200.pth';before_cp=sha(cp);model,_=frozen.build(seed,'C','houston');model.load_state_dict(torch.load(cp,map_location='cpu',weights_only=True)['model']);model.cuda().eval();model._diag_seed=seed;before=frozen.state_hash(model);stats={};audit={}
 for domain in ['source','target']:stats[domain],audit[domain]=extract(model,domain,folder);print(seed,domain,'frozen features extracted',flush=True)
 assert frozen.state_hash(model)==before and sha(cp)==before_cp;del model;torch.cuda.empty_cache();sl=np.load(OUT/'data/source_labels.npy');tl=np.load(OUT/'data/target_labels.npy');train=np.load(OUT/'data/source_probe_train_mask.npy');stages={};layers={}
 for stage in ['F3D','V','T','Z','F3D_mean','V_mean']:
  source=np.load(folder/f'source_{stage}.npy',mmap_mode='r');target=np.load(folder/f'target_{stage}.npy',mmap_mode='r');scaler=StandardScaler();a=scaler.fit_transform(source[train]);probe=LogisticRegression(C=1.,solver='lbfgs',max_iter=2000,tol=1e-6,random_state=98000)
  with warnings.catch_warnings(record=True) as warnings_seen:warnings.simplefilter('always');probe.fit(a,sl[train])
  converged=not any(issubclass(w.category,ConvergenceWarning) for w in warnings_seen);pred_source=probe.predict(scaler.transform(source[~train]));pred_target=probe.predict(scaler.transform(target));hmetrics=env[6].metrics(sl[~train],pred_source,probe.decision_function(scaler.transform(source[~train])),7);tmetrics=env[6].metrics(tl,pred_target,probe.decision_function(scaler.transform(target)),7);np.savez_compressed(folder/f'{stage}_source_probe.npz',coef=probe.coef_,intercept=probe.intercept_,scaler_mean=scaler.mean_,scaler_scale=scaler.scale_,source_heldout_predictions=pred_source,target_predictions=pred_target)
  geometry=separability(source,sl,target,tl);geometry_train=separability(source[train],sl[train],target,tl);stages[stage]={'probe_source_heldout':hmetrics,'probe_target':tmetrics,'probe_converged':converged,'probe_iterations':probe.n_iter_.tolist(),'geometry':geometry,'source_probe_train_centroid_target_geometry':geometry_train['source_to_target'],'domain_stats':domain_stats(source,target)};print(seed,stage,'probe OA',tmetrics['oa_percent'],'AA',tmetrics['aa_percent'],flush=True)
 for key in ['conv3d','bn3d','F3D','conv2d','bn2d','V']:
  s=np.load(folder/f'source_{key}.npy',mmap_mode='r');t=np.load(folder/f'target_{key}.npy',mmap_mode='r');layers[key]={'source_fullmap_stats':stats['source'][key],'target_fullmap_stats':stats['target'][key],'center_descriptor_domain_stats':domain_stats(s,t),'center_descriptor_geometry':separability(s,sl,t,tl)}
 neighbors=neighbor_stats(folder);result={'seed':seed,'epoch':200,'checkpoint_sha256':before_cp,'state_and_checkpoint_unchanged':True,'native_prediction_token_weight_audit':audit,'stages':stages,'CNN_layer_stats':layers,'neighbor_semantics':neighbors,'probe_target_GT_not_used_for_fit_or_tuning':True,'source_probe_partition_fixed_seed2100':True};dump(folder/'summary.json',result);print(seed,'completed',flush=True)
if __name__=='__main__':main()

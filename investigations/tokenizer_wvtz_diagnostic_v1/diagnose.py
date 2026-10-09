"""Read-only frozen Houston A/C checkpoints; no optimizer or target-label scoring."""
import sys
sys.dont_write_bytecode=True
import pathlib,json,hashlib,time,itertools
import numpy as np
import torch
from argparse import Namespace
BASE=pathlib.Path('/nas1/zhangzj26/dual_axis_hsi_uda/results/tokenizer_mechanism_v2')
OUT=pathlib.Path(__file__).parent
sys.path.insert(0,str(BASE));import common
SEEDS=[2100,2101,2102];N=2048

def hasharray(x):return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()
def dump(p,x):pathlib.Path(p).write_text(json.dumps(x,indent=2)+'\n')
def describe(v):
 v=np.asarray(v,float);return {'min':float(v.min()),'mean':float(v.mean()),'max':float(v.max()),'std':float(v.std())}
def assignment(w):
 eps=1e-30;p=w[:,:,None,:];q=w[:,None,:,:];mix=(p+q)/2
 entropy=-(w*np.log(w+eps)).sum(-1);u=w/np.linalg.norm(w,axis=-1,keepdims=True);cos=u@u.transpose(0,2,1);mask=~np.eye(4,dtype=bool)
 js=.5*(p*(np.log(p+eps)-np.log(mix+eps))+q*(np.log(q+eps)-np.log(mix+eps))).sum(-1)
 return {'entropy_nats':describe(entropy),'entropy_normalized_mean':float(entropy.mean()/np.log(169)),'token_cosine_mean':float(cos[:,mask].mean()),'token_JS_mean':float(js[:,mask].mean()),'peak_mean':float(w.max(-1).mean()),'top10_mass_mean':float(np.sort(w,axis=-1)[:,:,-10:].sum(-1).mean()),'spatial_mean_map':w.mean(0).tolist()}
def changes(w,a):
 e=1e-30;m=(w+a)/2;js=.5*(w*(np.log(w+e)-np.log(m+e))+a*(np.log(a+e)-np.log(m+e))).sum(-1)
 kl=(w*(np.log(w+e)-np.log(a+e))).sum(-1);cos=(w*a).sum(-1)/(np.linalg.norm(w,axis=-1)*np.linalg.norm(a,axis=-1))
 return {'JS':describe(js.mean(-1)),'KL':describe(kl.mean(-1)),'cosine':describe(cos.mean(-1)),'max_abs_weight_error':float(abs(w-a).max())}
def domain_metrics(source,target):
 s=np.asarray(source,np.float64);t=np.asarray(target,np.float64);ms=s.mean(0);mt=t.mean(0);cs=np.cov(s,rowvar=False);ct=np.cov(t,rowvar=False)
 distance=np.linalg.norm(ms-mt);variance=.5*(np.trace(cs)+np.trace(ct));norms=np.linalg.norm(s,axis=-1);normt=np.linalg.norm(t,axis=-1)
 mean_cos=float(np.dot(ms,mt)/(np.linalg.norm(ms)*np.linalg.norm(mt)+1e-30))
 # Fixed equal-count subset; no target labels, bandwidth selection on distance only.
 gen=np.random.default_rng(65000);ids=gen.choice(len(s),512,replace=False);idt=gen.choice(len(t),512,replace=False)
 pooled=torch.from_numpy(np.concatenate([s[ids],t[idt]]));squares=(pooled*pooled).sum(-1);d=(squares[:,None]+squares[None,:]-2*pooled@pooled.T).clamp_min(0)
 triangle=d[torch.triu_indices(len(d),len(d),offset=1).unbind()];med=float(triangle.median());med=max(med,1e-12);kernel=sum(torch.exp(-d/(med*scale)) for scale in [.5,1.,2.])/3
 n=512;k=kernel.numpy();within_s=(k[:n,:n].sum()-np.trace(k[:n,:n]))/(n*(n-1));within_t=(k[n:,n:].sum()-np.trace(k[n:,n:]))/(n*(n-1));cross=k[:n,n:].mean();mmd=float(within_s+within_t-2*cross)
 # Deterministic label permutation null; negative unbiased MMD estimates retained.
 perm=[];total_diag=np.trace(k)
 for _ in range(100):
  order=gen.permutation(1024);i=order[:512];j=order[512:];ss=k[np.ix_(i,i)].sum();tt=k[np.ix_(j,j)].sum();st=k[np.ix_(i,j)].mean();perm.append(float((ss+tt-total_diag)/(n*(n-1))-2*st))
 pvalue=(1+sum(v>=mmd for v in perm))/101
 return {'dimension':int(s.shape[1]),'mean_distance':float(distance),'mean_distance_per_sqrt_dimension':float(distance/np.sqrt(s.shape[1])),'relative_mean_distance':float(distance/np.sqrt(variance+1e-30)),'mean_cosine':mean_cos,'mean_spectral_angle_radians':float(np.arccos(np.clip(mean_cos,-1,1))),'CORAL_loss':float(np.linalg.norm(cs-ct,'fro')**2/(4*s.shape[1]**2)),'relative_covariance_gap':float(np.linalg.norm(cs-ct,'fro')/(np.linalg.norm(cs,'fro')+np.linalg.norm(ct,'fro')+1e-30)),'MMD_unbiased_RBF_multiscale':mmd,'MMD_kernel_median_sqdistance':med,'MMD_perm_p_100':pvalue,'source_norm':describe(norms),'target_norm':describe(normt),'source_mean':ms.tolist(),'target_mean':mt.tolist()}

@torch.inference_mode()
def extract(model,x,device,perturbation=None):
 records={k:[] for k in ['W','V','T','Z']};original=model._forward_semantic_tokens;latest={}
 def token(features):
  latest['V']=features.detach()
  if hasattr(model,'relation_scores'):w=model.relation_scores(model._relation_pending[0])
  else:w=model.conv_a(features).flatten(2).softmax(-1)
  latest['W']=w.detach();tokens=original(features);latest['T']=tokens.detach();return tokens
 model._forward_semantic_tokens=token
 try:
  for start in range(0,len(x),128):
   patch=torch.from_numpy(x[start:start+128]).to(device)
   if perturbation is not None:patch=torch.from_numpy(perturbation[start:start+128]).to(device)
   out=model(patch,patch,return_feat_prob=True);latest['Z']=out[3].detach()
   for k in records:records[k].append(latest[k].cpu().numpy())
 finally:model._forward_semantic_tokens=original
 return {k:np.concatenate(v) for k,v in records.items()}

def main():
 assert not (OUT/'summary.json').exists();torch.set_num_threads(2);env=common.environment();device=torch.device('cuda:0');start=time.monotonic()
 args=Namespace(dataset='houston',dataset_dir='/home/zhangzj26/TGRS_MLUDA-2024/datasets/Houston/',num_workers=0,seed=2100);env[4](2100);ls=common.loaders(args)
 sourcecoords=np.unique(np.concatenate([ls[0].dataset.indices,ls[1].dataset.indices]),axis=0);targetcoords=ls[3].dataset.indices
 rng=np.random.default_rng(57000);sid=rng.choice(len(sourcecoords),N,replace=False);tid=rng.choice(len(targetcoords),N,replace=False)
 cubes={'source':ls[0].dataset.data,'target':ls[3].dataset.data};coords={'source':sourcecoords[sid],'target':targetcoords[tid]}
 inputs={}
 for domain in ['source','target']:
  inputs[domain]=np.stack([np.array(cubes[domain][i-6:i+7,j-6:j+7],np.float32).transpose(2,0,1)[None] for i,j in coords[domain]])
  np.save(OUT/f'{domain}_X.npy',inputs[domain]);np.save(OUT/f'{domain}_coordinates.npy',coords[domain])
 del ls,cubes
 perturbations={}
 for domain,x in inputs.items():
  gen=np.random.default_rng(81000 if domain=='source' else 82000);alpha=gen.uniform(.98,1.02,(N,1,1,1,1)).astype(np.float32);noise=gen.standard_normal(x.shape).astype(np.float32);mild=alpha*x+.005*noise
  gain=gen.uniform(.9,1.1,(N,1,48,1,1)).astype(np.float32);offset=gen.uniform(-.01,.01,(N,1,48,1,1)).astype(np.float32);affine=gain*x+offset
  perturbations[domain]={'mild':mild,'band_affine':affine}
  for name,a in perturbations[domain].items():np.save(OUT/f'{domain}_X_{name}.npy',a)
 rows={};checkpoint_hashes={};inputaudit={d:{'X_sha256':hasharray(x),'coordinates_sha256':hasharray(coords[d]),'center_spectral_norm':describe(np.linalg.norm(x[:,0,:,6,6],axis=-1))} for d,x in inputs.items()}
 for seed in SEEDS:
  for arm in 'AC':
   folder=OUT/f'{arm}_{seed}';folder.mkdir(exist_ok=False);checkpoint=BASE/'houston'/f'{arm}_{seed}/checkpoints/epoch_200.pth';checkpoint_hashes[f'{arm}_{seed}']=common.sha(checkpoint)
   env[4](seed);model,_=common.build(seed,arm,'houston');model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True)['model']);model.to(device);model.eval();before=common.state_hash(model)
   domainrows={};features={}
   for domain,x in inputs.items():
    values=extract(model,x,device);features[domain]=values
    for k,v in values.items():np.save(folder/f'{domain}_{k}.npy',v)
    arow={'assignment':assignment(values['W']),'stability':{}}
    for name,a in perturbations[domain].items():
     # Spectral affine invariance applies toC's W; A's W iscomputedthroughactualCNN.
     pert=extract(model,x,device,a);np.save(folder/f'{domain}_W_{name}.npy',pert['W']);arow['stability'][name]=changes(values['W'],pert['W']);del pert
    domainrows[domain]=arow
   representations={}
   positions=np.random.default_rng(63000).integers(0,169,N)
   for stage in ['X_center','V_center','V_mean','V_local','T','Z']:
    pair={}
    for domain in ['source','target']:
     if stage=='X_center':a=inputs[domain][:,0,:,6,6]
     elif stage=='V_center':a=features[domain]['V'][:,:,6,6]
     elif stage=='V_mean':a=features[domain]['V'].mean((-2,-1))
     elif stage=='V_local':a=features[domain]['V'].reshape(N,64,169)[np.arange(N),:,positions]
     elif stage=='T':a=features[domain]['T'].reshape(N,-1)
     else:a=features[domain]['Z']
     pair[domain]=a
    representations[stage]=domain_metrics(pair['source'],pair['target'])
   # Vfullgrid mean shift and perpatchnorm supplement local channel distributions.
   s=features['source']['V'].reshape(N,-1);t=features['target']['V'].reshape(N,-1);dist=np.linalg.norm(s.mean(0)-t.mean(0));variance=.5*(s.var(0,ddof=1).sum()+t.var(0,ddof=1).sum())
   representations['V_fullgrid_summary']={'relative_mean_distance':float(dist/np.sqrt(variance+1e-30)),'source_norm':describe(np.linalg.norm(s,axis=-1)),'target_norm':describe(np.linalg.norm(t,axis=-1))}
   assert before==common.state_hash(model),'Read-only inference changedmodel';assert checkpoint_hashes[f'{arm}_{seed}']==common.sha(checkpoint)
   r={'arm':arm,'seed':seed,'checkpoint_sha256':checkpoint_hashes[f'{arm}_{seed}'],'domains':domainrows,'domain_gaps':representations,'read_only_state_unchanged':True};rows[f'{arm}_{seed}']=r;dump(folder/'diagnostic.json',r)
   dump(OUT/'status.json',{'state':'diagnosing','completed':list(rows),'updated':time.time()});print('Completed',arm,seed,flush=True);del model,features
 summary={'dataset':'Houston','endpoint':200,'samples_per_domain':N,'seeds':SEEDS,'runs':rows,'input_audit':inputaudit,'checkpoint_hashes':checkpoint_hashes,'seconds':time.monotonic()-start,'no_training':True,'target_labels_not_used_for_diagnostic_metrics':True,'eval_path':'model.eval runningBN; eachdomain native paired(model(x,x)); Z=final normalizedCLS beforeclassifier','marginal_gap_caveat':'Source/target class priors and scene content differ; marginal discrepancy alone does not identify a causal classification bottleneck.'}
 dump(OUT/'summary.json',summary);dump(OUT/'status.json',{'state':'completed','six_checkpoints':True,'no_training':True,'updated':time.time()})
if __name__=='__main__':main()

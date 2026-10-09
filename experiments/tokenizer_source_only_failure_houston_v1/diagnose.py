import json,argparse,hashlib,time
from pathlib import Path
import torch,numpy as np
from common import OUT,BASE,SEEDS,ARMS,build,environment,state_hash,sha,dump
EPOCHS=[0,1,10,50,100,150,200]

def separation(x,y):
 x=np.asarray(x,dtype=np.float64).reshape(len(y),-1);x/=np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1e-30);sums=np.stack([x[y==k].sum(0) for k in range(7)]);center=sums/np.maximum(np.linalg.norm(sums,axis=1,keepdims=True),1e-30);sim=x@center.T;wrong=sim.copy();wrong[np.arange(len(y)),y]=-np.inf
 loo=sums[y]-x;loo/=np.maximum(np.linalg.norm(loo,axis=1,keepdims=True),1e-30);margin=(loo*x).sum(1)-wrong.max(1)
 return {'macro_margin_LOO':float(np.mean([margin[y==k].mean() for k in range(7)])),'macro_within_class_distance_LOO':float(np.mean([(1-(loo*x).sum(1))[y==k].mean() for k in range(7)])),'centroid_cosine_matrix':(center@center.T).tolist(),'per_class_margin':{str(k+1):float(margin[y==k].mean()) for k in range(7)}},center

@torch.inference_mode()
def infer(model,loader,capture=True):
 model.eval();outs=[];features={k:[] for k in ['V_mean','V_center','T','Z']};wstats=[];acc={n:[] for n in ['bn3d','bn2d']}
 def hook(name):
  def f(m,inputs):
   a=inputs[0];axes=(0,*range(2,a.ndim));acc[name].append((a.mean(axes).cpu().numpy().astype(float),a.var(axes,correction=0).cpu().numpy().astype(float),int(a.numel()/a.shape[1])))
  return f
 mods={'bn3d':model.conv3d_features[1],'bn2d':model.conv2d_features[1]};hooks=[m.register_forward_pre_hook(hook(n)) for n,m in mods.items()];normstats=[]
 try:
  for x,_ in loader:
   x=x.cuda()
   if capture:
    logits,f=model(x,return_features=True);v=f['V'];features['V_mean'].append(v.mean((-2,-1)).cpu().numpy());features['V_center'].append(v[:,:,6,6].cpu().numpy());features['T'].append(f['T'].cpu().numpy());features['Z'].append(f['Z'].cpu().numpy());w=f['W'];wstats.append(torch.stack((-(w*w.clamp_min(1e-30).log()).sum(-1).mean(1)/np.log(169),w.max(-1).values.mean(1)),1).cpu().numpy());normstats.append(v.flatten(1).norm(dim=1).cpu().numpy())
   else:logits=model(x)
   outs.append(logits.cpu().numpy())
 finally:
  for h in hooks:h.remove()
 bn={}
 for n,m in mods.items():
  a=acc[n];total=sum(v[2] for v in a);mu=sum(v[0]*v[2] for v in a)/total;var=sum((v[1]+v[0]**2)*v[2] for v in a)/total-mu**2;rm=m.running_mean.cpu().numpy();rv=m.running_var.cpu().numpy();ratio=var/(rv+m.eps)
  bn[n]={'input_running_mean_z_rms':float(np.sqrt(np.mean((mu-rm)**2/(rv+m.eps)))),'input_running_var_ratio_quantiles':np.quantile(ratio,[0,.1,.5,.9,1]).tolist(),'running_mean_rms':float(np.sqrt(np.mean(rm**2))),'running_var_quantiles':np.quantile(rv,[0,.1,.5,.9,1]).tolist(),'running_var_CV':float(rv.std()/max(rv.mean(),1e-30)),'input_mean':mu.tolist(),'input_var':var.tolist(),'finite':bool(np.isfinite(mu).all() and np.isfinite(var).all()),'running_var_below_1e8_count':int((rv<1e-8).sum())}
 fs={k:np.concatenate(v) for k,v in features.items()} if capture else {}
 extra={'assignment_entropy_normalized_mean':float(np.concatenate(wstats)[:,0].mean()),'assignment_peak_mean':float(np.concatenate(wstats)[:,1].mean()),'full_spatial_V_norm_mean':float(np.concatenate(normstats).mean())} if capture else {}
 return np.concatenate(outs),fs,bn,extra

def make_eval_data(seed):
 from utils.dataset import load_mat_hsi,HSIDataset
 from torch.utils.data import DataLoader,Subset
 data=[]
 for name in ['Houston13','Houston18']:
  image,gt,_=load_mat_hsi(name,'/home/zhangzj26/TGRS_MLUDA-2024/datasets/Houston/',norm='normband');data.append(HSIDataset(np.pad(image,((6,6),(6,6),(0,0)),mode='reflect'),np.pad(gt,((6,6),(6,6)),mode='reflect'),13,data_aug=False))
 source,target=data;labels=np.asarray(target.labels);rng=np.random.default_rng(92000);ids=np.sort(np.concatenate([rng.choice(np.flatnonzero(labels==k),min(256,int((labels==k).sum())),replace=False) for k in range(7)]));assert np.array_equal(labels,np.load(BASE/'houston/evaluation_labels.npy'))
 generator=torch.Generator().manual_seed(93000)
 def loader(ds):return DataLoader(ds,batch_size=128,num_workers=4,shuffle=False,generator=generator)
 return loader(source),loader(Subset(target,ids)),loader(target),np.asarray(source.labels),labels[ids],labels,ids

def main():
 p=argparse.ArgumentParser();p.add_argument('--seed',type=int,choices=SEEDS,required=True);args=p.parse_args();torch.set_num_threads(2);env=environment();root=OUT/'diagnostics'/f'seed_{args.seed}';root.mkdir(exist_ok=False)
 # Every requested training run must be frozen before target data is read.
 for arm in ARMS:
  for seed in SEEDS:assert (OUT/'houston'/f'{arm}_{seed}'/'frozen_manifest.json').exists()
 source,target,full,sl,tl,fl,ids=make_eval_data(args.seed);np.save(root/'target_trajectory_indices.npy',ids);priors=np.bincount(fl,minlength=7)/len(fl);rows=[]
 for arm in ARMS:
  folder=OUT/'houston'/f'{arm}_{args.seed}';manifest=json.loads((folder/'frozen_manifest.json').read_text());env[4](args.seed);model,_=build(args.seed,arm);model.cuda();checkpoints={q['epoch']:q for q in manifest['checkpoints']}
  for epoch in EPOCHS:
   cp=checkpoints[epoch];assert sha(cp['path'])==cp['sha256'];model.load_state_dict(torch.load(cp['path'],map_location='cpu',weights_only=True)['model']);before=state_hash(model);domain={};arrays={};centers={}
   for name,loader,labels in [('source',source,sl),('target',target,tl)]:
    logits,features,bn,extra=infer(model,loader);metric=env[6].metrics(labels,logits.argmax(1),logits,7);metric['collapse']=max(metric['prediction_shares'])>=.95;metric['zero_recall_classes']=[k+1 for k,v in enumerate(metric['per_class_percent']) if v==0];stages={};centers[name]={}
    for stage,values in features.items():
     stat,center=separation(values,labels);stat['norm_mean']=float(np.linalg.norm(values.reshape(len(labels),-1),axis=1).mean());stat['norm_std']=float(np.linalg.norm(values.reshape(len(labels),-1),axis=1).std());stages[stage]=stat;centers[name][stage]=center;arrays[f'{name}_{stage}']=values
    if name=='target':metric['trajectory_priorweighted_OA_estimate']=float(np.dot(priors,np.asarray(metric['per_class_percent'])));metric['trajectory_stratified_sample_OA_not_official']=metric.pop('oa_percent')
    domain[name]={'metrics':metric,'BN_functional_mismatch':bn,'stages':stages,**extra}
   cross={stage:{'matched_class_centroid_cosine_mean':float(np.diag(centers['source'][stage]@centers['target'][stage].T).mean()),'source_target_centroid_cosine_matrix':(centers['source'][stage]@centers['target'][stage].T).tolist()} for stage in centers['source']}
   np.savez_compressed(root/f'{arm}_e{epoch:03d}_features.npz',**arrays,source_labels=sl,target_labels=tl)
   row={'arm':arm,'seed':args.seed,'epoch':epoch,'domains':domain,'class_centroid_correspondence':cross,'checkpoint_sha256':cp['sha256']};rows.append(row);assert state_hash(model)==before;print(json.dumps({'arm':arm,'seed':args.seed,'epoch':epoch,'trajectory_AA':domain['target']['metrics']['aa_percent']}),flush=True)
   if epoch==200:
    logits,_,bn,_=infer(model,full,capture=False);np.savez_compressed(root/f'{arm}_e200_full_predictions.npz',logits=logits,predictions=logits.argmax(1));metric=env[6].metrics(fl,logits.argmax(1),logits,7);metric['collapse']=max(metric['prediction_shares'])>=.95;metric['zero_recall_classes']=[k+1 for k,v in enumerate(metric['per_class_percent']) if v==0];row['official_epoch200_full_target_metrics']=metric;row['official_full_target_BN_mismatch']=bn;assert state_hash(model)==before
 dump(root/'trajectory.json',{'rows':rows,'target_GT_only_offline_after_all_training_frozen':True,'target_trajectory_count':len(ids),'target_class_counts':np.bincount(tl,minlength=7).tolist(),'source_count':len(sl),'fixed_target_indices_sha256':hashlib.sha256(ids.tobytes()).hexdigest(),'checkpoints_not_modified':True});print('Read-only diagnosis complete',flush=True)
if __name__=='__main__':main()

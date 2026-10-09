import argparse,json,warnings,time
import numpy as np,torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import ConvergenceWarning
from common import OUT,BASE,PREVIOUS,SEEDS,frozen,dump,sha,hasharray
from metrics import separability

class Patches(torch.utils.data.Dataset):
 def __init__(self,d):self.cube=np.load(OUT/'data'/f'{d}_cube.npy',mmap_mode='r');self.coords=np.load(OUT/'data'/f'{d}_coords.npy')
 def __len__(self):return len(self.coords)
 def __getitem__(self,i):
  x,y=self.coords[i];return torch.from_numpy(np.array(self.cube[x-6:x+7,y-6:y+7],np.float32).transpose(2,0,1)).unsqueeze(0)

@torch.inference_mode()
def components(model,domain,folder):
 conv=model.conv3d_features[0];weight=conv.weight.detach();masks={k:torch.zeros_like(weight) for k in ['spatial_center','spatial_neighbors','spectral_offset0','spectral_neighbors']};masks['spatial_center'][:,:,:,1,1]=1;masks['spatial_neighbors']=1-masks['spatial_center'];masks['spectral_offset0'][:,:,1,:,:]=1;masks['spectral_neighbors']=1-masks['spectral_offset0'];masked={k:weight*v for k,v in masks.items()};ds=Patches(domain);loader=torch.utils.data.DataLoader(ds,batch_size=128,num_workers=2,shuffle=False);arrays={};pos=0;errors={'spatial':0.,'spectral':0.};old=np.load(PREVIOUS/f'seed_{model._diag_seed}'/f'{domain}_conv3d.npy',mmap_mode='r')
 for x in loader:
  x=x.cuda();center=x[:,0,:,6,6];near=x[:,:,:,5:8,5:8];values={'X_center':center.cpu().numpy(),'X_neighbor3_mean':((near[:,0].sum((-2,-1))-center)/8).cpu().numpy(),'X_neighbor13_mean':((x[:,0].sum((-2,-1))-center)/168).cpu().numpy()}
  for key,w in masked.items():values[key]=F.conv3d(x,w,bias=None,padding=(1,1,1))[...,6,6].flatten(1).cpu().numpy()
  bias=conv.bias.detach().cpu().numpy()[:,None]*np.ones((8,48));actual=np.asarray(old[pos:pos+len(x)])
  for key,a,b in [('spatial','spatial_center','spatial_neighbors'),('spectral','spectral_offset0','spectral_neighbors')]:errors[key]=max(errors[key],float(np.max(np.abs(values[a]+values[b]+bias.reshape(1,-1)-actual))))
  for key,a in values.items():
   if key not in arrays:arrays[key]=np.lib.format.open_memmap(folder/f'{domain}_{key}.npy',mode='w+',dtype=np.float32,shape=(len(ds),a.shape[1]))
   arrays[key][pos:pos+len(x)]=a
  pos+=len(x)
 assert max(errors.values())<1e-6,errors
 for a in arrays.values():a.flush()
 energy={k:float((v*v).sum()/(weight*weight).sum()) for k,v in masked.items()};return {'reconstruction_max_errors':errors,'kernel_energy_fractions':energy}

def probe(source,target,sl,tl,train,folder,name,helper):
 scale=StandardScaler();a=scale.fit_transform(source[train]);clf=LogisticRegression(C=1.,solver='lbfgs',max_iter=2000,tol=1e-6,random_state=98000)
 with warnings.catch_warnings(record=True) as seen:warnings.simplefilter('always');clf.fit(a,sl[train])
 hs=scale.transform(source[~train]);ht=scale.transform(target);ps=clf.predict(hs);pt=clf.predict(ht);np.savez_compressed(folder/f'{name}_probe.npz',coef=clf.coef_,intercept=clf.intercept_,scaler_mean=scale.mean_,scaler_scale=scale.scale_,target_predictions=pt)
 geom=separability(source,sl,target,tl);return {'probe_source':helper.metrics(sl[~train],ps,clf.decision_function(hs),7),'probe_target':helper.metrics(tl,pt,clf.decision_function(ht),7),'geometry':geom,'converged':not any(issubclass(w.category,ConvergenceWarning) for w in seen),'iterations':clf.n_iter_.tolist(),'dimension':source.shape[1]}

@torch.inference_mode()
def perturbations(model,folder,seed):
 ds=Patches('target');tl=np.load(OUT/'data/target_labels.npy');rng=np.random.default_rng(107000);ids=np.sort(np.concatenate([rng.choice(np.flatnonzero(tl==k),min(256,int((tl==k).sum())),replace=False) for k in range(7)]));labels=tl[ids];priors=np.bincount(tl,minlength=7)/len(tl);x=torch.stack([ds[int(i)] for i in ids]).cuda();np.save(folder/'fixed_perturb_target_indices.npy',ids);grid=np.linspace(0,1,48);direction=(np.sin(2*np.pi*grid+.31)+.5*np.cos(4*np.pi*grid+.73)).astype(np.float32);direction/=np.sqrt(np.mean(direction**2));perm=np.random.default_rng(108000).permutation(8);locations=[(i,j) for i in range(5,8) for j in range(5,8) if (i,j)!=(6,6)];conditions=[('clean',None,0.)]+[(mode,None,a) for mode in ['spectral_all','spectral_center','spectral_neighbors'] for a in [.02,.05,.10]]+[(name,None,0.) for name in ['flip_horizontal','flip_vertical','flip_both','rot90','permute8neighbors']];old=PREVIOUS/f'seed_{seed}';probes={stage:np.load(old/f'{stage}_source_probe.npz') for stage in ['F3D','V']};clean={};records=[]
 def prefix(inp):
  f=model.conv3d_features(inp);v=model.conv2d_features(f.flatten(1,2));return {'F3D':f[...,6,6].flatten(1),'V':v[...,6,6]}
 for name,_,a in conditions:
  if name.startswith('spectral'):
   gain=torch.from_numpy(np.exp(a*direction)).cuda().reshape(1,1,48,1,1);changed=x*gain;changed=changed/changed.norm(dim=2,keepdim=True).clamp_min(1e-12)
   if name=='spectral_all':aug=changed
   elif name=='spectral_center':aug=x.clone();aug[:,:,:,6,6]=changed[:,:,:,6,6]
   else:aug=changed;aug[:,:,:,6,6]=x[:,:,:,6,6]
  elif name=='flip_horizontal':aug=x.flip(-1)
  elif name=='flip_vertical':aug=x.flip(-2)
  elif name=='flip_both':aug=x.flip((-2,-1))
  elif name=='rot90':aug=torch.rot90(x,1,(-2,-1))
  elif name=='permute8neighbors':
   aug=x.clone()
   for dst,src in zip(locations,[locations[i] for i in perm]):aug[:,:,:,dst[0],dst[1]]=x[:,:,:,src[0],src[1]]
  else:aug=x
  if name.startswith(('flip','rot','permute')) or name=='spectral_neighbors':assert torch.equal(aug[:,:,:,6,6],x[:,:,:,6,6])
  relative_global=(aug-x).flatten(1).norm(dim=1)/x.flatten(1).norm(dim=1);relative_local=(aug-x)[:,:,:,5:8,5:8].flatten(1).norm(dim=1)/x[:,:,:,5:8,5:8].flatten(1).norm(dim=1);feat={k:[] for k in ['F3D','V']}
  for start in range(0,len(x),128):
   ff=prefix(aug[start:start+128])
   for k,v in ff.items():feat[k].append(v.cpu().numpy())
  feat={k:np.concatenate(v) for k,v in feat.items()};record={'condition':name,'strength':a,'center_input_max_error':float((aug[:,:,:,6,6]-x[:,:,:,6,6]).abs().max()),'input_relative_energy_global_quantiles':np.quantile(relative_global.cpu().numpy(),[0,.1,.5,.9,1]).tolist(),'input_relative_energy_local3x3_quantiles':np.quantile(relative_local.cpu().numpy(),[0,.1,.5,.9,1]).tolist(),'stages':{}}
  for stage,value in feat.items():
   if name=='clean':
    clean[stage]=value;expected=np.load(old/f'target_{stage}.npy',mmap_mode='r')[ids];assert np.max(np.abs(value-expected))<1e-6
   base=clean[stage];outchange=np.linalg.norm(value-base,axis=1)/np.maximum(np.linalg.norm(base,axis=1),1e-12);co=1-(value*base).sum(1)/(np.maximum(np.linalg.norm(value,axis=1)*np.linalg.norm(base,axis=1),1e-12));local=relative_local.cpu().numpy();gain=np.divide(outchange,local,out=np.zeros_like(outchange),where=local>1e-8);pr=probes[stage];decision=((value-pr['scaler_mean'])/pr['scaler_scale'])@pr['coef'].T+pr['intercept'];pred=decision.argmax(1);recall=np.array([(pred[labels==k]==k).mean() for k in range(7)]);geometry=separability(np.load(old/f'source_{stage}.npy',mmap_mode='r'),np.load(OUT/'data/source_labels.npy'),value,labels)
   record['stages'][stage]={'truepriorweighted_probe_OA_estimate':float(np.dot(priors,recall)*100),'probe_AA':float(recall.mean()*100),'perclass_recall_percent':(recall*100).tolist(),'feature_relative_L2_change_quantiles':np.quantile(outchange,[0,.1,.5,.9,1]).tolist(),'feature_cosine_change_mean':float(co.mean()),'feature_response_gain_over_local_input_change_quantiles':np.quantile(gain[local>1e-8],[0,.1,.5,.9,1]).tolist() if (local>1e-8).any() else None,'target_class_margin':geometry['target']['macro_margin_LOO'],'zero_feature_fraction':float((value==0).mean())};np.savez_compressed(folder/f'{name}_{a:.2f}_{stage}.npz',features=value,predictions=pred,relative_local_input_energy=local)
  records.append(record);print(seed,name,a,'F3D AA',record['stages']['F3D']['probe_AA'],flush=True)
 return {'target_samples':len(ids),'class_counts':np.bincount(labels,minlength=7).tolist(),'indices_sha256':hasharray(ids),'spectral_direction':direction.tolist(),'neighbor_permutation':perm.tolist(),'records':records,'target_GT_only_offline_no_target_tuning':True}

def main():
 p=argparse.ArgumentParser();p.add_argument('--seed',type=int,choices=SEEDS,required=True);seed=p.parse_args().seed;torch.set_num_threads(2);env=frozen.environment();env[4](seed);folder=OUT/f'seed_{seed}';folder.mkdir(exist_ok=False);cp=BASE/'houston'/f'C_{seed}/checkpoints/epoch_200.pth';hashbefore=sha(cp);m,_=frozen.build(seed,'C','houston');m.load_state_dict(torch.load(cp,map_location='cpu',weights_only=True)['model']);m.cuda().eval();m._diag_seed=seed;state=frozen.state_hash(m);assert len(m.conv3d_features)==3 and isinstance(m.conv3d_features[0],torch.nn.Conv3d);conv=m.conv3d_features[0];assert conv.kernel_size==(3,3,3) and conv.padding==(1,1,1) and conv.stride==(1,1,1)
 decomposition={d:components(m,d,folder) for d in ['source','target']};sl=np.load(OUT/'data/source_labels.npy');tl=np.load(OUT/'data/target_labels.npy');train=np.load(OUT/'data/source_probe_train_mask.npy');results={};old=PREVIOUS/f'seed_{seed}'
 for name in ['conv3d','bn3d','conv2d','bn2d','X_center','X_neighbor3_mean','X_neighbor13_mean','spatial_center','spatial_neighbors','spectral_offset0','spectral_neighbors']:
  parent=old if name in ['conv3d','bn3d','conv2d','bn2d'] else folder;s=np.load(parent/f'source_{name}.npy',mmap_mode='r');t=np.load(parent/f'target_{name}.npy',mmap_mode='r');results[name]=probe(s,t,sl,tl,train,folder,name,env[6]);print(seed,name,'target',results[name]['probe_target']['oa_percent'],results[name]['probe_target']['aa_percent'],flush=True)
 for name in ['F3D','V']:
  previous=json.loads((old/'summary.json').read_text())['stages'][name];results[name]={'probe_source':previous['probe_source_heldout'],'probe_target':previous['probe_target'],'geometry':previous['geometry'],'converged':previous['probe_converged'],'dimension':previous['domain_stats']['descriptor_dimension'],'reused_previous_frozen_probe':True}
 curves=perturbations(m,folder,seed);gating={d:{stage:{'zero_or_negative_fraction':float((np.load(old/f'{d}_{stage}.npy',mmap_mode='r')<=0).mean()),'negative_fraction':float((np.load(old/f'{d}_{stage}.npy',mmap_mode='r')<0).mean())} for stage in ['bn3d','F3D']} for d in ['source','target']};assert state==frozen.state_hash(m) and sha(cp)==hashbefore
 dump(folder/'summary.json',{'seed':seed,'epoch':200,'architecture':{'Conv3d_layers':1,'input_shape':[1,48,13,13],'Conv3d_output_shape':[8,48,13,13],'spectral_positions_before_after':[48,48],'flattened_features':384,'Conv2d_output_channels':64,'flatten_is_invertible':True,'no_3D_spectral_compression':True},'checkpoint_sha256':hashbefore,'state_and_checkpoint_unchanged':True,'decomposition':decomposition,'probes_and_geometry':results,'perturb_diagnostics':curves,'ReLU_center_gating':gating});print(seed,'completed',flush=True)
if __name__=='__main__':main()

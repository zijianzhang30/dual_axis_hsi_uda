import argparse,time,json,warnings
from pathlib import Path
import numpy as np,torch
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import ConvergenceWarning
import common as c

def run(seed):
 # Enforce target-label read barrier in every extraction/probe process.
 origload=np.load
 def guarded(path,*args,**kwargs):
  assert 'target_labels' not in str(path) and 'target_gt' not in str(path),'Target label access before predictions frozen';return origload(path,*args,**kwargs)
 np.load=guarded;c.verify();torch.set_num_threads(2);f=c.frozen();env=f.environment();env[4](seed);model,_=f.build(seed,'C','houston');cp=c.BASE/'houston'/('C_%d/checkpoints/epoch_200.pth'%seed);model.load_state_dict(torch.load(cp,map_location='cpu',weights_only=True)['model']);model=model.cuda().eval();model.requires_grad_(False)
 before=f.state_hash(model);bn={k:v.clone() for k,v in model.state_dict().items() if 'running_' in k or 'num_batches_tracked' in k};folder=c.OUT/('seed_%d'%seed);folder.mkdir(exist_ok=False)
 # Zero-offset original tokenization, including original relation pending queue.
 ds=c.Windows('source',c.OFFSETS[0]);x=torch.stack([ds[i] for i in range(12)]).cuda()
 with torch.inference_mode():
  model._relation_pending=[x];reference=model._tokenize(x);assert not model._relation_pending;direct=c.tokens(model,x,c.ANCHORS[0]);error=float((reference-direct).abs().max());assert error<1e-6
  # Synthetic coordinate ramp, allfive windows must reference the exact same spectrum.
  cube=np.arange(25*25*48,dtype=np.float32).reshape(25,25,48);p=np.array([12,12])
  for off,anchor in zip(c.OFFSETS,c.ANCHORS):
   z=p+off;patch=torch.from_numpy(cube[z[0]-6:z[0]+7,z[1]-6:z[1]+7].transpose(2,0,1).copy()).unsqueeze(0).unsqueeze(0).cuda();assert np.array_equal(patch[0,0,:,anchor[0],anchor[1]].cpu().numpy(),cube[p[0],p[1]])
   rel=model.relation_scores.relations(patch,coordinates=anchor);assert float(rel[:,:,anchor[0],anchor[1]].abs().max())==0
 c.dump(folder/'preflight.json',{'passed':True,'zero_offset_original_token_max_error':error,'five_anchor_geometry_correct':True,'main_requires_grad_false':True,'BN_eval':True,'no_target_labels':True})
 times={}
 with torch.inference_mode():
  for domain in ('source','target'):
   n=len(c.Windows(domain,c.OFFSETS[0]));a=np.lib.format.open_memmap(folder/('%s_tokens.npy'%domain),mode='w+',dtype=np.float32,shape=(n,5,256));times[domain]=[]
   for vi,(off,anchor) in enumerate(zip(c.OFFSETS,c.ANCHORS)):
    ds=c.Windows(domain,off);loader=torch.utils.data.DataLoader(ds,batch_size=128,shuffle=False,num_workers=0,generator=torch.Generator().manual_seed(seed+85000));pos=0;torch.cuda.synchronize();start=time.monotonic()
    for x in loader:
     x=x.cuda();t=c.tokens(model,x,anchor).flatten(1);a[pos:pos+len(x),vi]=t.cpu().numpy();pos+=len(x)
    torch.cuda.synchronize();times[domain].append(time.monotonic()-start);assert pos==n;assert f.state_hash(model)==before;a.flush();print(seed,domain,'view',vi,'complete',flush=True)
   del a
 assert f.state_hash(model)==before and all(torch.equal(v,model.state_dict()[k]) for k,v in bn.items());assert c.sha(cp)==json.loads((c.OUT/'input_and_code_manifest.json').read_text())[str(cp)]
 source=np.load(folder/'source_tokens.npy',mmap_mode='r');target=np.load(folder/'target_tokens.npy',mmap_mode='r');mask=np.load(c.OUT/'data/source_train_mask.npy');labels=np.load(c.OUT/'data/source_labels.npy');X=np.asarray(source[mask]).reshape(-1,256);y=np.repeat(labels[mask],5)
 scaler=StandardScaler();X=scaler.fit_transform(X);probe=LogisticRegression(C=1.,solver='lbfgs',max_iter=2000,tol=1e-6,random_state=98000,class_weight=None)
 start=time.monotonic()
 with warnings.catch_warnings(record=True) as ws:warnings.simplefilter('always');probe.fit(X,y)
 converged=not any(issubclass(w.category,ConvergenceWarning) for w in ws);assert np.array_equal(probe.classes_,np.arange(7))
 np.savez_compressed(folder/'probe.npz',coef=probe.coef_,intercept=probe.intercept_,classes=probe.classes_,scaler_mean=scaler.mean_,scaler_scale=scaler.scale_)
 c.dump(folder/'probe_config.json',{**c.PROBE,'train_unique_pixels':int(mask.sum()),'train_feature_rows':len(y),'val_unique_pixels':int((~mask).sum()),'train_labels_are_original_pixel':True,'converged':converged,'iterations':probe.n_iter_.tolist(),'warnings':[str(w.message) for w in ws],'fit_seconds':time.monotonic()-start,'hyperparameters_locked_before_target_GT':True})
 for domain,feat in [('source',source),('target',target)]:
  probabilities=np.empty((len(feat),5,7),np.float32)
  for start in range(0,len(feat),1024):probabilities[start:start+1024]=probe.predict_proba(scaler.transform(np.asarray(feat[start:start+1024]).reshape(-1,256))).reshape(-1,5,7)
  np.savez_compressed(folder/('%s_predictions.npz'%domain),probabilities=probabilities,per_view_predictions=probabilities.argmax(-1),S0=probabilities[:,0].argmax(-1),S1=probabilities.mean(1).argmax(-1))
 paths=[folder/name for name in ('source_tokens.npy','target_tokens.npy','probe.npz','probe_config.json','source_predictions.npz','target_predictions.npz')]
 c.dump(folder/'frozen_manifest.json',{'seed':seed,'checkpoint':str(cp),'checkpoint_sha256':c.sha(cp),'model_state_unchanged':True,'BN_buffers_bitwise_unchanged':True,'main_model_eval_no_grad':True,'no_target_GT_accessed':True,'source_5views_label_count_not_unique_pixels':len(y),'encoding_seconds_per_view':times,'hashes':{str(p):c.sha(p) for p in paths}})
 print('PREDICTIONS_FROZEN_NO_TARGET_GT',seed,flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--seed',type=int,choices=c.SEEDS,required=True);run(p.parse_args().seed)

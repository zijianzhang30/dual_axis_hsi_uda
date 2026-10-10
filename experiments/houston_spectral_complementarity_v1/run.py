import argparse,time,warnings
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import ConvergenceWarning
from common import *

def main(seed):
 verify();folder=OUT/('seed_%d'%seed);folder.mkdir();train=np.load(BASE/'data/source_probe_train_mask.npy');y=np.load(BASE/'data/source_labels.npy');rows=[];start=time.monotonic();xs={k:np.load(OUT/'data'/('source_'+k+'.npy')) for k in ('RAW','STRUCT')};xt={k:np.load(OUT/'data'/('target_'+k+'.npy')) for k in ('RAW','STRUCT')}
 def fit(name,source,target,representation_state=None):
  ts=time.monotonic();scale=StandardScaler();a=scale.fit_transform(source[train]);model=LogisticRegression(C=1.,solver='lbfgs',max_iter=2000,tol=1e-6,random_state=98000)
  with warnings.catch_warnings(record=True) as ww:
   warnings.simplefilter('always');model.fit(a,y[train])
  assert not any(issubclass(w.category,ConvergenceWarning) for w in ww),(name,'probe_not_converged_no_auto_tuning')
  zs=model.decision_function(scale.transform(source));zt=model.decision_function(scale.transform(target));assert np.isfinite(zs).all() and np.isfinite(zt).all();assert np.array_equal(model.classes_,np.arange(7))
  p=folder/(name+'.npz');np.savez_compressed(p,coef=model.coef_,intercept=model.intercept_,scaler_mean=scale.mean_,scaler_scale=scale.scale_,source_logits=zs,target_logits=zt,source_predictions=model.predict(scale.transform(source)),target_predictions=model.predict(scale.transform(target)))
  row={'name':name,'dimensions':source.shape[1],'stored_linear_parameters':model.coef_.size+model.intercept_.size,'independent_multinomial_parameters':6*(source.shape[1]+1),'iterations':model.n_iter_.tolist(),'converged':True,'source_only_fit':True,'target_GT_read':False,'seconds':time.monotonic()-ts,'path':str(p),'sha256':sha(p),'projection':representation_state};rows.append(row);dump(folder/'probes.json',rows);print(seed,name,'fitted',source.shape[1],flush=True)
 for k in ('RAW','STRUCT'):fit(k,xs[k],xt[k])
 for stage in ('T','Z'):
  source=np.asarray(np.load(BASE/('seed_%d'%seed)/('source_'+stage+'.npy')),np.float64);target=np.asarray(np.load(BASE/('seed_%d'%seed)/('target_'+stage+'.npy')),np.float64);d=source.shape[1];old=np.load(BASE/('seed_%d'%seed)/(stage+'_source_probe.npz'))
  # Reuse identical historical source-only fit; save full logits for common scoring.
  a=(source-old['scaler_mean'])/old['scaler_scale'];b=(target-old['scaler_mean'])/old['scaler_scale'];zs=a@old['coef'].T+old['intercept'];zt=b@old['coef'].T+old['intercept'];assert np.array_equal(zs[~train].argmax(1),old['source_heldout_predictions']) and np.array_equal(zt.argmax(1),old['target_predictions'])
  p=folder/(stage+'.npz');np.savez_compressed(p,coef=old['coef'],intercept=old['intercept'],scaler_mean=old['scaler_mean'],scaler_scale=old['scaler_scale'],source_logits=zs,target_logits=zt,source_predictions=zs.argmax(1),target_predictions=zt.argmax(1));rows.append({'name':stage,'dimensions':d,'stored_linear_parameters':7*(d+1),'independent_multinomial_parameters':6*(d+1),'source_only_fit':True,'reused_matching_historical_probe':True,'historical_predictions_equal':True,'target_GT_read':False,'path':str(p),'sha256':sha(p)});dump(folder/'probes.json',rows)
  for k in ('RAW','STRUCT'):fit(stage+'_'+k,np.concatenate([source,xs[k]],1),np.concatenate([target,xt[k]],1))
  # Matched total dimensionality. Same projection for RAW and STRUCT, source-train only.
  sc=StandardScaler();a=sc.fit_transform(source[train]);pca=PCA(n_components=d-48,svd_solver='full',whiten=False,random_state=98000);pca.fit(a);ss=pca.transform(sc.transform(source));tt=pca.transform(sc.transform(target));p=folder/(stage+'_source_PCA.npz');np.savez_compressed(p,scaler_mean=sc.mean_,scaler_scale=sc.scale_,pca_mean=pca.mean_,components=pca.components_,explained_variance=pca.explained_variance_,explained_variance_ratio=pca.explained_variance_ratio_);pr={'path':str(p),'sha256':sha(p),'fit_only_source_train':True,'output_dim':d-48,'source_train_explained_variance_fraction':float(pca.explained_variance_ratio_.sum())}
  fit(stage+'_PCA_only',ss,tt,pr)
  for k in ('RAW','STRUCT'):fit(stage+'_PCA_'+k,np.concatenate([ss,xs[k]],1),np.concatenate([tt,xt[k]],1),pr)
 verify();assert len(rows)==14
 files={r['path']:r['sha256'] for r in rows}
 for r in rows:
  if r.get('projection'):files[r['projection']['path']]=r['projection']['sha256']
 dump(folder/'frozen_manifest.json',{'seed':seed,'probes':len(rows),'all_source_only_fit':True,'target_GT_read':False,'new_main_network_training':False,'main_network_forwards':0,'seconds':time.monotonic()-start,'files':files});print('SPECTRAL_SEED_COMPLETE',seed,flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--seed',type=int,choices=SEEDS,required=True);main(p.parse_args().seed)

"""Read-only frozen endpoint analysis. No optimizer, no training, no checkpoint search."""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

PROJECT=Path('/home/zhangzj26/dual_axis_hsi_uda')
BASE=PROJECT/'results/mluda_mechanism_houston_v1'
OUT=Path('/nas1/zhangzj26/dual_axis_hsi_uda/investigations/mluda_houston_m2_analysis_v1')
SEEDS=(1174,1370,1417)

def dump(p,x):p.write_text(json.dumps(x,indent=2)+'\n')
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for x in iter(lambda:f.read(1024*1024),b''):h.update(x)
 return h.hexdigest()

def scalar_stats(x):
 return {'mean':float(x.mean()),'std':float(x.std(ddof=1)) if len(x)>1 else 0.,'min':float(x.min()),'max':float(x.max())}

def class_analysis():
 OUT.mkdir(parents=True,exist_ok=False)
 (PROJECT/'investigations/mluda_houston_m2_analysis_v1/results').symlink_to(OUT,target_is_directory=True)
 gt=np.load(BASE/'preprocessed/target_labels.npy');full=np.bincount(gt[gt>0].astype(int),minlength=8)[1:]
 records=[]
 for seed in SEEDS:
  a=json.loads((BASE/f'M0_{seed}/metrics.json').read_text()); b=json.loads((BASE/f'M2_{seed}/metrics.json').read_text())
  ca,cb=np.array(a['confusion']),np.array(b['confusion']);n=ca.sum(1);assert np.array_equal(n,cb.sum(1))
  delta=cb-ca; dr=np.array(b['recall'])-a['recall'];oa=n/a['native_denominator_n']*dr
  assert abs(oa.sum()-(b['oa']-a['oa']))<1e-10
  classes=[]
  for k in range(7):
   flows=[{'true_class':k+1,'predicted_class':j+1,'M0_count':int(ca[k,j]),'M2_count':int(cb[k,j]),'delta_count':int(delta[k,j]),'delta_conditional_percent':float(100*delta[k,j]/n[k])} for j in range(7) if j!=k]
   classes.append({'class':k+1,'target_n_full':int(full[k]),'target_p_full':float(full[k]/full.sum()),'target_n_evaluated':int(n[k]),
    'p_evaluated':float(n[k]/n.sum()),'p_native_OA':float(n[k]/a['native_denominator_n']),
    'recall_M0':a['recall'][k],'recall_M2':b['recall'][k],'delta_recall_pp':float(dr[k]),'delta_OA_pp':float(oa[k]),'delta_AA_pp':float(dr[k]/7),
    'error_flows':sorted(flows,key=lambda z:abs(z['delta_count']),reverse=True)})
  pa=np.load(BASE/f'M0_{seed}/predictions.npy');pb=np.load(BASE/f'M2_{seed}/predictions.npy');y=np.load(BASE/f'M0_{seed}/labels.npy')
  assert np.array_equal(y,np.load(BASE/f'M2_{seed}/labels.npy'))
  migration=np.zeros((7,7,7),dtype=int);np.add.at(migration,(y,pa,pb),1)
  np.save(OUT/f'prediction_transitions_{seed}.npy',migration)
  records.append({'seed':seed,'delta_oa':b['oa']-a['oa'],'delta_aa':b['aa']-a['aa'],'classes':classes,
   'confusion_M0':ca.tolist(),'confusion_M2':cb.tolist(),'confusion_delta':delta.tolist(),
   'prediction_transitions_axes':['true class','M0 prediction','M2 prediction']})
 # Gate is a direct descriptive check, not a significance test or accuracy tuning.
 gate={'start_R':False,'reason':'1174 has negative AA; majority class6 recall increases in1174/1417, falls only in1370. Thus the proposed pattern is not supported by all three seeds.',
       'no_new_training':True,'no_target_best':True}
 dump(OUT/'class_analysis.json',{'runs':records,'full_target_counts':full.tolist(),'gate':gate,
  'denominator_note':'Native OA uses53200; evaluated53184. p_native=n_evaluated_class/53200 exactly reconstructs native deltaOA; p_evaluated and full-cube p also reported.'})
 dump(OUT/'status.json',{'state':'class_analysis_complete','residual_R_started':False})
 print('Class analysis complete; R gate FAIL',flush=True)

def replay(seed):
 from common import tensor_hash,state_hash
 import net2,utils
 s=np.load(BASE/'preprocessed/source.npy');t=np.load(BASE/'preprocessed/target.npy')
 ls=np.load(BASE/'preprocessed/source_labels.npy');lt=np.load(BASE/'preprocessed/target_labels.npy')
 utils.set_seed(seed)
 sx,sy=utils.get_sample_data(s,ls,3,180)
 _,tx,ty,_,permutation,rows,columns=utils.get_all_data(t,lt,3)
 frozen=json.loads((BASE/f'M0_{seed}/frozen.json').read_text())
 assert tensor_hash(sx)==frozen['samples']['source_data']
 assert tensor_hash(tx)==frozen['samples']['target_data']
 assert tensor_hash(ty)==frozen['samples']['target_labels']
 source_ds=TensorDataset(torch.tensor(sx),torch.tensor(sy));target_ds=TensorDataset(torch.tensor(tx),torch.tensor(ty))
 sl=DataLoader(source_ds,batch_size=32,shuffle=True,drop_last=True)
 unused=DataLoader(source_ds,batch_size=32,shuffle=True,drop_last=True)
 tl=DataLoader(target_ds,batch_size=32,shuffle=True,drop_last=True)
 test=DataLoader(target_ds,batch_size=32,shuffle=False,drop_last=True)
 original=net2.DSANSS(48,7,7)
 initial=json.loads((BASE/f'M0_{seed}/initialization.json').read_text())
 assert state_hash(original.state_dict())==initial['native_state_sha256']
 # Native augmentations use NumPy only; native model dropout uses CUDA only.
 # Replay only CPU loader RNG, after same native model initialization, to recover
 # final S reference. No optimizer or model training forward is executed.
 for epoch in range(100):
  si,ti=iter(sl),iter(tl)
  for i in range(1,len(sl)):
   source,label=next(si);next(ti)
 assert tensor_hash(source)==frozen['test_source_pair_sha256']
 assert np.array_equal(ty[:53184],np.load(BASE/f'M0_{seed}/labels.npy'))
 np.save(OUT/f'test_source_pair_{seed}.npy',source.numpy())
 return original,source,test

def features(seed):
 sys.path.insert(0,str(BASE))
 sys.path.insert(0,str(BASE/'native_snapshot'))
 from common import tensor_hash,state_hash
 from mechanisms import install
 import net2
 # Diagnostic inference only: stable common FP32 backend for both frozen checkpoints.
 torch.backends.cudnn.benchmark=False
 torch.backends.cudnn.deterministic=True
 torch.backends.cudnn.allow_tf32=False
 torch.backends.cuda.matmul.allow_tf32=False
 original,source,loader=replay(seed)
 a=original
 b=net2.DSANSS(48,7,7);install(b,False,True,seed)
 for arm,model in [('M0',a),('M2',b)]:
  d=BASE/f'{arm}_{seed}';f=json.loads((d/'frozen.json').read_text())
  assert sha(d/'epoch100.pth')==f['checkpoint_sha256']
  model.load_state_dict(torch.load(d/'epoch100.pth',map_location='cpu',weights_only=False),strict=True)
  model.cuda().eval()
 before=[state_hash(m.state_dict()) for m in [a,b]]
 captured={'a':[],'b_avg':[],'b_rel':[]}
 def ha(mod,args,output):
  captured['a'].append(output.reshape(output.shape[0],288).detach().cpu())
 def hb(mod,args,output):
  captured['b_avg'].append(args[0].mean((-2,-1)).detach().cpu())
  captured['b_rel'].append(output.detach().cpu())
 h1=a.feature_layers.avgpool.register_forward_hook(ha)
 h2=b.feature_layers.relation_pool.register_forward_hook(hb)
 collected={k:[] for k in ['M0_avg','M2_avg','M2_rel']};pred0=[];pred2=[];logit0=[];logit2=[]
 with torch.no_grad():
  for data,_ in loader:
   captured={'a':[],'b_avg':[],'b_rel':[]}
   ra=a(source.cuda(),data.cuda());rb=b(source.cuda(),data.cuda())
   assert all(len(v)==2 for v in captured.values())
   collected['M0_avg'].append(captured['a'][1].numpy())
   collected['M2_avg'].append(captured['b_avg'][1].numpy())
   collected['M2_rel'].append(captured['b_rel'][1].numpy())
   pred0.append(ra[8].argmax(1).cpu().numpy());pred2.append(rb[8].argmax(1).cpu().numpy())
   logit0.append(ra[8].cpu().numpy());logit2.append(rb[8].cpu().numpy())
 h1.remove();h2.remove()
 assert before==[state_hash(m.state_dict()) for m in [a,b]],'Inference modified checkpoint state'
 reproduction={}
 for arm,preds,logits in [('M0',pred0,logit0),('M2',pred2,logit2)]:
  now=np.concatenate(preds);old=np.load(BASE/f'{arm}_{seed}/predictions.npy');bad=np.where(now!=old)[0]
  l=np.concatenate(logits);top=np.sort(l,axis=1)[:,-2:];margins=top[:,1]-top[:,0]
  reproduction[arm]={'mismatches':int(len(bad)),'n':len(old),'mismatch_indices':bad.tolist(),'diagnostic_top2_margins_at_disagreement':margins[bad].tolist(),'maximum_native_OA_difference_bound_pp':float(100*len(bad)/53200)}
  print(seed,arm,'prediction mismatches',len(bad),flush=True)
  np.save(OUT/f'diagnostic_predictions_{arm}_{seed}.npy',now)
 dump(OUT/f'prediction_reproduction_{seed}.json',reproduction)
 # Original frozen predictions remain the sole source of all recall/OA calculations.
 # Reloaded inference may differ slightly with common diagnostic FP32 kernels.
 assert all(v['mismatches']/v['n']<.001 for v in reproduction.values()),reproduction
 arrays={k:np.concatenate(v) for k,v in collected.items()}
 np.savez_compressed(OUT/f'pool_features_{seed}.npz',**arrays)
 labels=np.load(BASE/f'M0_{seed}/labels.npy')
 comparisons={}
 for left,right,name in [('M0_avg','M2_rel','cross_checkpoint_avg_to_relation'),('M2_avg','M2_rel','within_M2_pool_operator'),('M0_avg','M2_avg','cross_checkpoint_CNN_mean')]:
  x,z=arrays[left],arrays[right];normx=np.linalg.norm(x,axis=1);normz=np.linalg.norm(z,axis=1)
  cosine=(x*z).sum(1)/(normx*normz+1e-12)
  relative=np.linalg.norm(z-x,axis=1)/(normx+1e-12)
  classes=[]
  for k in range(7):
   mask=labels==k;cx=x[mask].mean(0);cz=z[mask].mean(0)
   classes.append({'class':k+1,'n':int(mask.sum()),'left_norm':scalar_stats(normx[mask]),'right_norm':scalar_stats(normz[mask]),
    'cosine':scalar_stats(cosine[mask]),'relative_shift':scalar_stats(relative[mask]),
    'left_class_mean':cx.tolist(),'right_class_mean':cz.tolist(),
    'class_mean_norm_left':float(np.linalg.norm(cx)),'class_mean_norm_right':float(np.linalg.norm(cz)),
    'class_mean_cosine':float(np.dot(cx,cz)/(np.linalg.norm(cx)*np.linalg.norm(cz)+1e-12)),
    'class_mean_shift_norm':float(np.linalg.norm(cz-cx)),
    'class_mean_relative_shift':float(np.linalg.norm(cz-cx)/(np.linalg.norm(cx)+1e-12))})
  comparisons[name]={'left':left,'right':right,'left_norm':scalar_stats(normx),'right_norm':scalar_stats(normz),
   'cosine':scalar_stats(cosine),'relative_shift':scalar_stats(relative),'classes':classes}
 dump(OUT/f'feature_comparison_{seed}.json',{'seed':seed,'stage':'post_CA_SA/pre_cross_attention',
  'source_pair_hash_verified':True,'prediction_reproduction':reproduction,'predictions_exactly_reproduced':all(v['mismatches']==0 for v in reproduction.values()),'diagnostic_backend':{'cudnn_benchmark':False,'cudnn_deterministic':True,'conv_tf32':False,'matmul_tf32':False},'checkpoint_states_unchanged':True,
  'comparisons':comparisons,'limitations':'Cross-checkpoint differences include co-adaptation and channel-basis changes; within-M2 comparison isolates pooling on identical CNN values, not a trained M0 counterfactual.'})
 print('Feature analysis complete seed',seed,flush=True)

def main():
 p=argparse.ArgumentParser();p.add_argument('--classes',action='store_true');p.add_argument('--seed',type=int,choices=SEEDS);args=p.parse_args()
 if args.classes:class_analysis()
 else:features(args.seed)

if __name__=='__main__':main()

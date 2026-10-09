"""Eq9 fixed-checkpoint feasibility diagnostic and minimal isolated source-pairing kernel.
Target GT is never a screening or source-pairing argument. No model training.
"""
import csv
import hashlib
import json
import math
import shutil
from pathlib import Path
import numpy as np
from scipy.io import loadmat
from sklearn.model_selection import train_test_split

ROOT=Path('/home/zhangzj26/dual_axis_hsi_uda')
OUT=Path('/nas1/zhangzj26/dual_axis_hsi_uda/results/bida_hyrank_eq9_diagnostic_v1')
BASE=OUT.parent/'bida_hyrank_paper_repro_v4'
SEEDS=(2100,2101,2102)
THRESHOLD=.5*math.log(12)
PLAN={'scope':'read-only official release Eq9 diagnostic; not class-pairing training',
 'seeds':list(SEEDS),'epochs':list(range(10,201,10)),'highlight_epochs':[10,20,50,100,200],
 'C':12,'entropy_threshold':THRESHOLD,'threshold_source':'paper sectionII-B 0.5lnC; not target-selected',
 'entropy':'per-sample -sum_c p log p, following sample-selection prose; Eq9 batch-averaging notation needs clarification',
 'features':'existing frozen original BiDA probabilities, full10317-pixel evaluation cohort',
 'source_pool':'actual source95% training split, preserving release hardcoded random_state23; CLI seed ignored by source split',
 'pairing_kernel':'each retained target pseudo-class selects a labeled source index of that class, uniformly with replacement, private RNG',
 'kernel_limits':'variable-size pairs only; no refill,initialization,cache,warmup or optimizer-budget policies silently invented',
 'GT':'only offline evaluation after selection and pair choices saved',
 'limitations':['Frozen eval/running-BN/no-dropout probabilities do not equal actual training-iteration probabilities.',
               'Better pseudo-class matching than random matching does not prove end-to-end OA gains.',
               'Kernel is a declared minimal reconstruction, not the missing author sampler implementation.'],
 'no_training':True,'no_Ours':True,'no_change_to_running_MMD_repair':True}

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def dump(path,x):Path(path).write_text(json.dumps(x,indent=2)+'\n')
def screen(logits):
 z=np.asarray(logits,dtype=np.float64);z=z-z.max(1,keepdims=True)
 p=np.exp(z);p/=p.sum(1,keepdims=True)
 entropy=-(p*np.log(np.maximum(p,1e-300))).sum(1)
 return p.argmax(1),entropy,entropy<=THRESHOLD

def pair_kernel(source_labels,logits,rng_seed):
 """No target labels; source pool is label-addressed. Empty results stay empty."""
 pseudo,entropy,accepted=screen(logits);target_indices=np.flatnonzero(accepted)
 rng=np.random.RandomState(rng_seed);src=[];tar=[];unavailable=[]
 for i in target_indices:
  bank=np.flatnonzero(source_labels==pseudo[i])
  if len(bank)==0:unavailable.append(int(i));continue
  src.append(int(rng.choice(bank)));tar.append(int(i))
 source_indices=np.asarray(src,dtype=np.int64);target_indices=np.asarray(tar,dtype=np.int64)
 assert np.array_equal(source_labels[source_indices],pseudo[target_indices])
 return source_indices,target_indices,pseudo,entropy,accepted,np.asarray(unavailable,dtype=np.int64)

def pct(mask,x):return float(x[mask].mean()*100) if mask.any() else None
def main():
 assert not (OUT/'frozen_plan.json').exists(),'No overwrite'
 dump(OUT/'frozen_plan.json',PLAN)
 # Source split reconstructed without any image/model forward; verify against actual native run indices.
 data=Path('/nas1/zhangzj26/dual_axis_hsi_uda/datasets/HyRANK_author_v1/HyRANK')
 source_gt=loadmat(data/'Dioni_gt_out68.mat')['map'].astype(np.int64)-1
 valid=np.nonzero(source_gt>=0);x=list(zip(*valid));y=source_gt[valid]
 train,_=train_test_split(x,train_size=.95,stratify=y,random_state=23)
 mask=np.full_like(source_gt,-1)
 ix=tuple(np.asarray(train).T);mask[ix]=source_gt[ix]
 padded=np.pad(mask,((6,6),(6,6)),mode='reflect');r,col=np.nonzero(padded>=0)
 keep=(r>6)&(r<source_gt.shape[0]+6)&(col>6)&(col<source_gt.shape[1]+6)
 source_coords=np.column_stack((r[keep],col[keep]));source_labels=padded[r[keep],col[keep]]
 assert len(source_labels)==19021 and set(source_labels)==set(range(12))
 for seed in SEEDS:
  cfg=json.loads((OUT.parent/('hyrank_A_no_extra_val_v1/seed_%d/config.json'%seed)).read_text())
  assert hashlib.sha256(source_coords.tobytes()).hexdigest()==cfg['data_protocol']['index_sha256']['0']
 np.savez(OUT/'SOURCE_ONLY_POOL.npz',coords=source_coords,labels=source_labels)
 # Isolated implementation preflight: certain/uniform/empty/missing-source-class; no global RNG drift.
 logits=np.zeros((13,12));logits[:12]=np.eye(12)*20
 global_rng=np.random.get_state()
 si,ti,p,e,a,u=pair_kernel(np.arange(12),logits,731005)
 assert len(si)==12 and np.array_equal(ti,np.arange(12)) and not a[-1]
 assert len(pair_kernel(np.arange(12),np.zeros((5,12)),731005)[0])==0
 assert len(pair_kernel(np.array([0]),np.eye(12)*20,731005)[-1])==11
 now=np.random.get_state();assert np.array_equal(global_rng[1],now[1]) and global_rng[2:]==now[2:]
 dump(OUT/'kernel_preflight.json',{'passed':True,'threshold':THRESHOLD,'target_label_argument_absent':True,
  'source_class_matches_pseudo_label':True,'uniform_entropy_is_ln12_rejected':True,
  'empty_selection_explicit':True,'missing_source_class_explicit':True,'global_rng_unchanged':True,
  'complete_training_sampler_not_implemented':True})
 rows=[];class_rows=[];selection_dir=OUT/'selection_NO_GT';selection_dir.mkdir()
 originals={}
 # Selections + source matches are all persisted before target GT is loaded.
 for seed in SEEDS:
  manifest=json.loads((BASE/('baseline_%d/full_evaluation.json'%seed)).read_text())
  for item in manifest['records']:
   assert sha(item['path'])==item['sha256'];originals[item['path']]=item['sha256']
   logits=np.load(item['path'])['logits'];assert logits.shape==(10317,12)
   si,ti,pred,entropy,accepted,unavailable=pair_kernel(source_labels,logits,seed+731000+item['epoch'])
   np.savez(selection_dir/('%d_epoch%03d.npz'%(seed,item['epoch'])),source_indices=si,target_indices=ti,
           pseudo=pred,entropy=entropy,accepted=accepted,unavailable=unavailable)
 dump(OUT/'ALL_SELECTIONS_AND_SOURCE_PAIRS_LOCKED_NO_TARGET_GT.json',
      {str(p):sha(p) for p in selection_dir.glob('*.npz')})
 target_gt=loadmat(data/'Loukia_gt_out68.mat')['map'].astype(np.int64)-1
 source_prior=np.bincount(source_labels,minlength=12)/len(source_labels)
 for seed in SEEDS:
  coords=np.load(BASE/('baseline_%d/full_indices.npy'%seed));truth=target_gt[coords[:,0]-6,coords[:,1]-6]
  assert np.all(truth>=0) and len(truth)==10317
  for epoch in PLAN['epochs']:
   z=np.load(selection_dir/('%d_epoch%03d.npz'%(seed,epoch)))
   pred,entropy,accepted=z['pseudo'],z['entropy'],z['accepted'];correct=pred==truth
   target_truth_prior=np.bincount(truth[accepted],minlength=12)/accepted.sum()
   missing=list(set(range(12))-set(pred[accepted]))
   rows.append({'seed':seed,'epoch':epoch,'retained_count':int(accepted.sum()),'coverage_percent':float(accepted.mean()*100),
      'unfiltered_OA_percent':float(correct.mean()*100),'retained_pseudo_precision_percent':pct(accepted,correct),
      'rejected_count':int((~accepted).sum()),'rejected_accuracy_percent':pct(~accepted,correct),
      'incorrect_prediction_retained_percent':pct(~correct,accepted),
      'mean_entropy':float(entropy.mean()),'missing_retained_pseudo_classes':[i+1 for i in sorted(missing)],
      'declared_source_match_to_true_target_class_percent':pct(accepted,correct),
      'random_independent_source_match_expectation_percent':float(np.dot(source_prior,target_truth_prior)*100)})
   for k in range(12):
    true_mask=truth==k;selected=accepted&(pred==k)
    class_rows.append({'seed':seed,'epoch':epoch,'class':k+1,'source_train_count':int((source_labels==k).sum()),
       'target_true_count':int(true_mask.sum()),'target_true_retained_percent':pct(true_mask,accepted),
       'target_true_retained_correct_percent':pct(true_mask,accepted&correct),
       'selected_pseudo_class_count':int(selected.sum()),'selected_pseudo_class_precision_percent':pct(selected,correct)})
 for name,data_rows in [('screening_by_epoch.csv',rows),('screening_per_class.csv',class_rows)]:
  with (OUT/name).open('w') as f:
   w=csv.DictWriter(f,fieldnames=list(data_rows[0]));w.writeheader();w.writerows(data_rows)
 aggregates=[]
 for epoch in PLAN['highlight_epochs']:
  rr=[r for r in rows if r['epoch']==epoch]
  aggregates.append({'epoch':epoch,**{key:{'mean':float(np.mean([r[key] for r in rr])),
      'sample_std':float(np.std([r[key] for r in rr],ddof=1)),'per_seed':[r[key] for r in rr]}
   for key in ('coverage_percent','retained_pseudo_precision_percent','incorrect_prediction_retained_percent','random_independent_source_match_expectation_percent')}})
 dump(OUT/'summary.json',{'plan':PLAN,'rows':rows,'class_rows':class_rows,'aggregate':aggregates,
      'source_split_random_state23_verified_all_seeds':True,'no_training':True,'Ours_paused':True})
 assert all(sha(path)==h for path,h in originals.items())
 dump(OUT/'original_prediction_hashes_unchanged.json',originals)
 lines=['Official BiDA Eq9 screening + source-pairing kernel: read-only feasibility audit',
   'Fixed cutoff0.5ln12=%.9f. All20 existing epoch endpoints, no target-selected stage/threshold.'%THRESHOLD,
   'All selections and source pairs saved BEFORE loading target GT; GT only scores frozen selections.',
   'Source split uses release hardcoded random_state23, not CLI seed; actual19021 source indices verified for all3 seeds.',
   'Kernel picks labeled source of target pseudo-class with private RNG, no target label argument.',
   'This is only a variable-size pairing kernel, not a completed next-iteration training sampler.',
   'Unspecified training policies: initial pseudo-label generation; next-iteration target cache; no retained target batch; filling128 samples;',
   'source CE class-frequency changes; refresh cadence and probability train/eval mode; source pool vs current minibatch matching.',
   'Kernel defaults (global source pool, sampling with replacement) are declared reconstruction choices, not author-confirmed implementation.',
   '', 'Highlighted fixed epochs:']
 for r in aggregates:lines.append(json.dumps(r))
 lines += ['','Limits: all probabilities are frozen eval-mode outputs. This cannot establish actual training-time pseudo reliability or OA gain from pairing.',
  'Existing MMD argument repair continues independently; no augmentation/MMD changes are mixed into this kernel.',
  'No Ours, new pairing training, hyperparameter sweep, or target-GT-selected implementation was launched.']
 (OUT/'RESULTS.txt').write_text('\n'.join(lines)+'\n')
 shutil.copy2(Path(__file__),OUT/'diagnose.py')
 link=ROOT/'results/bida_hyrank_eq9_diagnostic_v1'
 if not link.exists():link.symlink_to(OUT,target_is_directory=True)
 print(json.dumps(aggregates,indent=2))

if __name__=='__main__':main()

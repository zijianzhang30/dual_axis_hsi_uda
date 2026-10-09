"""Report reused release evidence and count-matched evaluation. No tuning/launching."""
import ast
import csv
import json
import math
from pathlib import Path
import numpy as np
from bs4 import BeautifulSoup
import study as s

def stats(rows,key):return {'mean':float(np.mean([r[key] for r in rows])),'sample_std':float(np.std([r[key] for r in rows],ddof=1))}
def rows_from(records,arm,seed):
 output=[]
 for protocol,field in [('native','metrics'),('paper_full','paper_full_metrics')]:
  fixed=records[-1];best=max(records,key=lambda r:r[field]['oa_percent'])
  for endpoint,item in [('fixed200',fixed),('target_best_diagnostic',best)]:
   zero=[i+1 for i,x in enumerate(item[field]['per_class_percent']) if x==0]
   output.append({'arm':arm,'seed':seed,'protocol':protocol,'endpoint':endpoint,'selected_epoch':item['epoch'],
      **{k:item[field][k] for k in ['n','oa_percent','aa_percent','kappa','per_class_percent']},
      'zero_recall_classes':zero})
 return output

def main():
 s.verify_originals();paper=BeautifulSoup((s.OUT/'paper.html').read_text(),'html.parser')
 counts=[];reported_recalls=[];reported_oa=None
 for row in paper.find(id='S3.T5').select('tr'):
  cells=[x.get_text(' ',strip=True) for x in row.find_all(['td','th'])]
  if cells and cells[0].isdigit():counts.append(int(cells[-1]))
 for row in paper.find(id='S3.T14').select('tr'):
  cells=[x.get_text(' ',strip=True) for x in row.find_all(['td','th'])]
  if cells and cells[0].isdigit():reported_recalls.append(float(cells[-1]))
  if cells and cells[0].startswith('OA'):reported_oa=float(cells[-1])
 assert sum(counts)==10317 and len(reported_recalls)==12 and reported_oa==68.89
 weighted=float(np.dot(counts,reported_recalls)/sum(counts))
 anomalies={'source':'arXiv v1 TablesV/XIV; final publisher text unverified',
     'if_CA_is_recall_and_counts_are_same_evaluation_cohort':True,
     'weighted_OA_from_class_CA':weighted,'reported_OA':reported_oa,'difference_pp':weighted-reported_oa,
     'rounding_bound_pp':.005,
     'interpretation':'requires CA definition, exact evaluation mask and table aggregation; not proof of paper error'}
 native_source=ast.parse((s.OUT/'released_train_pipeline.py').read_text())
 fn=next(n for n in native_source.body if isinstance(n,ast.FunctionDef) and n.name=='train')
 names=sorted({n.id for n in ast.walk(fn) if isinstance(n,ast.Name)})
 assert not any(x in names for x in ('pseudo_labels','uncertainty','entropy_threshold'))
 evidence=[
  {'item':'version','paper':'author arXiv v1; IEEE DOI known but full publisher text blocked','release':s.COMMIT,'status':'pinned archive; local server copy has no .git; core SHA/AST verified against archived commit'},
  {'item':'data','paper':'TableV scene dimensions176 bands12 common classes and counts','release':'author out68 MAT files; raw counts match; normalized full arrays match existing independent audit','status':'verified; reuse data_checks.json rather than repeating full cube experiment'},
  {'item':'normalization','paper':'specific normalization not established from text','release':'per-pixel spectral L2 normband; explicit reshaped return','status':'release-only setting, not evidence of paper protocol identity'},
  {'item':'source_sampling','paper':'train fraction/indices/seed count not established','release':'stratified random95% source;19021 train1002 val; source duplication re_ratio1','status':'missing HyRANK-specific paper config'},
  {'item':'budget','paper':'epoch/batch/optimizer recipe not established','release':'200 epochs,bs128,SGD .01,momentum0,decay0; zip gives80 updates/epoch,not149 source batches','status':'generic release defaults; no claimed paper-specific optimum'},
  {'item':'patch_tokens','paper':'13x13,4 tokens; eight attention heads','release':'13x13,4 tokens,dim64,depth3,8 heads','status':'patch/token/head match; dim/depth unconfirmed in paper'},
  {'item':'extractor','paper':'sectionII-A mentions MaxPool blocks','release':'Conv3D-BN-ReLU,Conv2D-BN-ReLU; no MaxPool','status':'structural text/code difference; unspecified pooling settings prevent unique repair'},
  {'item':'classwise_pairing','paper':'sectionII-B same-class pairs using target probabilities, entropy cutoff0.5lnC','release':'independent shuffled loaders zip; no pseudo-label selection or classwise sampler on executed path','status':'missing central implementation; C=12 cutoff1.242453, not used in release'},
  {'item':'pairing_details','paper':'next-iteration pairing described','release':'no startup/cache/refill/missing-source-class/replacement rules','status':'would require invented policies; no automatic reconstructed algorithm'},
  {'item':'augmentation','paper':'sectionII-C independent perturbations; Gaussian sigma.05; TableIX usesRandomRC+Gauss','release':'data_aug True means random flip/rotation; radiation/mixture flags False; teacher sees same augmented input as student','status':'independent view/crop/Gaussian path absent; crop/resize distribution unspecified'},
  {'item':'MMD','paper':'Eq14 compares distinct intra-domain and coupled token representations','release':'logit MMD after100; coupled second term compares a tensor to itself; average divided by2','status':'self term zero loss/gradient; only distinct coupled argument tested as single factor; feature placement not changed'},
  {'item':'distillation','paper':'Eq10/11 cross-branch soft supervision','release':'coupled logits softmax->CE against self logits; teacher-side soft distribution not detached','status':'formula broadly corresponds; stop-gradient policy unspecified, not automatically called a bug'},
  {'item':'consistency','paper':'teacher/student consistency with different noises','release':'probabilities passed to softmax_mse_loss, which softmaxes again; normalization byclasses andbatch','status':'verified double-softmax behavior; not altered'},
  {'item':'EMA','paper':'EMA teacher, student inference','release':'independently initialized teacher;alpha cap.999;parameters only EMA;teacher remains train and updates own BN buffers','status':'EMA detail underspecified in paper; behavior preserved'},
  {'item':'BN','paper':'BN/running-stat configuration not specified','release':'native source/target sequential BatchNorm, momentum.1; eval running stats','status':'preserved; no Joint BN or Ours tokenizer'},
  {'item':'evaluation','paper':'TableV gives10317 Loukia pixels; CA/OA/Kappa; selection/runs/mask not fully specified','release':'native strict boundary condition retains10287; student target-self inference every10; saves target-OA best','status':'full10317 count-matched evaluation added separately; cannot establish exact unpublished paper mask'}
 ]
 s.dump(s.OUT/'protocol_audit.json',{'items':evidence,'paper_table_consistency':anomalies,
    'executed_train_names':names,'prior_numerical_equivalence_reused':True,
    'new_controlled_factor':s.PLAN['single_training_factor'],'Ours_paused':True})
 results=[]
 for seed in s.SEEDS:
  full=json.loads((s.OUT/('baseline_%d/full_evaluation.json'%seed)).read_text())
  records=[{'epoch':r['epoch'],'metrics':r['native_metrics'],'paper_full_metrics':r['paper_full_metrics']} for r in full['records']]
  results.extend(rows_from(records,'released',seed))
 all_repair=all((s.OUT/('repair_%d/frozen_manifest.json'%seed)).exists() for seed in s.SEEDS)
 if all_repair:
  for seed in s.SEEDS:
   m=json.loads((s.OUT/('repair_%d/frozen_manifest.json'%seed)).read_text());assert m['completed_epochs']==200
   for v in m['evaluations']:
    for kind in ('prediction','full_prediction'):assert s.sha(v[kind+'_path'])==v[kind+'_sha256']
   results.extend(rows_from(m['evaluations'],'MMD_argument_only',seed))
 aggregate=[];paired=[]
 for arm in sorted({r['arm'] for r in results}):
  for protocol in ('native','paper_full'):
   for endpoint in ('fixed200','target_best_diagnostic'):
    rr=[r for r in results if (r['arm'],r['protocol'],r['endpoint'])==(arm,protocol,endpoint)]
    aggregate.append({'arm':arm,'protocol':protocol,'endpoint':endpoint,
        **{k:stats(rr,k) for k in ('oa_percent','aa_percent','kappa')},'epochs':[r['selected_epoch'] for r in rr]})
 if all_repair:
  for seed in s.SEEDS:
   for protocol in ('native','paper_full'):
    for endpoint in ('fixed200','target_best_diagnostic'):
     get=lambda arm:next(r for r in results if (r['arm'],r['seed'],r['protocol'],r['endpoint'])==(arm,seed,protocol,endpoint))
     left,right=get('MMD_argument_only'),get('released')
     paired.append({'seed':seed,'protocol':protocol,'endpoint':endpoint,
       **{k:left[k]-right[k] for k in ('oa_percent','aa_percent','kappa')}})
 s.dump(s.OUT/'summary.json',{'status':'controlled_repair_complete' if all_repair else 'audit_and_existing_checkpoint_eval_complete_repair_pending',
    'rows':results,'aggregate':aggregate,'paired':paired,'paper_OA_separate':68.89,'paper_metric_anomaly':anomalies,
    'evaluation_boundary_max_difference_pp':30/10317*100,'no_followup_factors':True,'no_Ours_training':True})
 with (s.OUT/'fixed_and_target_best.csv').open('w') as f:
  rr=[{k:r[k] for k in ['arm','seed','protocol','endpoint','selected_epoch','n','oa_percent','aa_percent','kappa','zero_recall_classes']} for r in results]
  w=csv.DictWriter(f,fieldnames=list(rr[0]));w.writeheader();w.writerows(rr)
 with (s.OUT/'per_class_recall.csv').open('w') as f:
  rr=[{'arm':r['arm'],'seed':r['seed'],'protocol':r['protocol'],'endpoint':r['endpoint'],'class':i+1,'recall_percent':v} for r in results for i,v in enumerate(r['per_class_percent'])]
  w=csv.DictWriter(f,fieldnames=list(rr[0]));w.writeheader();w.writerows(rr)
 lines=['Official BiDA HyRANK paper reproduction audit; Ours new experiments paused.',
    'Locked commit: '+s.COMMIT,
    'Author preprint arXiv2507.02268v1 used for method/table comparison; IEEE final full-text identity not verified.',
    'Existing actual-main/no-extra-validation trajectories reused, not rerun.',
    'Release-specific data loader runtime compatibility fixes are disclosed separately; all old code/results hashes unchanged.',
    '', 'Protocol audit:']
 for r in evidence:lines.append(json.dumps(r,ensure_ascii=False))
 lines.extend(['','Existing native vs TableV-full-mask endpoint results:'])
 for r in aggregate:lines.append(json.dumps(r))
 lines.extend(['','Critical findings:',
    'The count-matched full10317 mask closes essentially none of the~9.55pp target-best gap; any30-pixel boundary difference is bounded by0.2908pp.',
    'Missing uncertainty-based classwise pairing and independent ARS input views mean the released executed path does not reproduce central paper-described mechanisms.',
    'The second release MMD term is identically zero with zero gradient. A single-argument repair is isolated; it does not restore feature-level MMD or missing pairing/augmentation.',
    'Other behaviors (EMA initialization/buffers,distill detach,BN,budget) are documented but not declared causal errors where paper detail is missing.',
    'Reported class CA weighted byTableV counts yields71.4428%, not68.89%, if CA is recall on that same cohort. Exact CA definition/mask/table aggregation needs confirmation.',
    '', 'Controlled repair status: '+('completed; paired results below' if all_repair else 'preflight passed; three fixed-seed200-epoch jobs are the only planned new training. Results pending.'),
    json.dumps(paired,indent=2),
    '', 'Author information required to faithfully reproduce:',
    'Actual HyRANK experiment config/command/seeds/train budget/source indices; exact evaluation mask and CA definition;',
    'classwise pseudo-label sampler and initialization/cache/refill/no-class cases;',
    'independent teacher/student augmentation pipeline including crop/resize settings;',
    'model/feature-MMD implementation and pretrained student/teacher checkpoints;',
    'EMA/BN initialization/update policy; whether publisher version changes these descriptions.',
    'No message sent to authors. No target-GT tuning, hyperparameter sweep, automatic new factors, or Ours training.'])
 (s.OUT/'AUDIT_REPORT.txt').write_text('\n'.join(lines)+'\n')
 print(json.dumps({'status':all_repair,'aggregate':aggregate},indent=2),flush=True)

if __name__=='__main__':main()

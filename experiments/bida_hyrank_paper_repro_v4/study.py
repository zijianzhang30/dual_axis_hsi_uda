"""Independent official-BiDA audit and one-factor controlled repair. No Ours imports/models."""
import ast
import copy
import difflib
import hashlib
import importlib.util
import json
import os
import random
import shutil
import sys
import urllib.request
from pathlib import Path
import numpy as np
import torch

ROOT=Path('/home/zhangzj26/dual_axis_hsi_uda')
OUT=Path('/nas1/zhangzj26/dual_axis_hsi_uda/results/bida_hyrank_paper_repro_v4')
PIN=OUT.parent/'hyrank_repro_audit_v2/dependency_snapshot/official_BiDA_pinned'
LIVE=Path('/home/zhangzj26/IEEE_TCSVT_BiDA')
BASE=OUT.parent/'hyrank_A_no_extra_val_v1'
COMMIT='66a92b542ffa1df3c8b628af3d8bd001b799f1f6'
SEEDS=(2100,2101,2102)
PLAN={'purpose':'official BiDA HyRANK audit and minimal release repair, not Ours',
 'commit':COMMIT,'paper':'https://arxiv.org/html/2507.02268v1',
 'journal_final':'10.1109/TCSVT.2025.3586282; full publisher text inaccessible, version identity not assumed',
 'seeds':list(SEEDS),'epochs':200,'source_ratio':.95,'patch':13,'batch':128,'lr':.01,
 'optimizer':'native persistent SGD, momentum0,weight_decay0,no scheduler',
 'loss_weights':{'lambda1':.1,'lambda2':1.,'EMA_decay':.999},
 'single_training_factor':'second MMD argument: MMD(out_x_fusion_src,out_x_fusion_src) -> MMD(out_x_fusion_src,out_x_fusion)',
 'unchanged':'logit MMD placement,half coefficient,epoch>100 activation,all other losses,teacher initialization/modes,BN,augmentation,independent sampling and200-epoch budget',
 'baseline':'reuse three completed native-equivalent runs; every10 trajectory matches actual main.py output',
 'evaluation':'native10287 and full original-scene10317 separately; fixed200 primary, OA-best every10 diagnostic only; AA/Kappa at that same OA-best endpoint',
 'GT':'target class values only evaluation/offline audit, not training, thresholds, hyperparameters or further factor choice',
 'not_exact_paper_reproduction':'missing class matching/uncertainty/input-view implementation remains; this is one-factor release repair, not full recovery',
 'no_sweep':True,'no_automatic_followup_factors':True,'Ours_training_paused':True}

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def dump(path,obj):Path(path).write_text(json.dumps(obj,indent=2)+'\n')
def module(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def prepare():
 assert not (OUT/'frozen_plan.json').exists(),'Never overwrite a study'
 assert (PIN/'.git/HEAD').read_text().strip()==COMMIT
 dump(OUT/'frozen_plan.json',PLAN)
 originals={}
 for base in [LIVE/'main.py',LIVE/'train_pipeline.py',LIVE/'README.md']+list((LIVE/'models').rglob('*.py'))+list((LIVE/'utils').rglob('*.py'))+list((LIVE/'loss').rglob('*.py')):
  originals[str(base)]=sha(base)
 for seed in SEEDS:
  f=BASE/('seed_%d'%seed);m=json.loads((f/'frozen_manifest.json').read_text());assert m['completed_epochs']==200
  originals[str(f/'frozen_manifest.json')]=sha(f/'frozen_manifest.json');originals[str(f/'history.jsonl')]=sha(f/'history.jsonl')
  for item in m['evaluations']:
   for kind in ('prediction','checkpoint'):
    p=item[kind+'_path'];assert sha(p)==item[kind+'_sha256'];originals[p]=sha(p)
 for p in (LIVE/'utils/dataset.py',ROOT/'experiments/official_bench_v1/common.py',ROOT/'experiments/hyrank_A_no_extra_val_v1/control.py'):
  originals[str(p)]=sha(p)
 for p in Path('/nas1/zhangzj26/dual_axis_hsi_uda/datasets/HyRANK_author_v1/HyRANK').glob('*.mat'):originals[str(p)]=sha(p)
 dump(OUT/'original_files_manifest.json',originals)
 runtime=OUT/'runtime_snapshot';runtime.mkdir()
 for name in ['models','utils','loss']:shutil.copytree(LIVE/name,runtime/name,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
 for name in ['main.py','train_pipeline.py','README.md','requirements.txt']:shutil.copy2(LIVE/name,runtime/name)
 source=(runtime/'train_pipeline.py').read_text();old='out_x_fusion_src, out_x_fusion_src';new='out_x_fusion_src, out_x_fusion'
 assert source.count(old)==1
 (OUT/'released_train_pipeline.py').write_text(source)
 patched=source.replace(old,new);(runtime/'train_pipeline.py').write_text(patched)
 (OUT/'single_factor_MMD.patch').write_text(''.join(difflib.unified_diff(source.splitlines(True),patched.splitlines(True),fromfile='released/train_pipeline.py',tofile='one_factor/train_pipeline.py')))
 (OUT/'dataset_runtime_compatibility.patch').write_text(''.join(difflib.unified_diff((PIN/'utils/dataset.py').read_text().splitlines(True),(LIVE/'utils/dataset.py').read_text().splitlines(True),fromfile='pinned/utils/dataset.py',tofile='already_audited_runtime/utils/dataset.py')))
 # Freeze a loader/build helper, preserving paths explicitly rather than using its new __file__ depth.
 helper=(ROOT/'experiments/official_bench_v1/common.py').read_text()
 helper=helper.replace('ROOT=Path(__file__).resolve().parents[2]',"ROOT=Path('/home/zhangzj26/dual_axis_hsi_uda')")
 helper=helper.replace("BIDA=Path('/home/zhangzj26/IEEE_TCSVT_BiDA')",'BIDA=Path('+repr(str(runtime))+')')
 (OUT/'frozen_official_helper.py').write_text(helper)
 shutil.copy2(ROOT/'experiments/hyrank_A_no_extra_val_v1/control.py',OUT/'control_snapshot.py')
 for name,url in [('paper.html',PLAN['paper']),('github_commit.json','https://api.github.com/repos/YuxiangZhang-BIT/IEEE_TCSVT_BiDA/commits/'+COMMIT)]:
  req=urllib.request.Request(url,headers={'User-Agent':'BiDA-reproduction-audit'})
  try:(OUT/name).write_bytes(urllib.request.urlopen(req,timeout=30).read())
  except Exception as e:dump(OUT/(name+'.download_failure.json'),{'url':url,'error':str(e)})
 source_audit=json.loads((OUT.parent/'hyrank_code_audit_v3/AUDIT.json').read_text())
 dump(OUT/'reused_prior_audit.json',source_audit)
 dump(OUT/'Ours_PAUSED.json',{'Ours_training_paused_at_user_request':True,'running_Ours_jobs_seen':0,'no_new_Ours_queue':True})
 link=ROOT/'results/bida_hyrank_paper_repro_v4'
 if not link.exists():link.symlink_to(OUT,target_is_directory=True)

def environment():
 c=module('frozen_official_release_helper',OUT/'frozen_official_helper.py')
 # Reuse instrumentation definitions only, with frozen official dependencies supplied explicitly.
 tree=ast.parse((OUT/'control_snapshot.py').read_text())
 nodes=[n for n in tree.body if isinstance(n,(ast.ClassDef,ast.FunctionDef))]
 ns={'c':c,'ROOT':ROOT,'OUT':OUT,'SEEDS':SEEDS,'ast':ast,'hashlib':hashlib,'json':json,
     'pickle':__import__('pickle'),'random':random,'sys':sys,'Path':Path,'np':np,'torch':torch,'F':torch.nn.functional}
 exec(compile(ast.Module(body=nodes,type_ignores=[]),str(OUT/'control_snapshot.py'),'exec'),ns)
 return c,ns

def verify_originals():
 for p,h in json.loads((OUT/'original_files_manifest.json').read_text()).items():assert sha(p)==h,p

def full_geometry(dataset):
 mask=dataset.label>=0;x,y=np.nonzero(mask);r=6
 valid=(x>=r)&(x<dataset.data.shape[0]-r)&(y>=r)&(y<dataset.data.shape[1]-r)
 coords=np.column_stack((x[valid],y[valid]));assert len(coords)==10317
 native=dataset.indices;lookup={tuple(v):i for i,v in enumerate(coords)}
 native_into_full=np.asarray([lookup[tuple(v)] for v in native]);present=np.zeros(len(coords),bool);present[native_into_full]=True
 missing=np.flatnonzero(~present);assert len(native)==10287 and len(missing)==30
 images=[]
 for row,col in coords[missing]:
  patch=dataset.data[row-r:row+r+1,col-r:col+r+1]
  images.append(torch.from_numpy(np.array(patch.transpose(2,0,1),dtype=np.float32,copy=True)).unsqueeze(0))
 labels=np.asarray([dataset.label[row,col] for row,col in coords],dtype=np.int64)
 return coords,native_into_full,missing,torch.stack(images),labels

@torch.inference_mode()
def extra_logits(model,images):
 assert not model.training
 # Direct method bypasses top-level diagnostic hooks, not model computations.
 x=images.cuda();n=len(x);x=torch.cat([x,x[-1:].expand(128-n,*x.shape[1:])])
 return model.forward(x,x)[1][:n].cpu().numpy()

def baseline_full_eval(seed):
 torch.set_num_threads(2);c,ns=environment();c.seed_worker(seed);args=c.options(seed);args.device='0'
 values,audit=ns['loaders'](args);ds=values[-1].dataset;coords,mapping,missing,images,labels=full_geometry(ds)
 folder=OUT/('baseline_%d'%seed);folder.mkdir(exist_ok=False)
 model,ema,initial=c.build(args,'A');del ema;model=model.cuda().eval()
 old=json.loads((BASE/('seed_%d'%seed)/'frozen_manifest.json').read_text());records=[]
 for item in old['evaluations']:
  saved=torch.load(item['checkpoint_path'],map_location='cpu',weights_only=False);assert saved['epoch']==item['epoch']
  model.load_state_dict(saved['model'],strict=True);before=c.helper.state_hash(model)
  z=np.load(item['prediction_path']);full=np.empty((10317,12),np.float32);full[mapping]=z['logits'];full[missing]=extra_logits(model,images)
  assert c.helper.state_hash(model)==before and all(p.grad is None for p in model.parameters())
  path=folder/('full_predictions_%03d.npz'%item['epoch']);np.savez_compressed(path,logits=full,predictions=full.argmax(1))
  # Target GT is used only to score saved outputs from a frozen existing checkpoint.
  metrics=c.helper.metrics(labels,full.argmax(1),full,12)
  records.append({'epoch':item['epoch'],'native_metrics':item['metrics'],'paper_full_metrics':metrics,
       'path':str(path),'sha256':sha(path),'native_logits_reused_without_change':True})
 np.save(folder/'full_indices.npy',coords)
 dump(folder/'full_evaluation.json',{'seed':seed,'full_count':10317,'native_count':10287,'missing_count':30,
       'OA_boundary_difference_bound_pp':30/10317*100,'records':records,
       'fixed200':records[-1],'OA_best_diagnostic':max(records,key=lambda r:r['paper_full_metrics']['oa_percent'])})
 print('BASELINE_FULL_EVAL_COMPLETE seed=%d'%seed,flush=True)

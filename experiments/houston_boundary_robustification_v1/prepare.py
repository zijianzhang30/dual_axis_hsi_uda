import csv,json,shutil
import numpy as np
import torch
from study import *
from boundary import perturbed,loss

def main():
 OUT.mkdir(exist_ok=False);(OUT/'logs').mkdir();(OUT/'weights').mkdir();(OUT/'experiment_code').mkdir()
 gate=json.loads((GEOM/'ALL_BOUNDARY_STRESS_FROZEN.json').read_text())
 assert not gate['target_GT_opened']
 dependencies={};rows=[]
 rel='unlabeled_pair_statistics_per_seed.csv';assert sha(GEOM/rel)==gate['files_sha256'][rel]
 stats=list(csv.DictReader((GEOM/rel).open()))
 for seed in SEEDS:
  ss=[r for r in stats if int(r['seed'])==seed];assert len(ss)==21
  activity=np.array([float(r['competition_n'])/52901 for r in ss]);stress=np.array([float(r['stress_source_target_median_ratio']) for r in ss]);assert np.isfinite(stress).all() and (stress>0).all() and np.isclose(activity.sum(),1)
  for arm,raw in [('B1',activity),('B2',activity*stress)]:
   q=raw/raw.sum();matrix=np.zeros((7,7))
   for j,(a,b) in enumerate(PAIRS):
    assert ss[j]['pair']==f'C{a+1}/C{b+1}';matrix[a,b]=matrix[b,a]=q[j]
    rows.append({'seed':seed,'arm':arm,'pair':ss[j]['pair'],'activity':float(activity[j]),'stress':float(stress[j]),'raw_weight':float(raw[j]),'normalized_pair_weight':float(q[j])})
   np.savez_compressed(OUT/'weights'/f'{arm}_{seed}.npz',activity=activity,stress=stress,raw=raw,normalized=q,matrix=matrix)
  for name in ['config.json','history.jsonl','frozen_manifest.json','predictions_200.npz','checkpoints/epoch_200.pth']:
   p=BASE/'houston'/f'C_{seed}'/name;dependencies[str(p)]=sha(p)
  mf=json.loads((BASE/'houston'/f'C_{seed}'/'frozen_manifest.json').read_text())
  cp=mf['checkpoints'][-1];assert cp['epoch']==200 and sha(cp['path'])==cp['sha256']
  assert sha(mf['prediction']['path'])==mf['prediction']['sha256']
 for p in list(BASE.glob('*.py'))+list((BASE/'snapshot').rglob('*.py')):dependencies[str(p)]=sha(p)
 dependencies[str(GEOM/rel)]=sha(GEOM/rel);dependencies[str(GEOM/'ALL_BOUNDARY_STRESS_FROZEN.json')]=sha(GEOM/'ALL_BOUNDARY_STRESS_FROZEN.json')
 with (OUT/'pair_weights.csv').open('x',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 # Analytic/unit preflight, independent synthetic data, no training runs/GT.
 with torch.random.fork_rng(devices=[]):
  torch.manual_seed(987);head=torch.nn.Linear(64,7).double();z=torch.randn(28,64,dtype=torch.double,requires_grad=True);y=torch.arange(28)%7
  weights=torch.ones(7,7,dtype=torch.double)/21;weights.fill_diagonal_(0)
  zp,m,n=perturbed(z,y,head);after=Flinear(zp,head);mask=~torch.nn.functional.one_hot(y,7).bool()
  own=after.gather(2,y[:,None,None].expand(-1,7,1)).squeeze(-1)-after.diagonal(dim1=1,dim2=2)
  error=float((own[mask]-.5*m[mask]).abs().max());assert error<1e-12
  actual=(zp-z[:,None,:]).norm(dim=-1);assert torch.allclose(actual[mask],.5*m[mask].abs()/n[mask])
  l,_,_,_=loss(z,y,head,weights);manual=0
  for a,b in PAIRS:
   idx=(y==a)|(y==b);competitor=torch.where(y[idx]==a,b,a)
   ce=torch.nn.functional.cross_entropy(after[idx,competitor],y[idx]);manual+=ce/21
  assert torch.allclose(l,manual,atol=1e-12);l.backward();assert z.grad.abs().sum()>0 and head.weight.grad.abs().sum()>0
 dump(OUT/'preflight.json',{'passed':True,'perturbation_same_side_half_margin_max_error':error,'norm_and_vectorized_pair_loss_verified':True,'gradients_to_feature_and_live_head':True})
 for p in HERE.iterdir():
  if p.is_file():shutil.copyfile(p,OUT/'experiment_code'/p.name);dependencies[str(p)]=sha(p)
 for p in OUT.rglob('*'):
  if p.is_file():dependencies[str(p)]=sha(p)
 dump(OUT/'PRETRAIN_WEIGHTS_CODE_FROZEN.json',{'target_GT_used_to_construct_weights':False,'no_GT_formula_pair_threshold_selection':True,'all21_pairs':True,'files_sha256':dependencies})
 print('Pretraining weights/code frozen; synthetic geometry/loss checks passed.',flush=True)

def Flinear(z,head):return torch.nn.functional.linear(z,head.weight,head.bias)
if __name__=='__main__':main()

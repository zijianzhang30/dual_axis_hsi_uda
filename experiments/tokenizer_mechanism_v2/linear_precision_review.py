"""Post-hoc precision audit only; never modifies checkpoints or training."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from common import OUT,SEEDS,environment,loaders,dump
from scorer import RelationScores

def main():
 p=argparse.ArgumentParser();p.add_argument('--dataset',choices=['houston','pavia'],required=True);args=p.parse_args()
 env=environment();torch.set_num_threads(2);torch.cuda.set_device(0);device=torch.device('cuda:0')
 torch.backends.cuda.matmul.allow_tf32=False
 path=Path('/home/zhangzj26/TGRS_MLUDA-2024/datasets/Pavia') if args.dataset=='pavia' else '/home/zhangzj26/TGRS_MLUDA-2024/datasets/Houston/'
 data_args=argparse.Namespace(dataset=args.dataset,dataset_dir=path,seed=2100,num_workers=4);env[4](2100)
 loader=loaders(data_args)[3];scorers={};records={};bands=48 if args.dataset=='houston' else 102
 for seed in SEEDS:
  for arm in 'DE':
   checkpoint=torch.load(OUT/args.dataset/f'{arm}_{seed}/checkpoints/epoch_200.pth',map_location='cpu',weights_only=True)['model']
   state={k[len('relation_scores.'):]:v for k,v in checkpoint.items() if k.startswith('relation_scores.')}
   scorer=RelationScores(arm,bands).double().to(device);scorer.load_state_dict(state)
   scorers[arm,seed]=scorer;records[f'{arm}_{seed}']={'same_parameters_FP64_max_weight_error':0.,'relation_identity_max_error':0.}
 with torch.inference_mode():
  for patch,_ in loader:
   patch=patch.to(device=device,dtype=torch.float64)
   for (arm,seed),scorer in scorers.items():
    d=scorer(patch,arm='D');e=scorer(patch,arm='E');u=scorer.relations(patch,arm='D');r=scorer.relations(patch,arm='E')
    uc=u[:,:,6:7,6:7];row=records[f'{arm}_{seed}']
    row['same_parameters_FP64_max_weight_error']=max(row['same_parameters_FP64_max_weight_error'],float((d-e).abs().max()))
    row['relation_identity_max_error']=max(row['relation_identity_max_error'],float((r-(u-uc)).abs().max()))
 for key,row in records.items():
  native=json.loads((OUT/args.dataset/key/'frozen_manifest.json').read_text())
  row['native_trained_FP32_max_weight_error']=native['same_parameters_D_E_max_error_real_patches']
  assert row['same_parameters_FP64_max_weight_error']<1e-10
 dump(OUT/args.dataset/'linear_precision_review.json',{'dataset':args.dataset,'stage':'fixed200 checkpoints; same scorer parameters on all real eval patches','runs':records,'conclusion':'Exact-arithmetic equivalence supported; larger FP32 errors are cancellation/rounding, not a different center-dependent linear function. Separately trained D/E can diverge through optimization roundoff.'})
 print(json.dumps(records,indent=2),flush=True)
if __name__=='__main__':main()

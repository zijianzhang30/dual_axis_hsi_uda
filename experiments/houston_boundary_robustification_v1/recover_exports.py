"""Offline source-export recovery only; never train/score target/recompute weights."""
import json,shutil
from argparse import Namespace
import numpy as np
import torch
import study as s
from train import source_margin_export

def main():
 torch.set_num_threads(2)
 audit=s.OUT/'export_recovery_audit';audit.mkdir(exist_ok=False)
 old=json.loads((s.OUT/'PRETRAIN_WEIGHTS_CODE_FROZEN.json').read_text());shutil.copyfile(s.OUT/'PRETRAIN_WEIGHTS_CODE_FROZEN.json',audit/'ORIGINAL_PRETRAIN_WEIGHTS_CODE_FROZEN.json')
 immutable={}
 for seed in s.SEEDS:
  f=s.OUT/f'B0_{seed}'
  for p in f.rglob('*'):
   if p.is_file():immutable[str(p)]=s.sha(p)
  source=np.load(f/'source_frozen.npz');hp=np.load(f/'frozen_head.npz');head=torch.nn.Linear(64,7)
  with torch.no_grad():head.weight.copy_(torch.as_tensor(hp['weight']));head.bias.copy_(torch.as_tensor(hp['bias']))
  logits=source['rawZ']@hp['weight'].T+hp['bias'];assert np.array_equal(logits.argmax(1),source['predictions']) and np.abs(logits-source['logits']).max()<2e-5
  source_margin_export(Namespace(nn1=head),torch.device('cpu'),f,source['rawZ'],source['logits'],source['labels'],source['train_mask'])
  state=torch.load(f/'checkpoints/epoch_200.pth',map_location='cpu',weights_only=True)['model'];arch=torch.load(s.BASE/'houston'/f'C_{seed}'/'checkpoints/epoch_200.pth',map_location='cpu',weights_only=True)['model']
  history=[json.loads(x) for x in (f/'history.jsonl').read_text().splitlines()];assert len(history)==200 and sum(x['steps'] for x in history)==3600
  current=np.load(f/'predictions_200.npz');archp=np.load(s.BASE/'houston'/f'C_{seed}'/'predictions_200.npz')
  replay={'state_exact':all(torch.equal(v,arch[k]) for k,v in state.items()),'max_state_error':max(float((v-arch[k]).abs().max()) for k,v in state.items()),
   'predictions_exact':bool(np.array_equal(current['predictions'],archp['predictions'])),'max_logit_error':float(np.abs(current['logits']-archp['logits']).max())}
  manifest={'arm':'B0','seed':seed,'completed_epochs':200,'steps':3600,'train_seconds':history[-1]['elapsed_seconds'],'source_val_final':history[-1]['source_val'],
   'checkpoint_records':[{'epoch':int(p.stem.split('_')[-1]),'path':str(p),'sha256':s.sha(p)} for p in sorted((f/'checkpoints').glob('*.pth'))],
   'archived_B0_replay':replay,'target_GT_scored':False,'offline_export_recovery_only':True,
   'files_sha256':{str(p.relative_to(f)):s.sha(p) for p in f.rglob('*') if p.is_file()}}
  s.dump(f/'FINAL_PREDICTIONS_FROZEN.json',manifest);print('B0 export recovered',seed,replay,flush=True)
 for p,h in immutable.items():assert s.sha(p)==h
 # Preserve old source snapshots; amended code hashes are frozen BEFORE B1/B2.
 deps=old['files_sha256'].copy()
 for p in s.HERE.iterdir():
  if p.is_file():deps[str(p)]=s.sha(p)
 for p in audit.rglob('*'):
  if p.is_file():deps[str(p)]=s.sha(p)
 recovery={'reason':'source cache int16 labels must be torch.long for offline indexing; no training/loss/statistics changes',
  'original_pretrain_manifest_sha256':s.sha(audit/'ORIGINAL_PRETRAIN_WEIGHTS_CODE_FROZEN.json'),'all_existing_B0_training_and_prediction_files_unchanged':True,
  'immutable_B0_files_sha256':immutable,'target_GT_unopened':True,'B0_retraining_count':0}
 s.dump(audit/'RECOVERY_RECORD.json',recovery);deps[str(audit/'RECOVERY_RECORD.json')]=s.sha(audit/'RECOVERY_RECORD.json')
 s.dump(s.OUT/'PRETRAIN_WEIGHTS_CODE_FROZEN.json',{**old,'offline_export_fix_amendment_before_B1_B2':recovery,'files_sha256':deps})
 s.verify()
if __name__=='__main__':main()

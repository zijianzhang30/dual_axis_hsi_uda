import ast,json,shutil
from pathlib import Path
import study as s

def main():
 s.env().verify_files();s.OUT.mkdir(exist_ok=False);(s.OUT/'code_snapshot').mkdir();refs={};parent={}
 assert s.sha(Path(__file__).parent/'attention.py')==s.sha(s.PREVIOUS/'code_snapshot/attention.py')
 for name in ('code_manifest.json','baseline_manifest.json','parent_manifest.json'):
  p=s.PREVIOUS/name;parent[str(p)]=s.sha(p)
  for f,h in json.loads(p.read_text()).items():assert s.sha(f)==h,f;parent[f]=h
 assert json.loads((s.PREVIOUS/'status.json').read_text())['status']=='completed_two_exploratory_trainings_and_offline_scoring'
 for n in ('summary.json','status.json','preflight.json','final_training_audit.json','first100_prediction_reproduction.json'):
  p=s.PREVIOUS/n;parent[str(p)]=s.sha(p)
 for seed in s.SEEDS:
  folder=s.baseline(seed);cfg=json.loads((folder/'config.json').read_text());rows=[json.loads(x) for x in (folder/'history.jsonl').read_text().splitlines()];assert cfg['seed']==seed and cfg['epochs']==200 and cfg['patch']==13 and cfg['batch']==128 and cfg['lr']==.01 and cfg['depth']==3 and cfg['tokens']==4 and cfg['dim']==64 and cfg['loss']=='source-self CE only' and cfg['BN']=='unchanged Full Joint/momentum.19' and len(rows)==200
  for n in ('config.json','history.jsonl','frozen_manifest.json'):
   p=folder/n;refs[str(p)]=s.sha(p)
  for arm in ('TH0',)+s.ARMS if seed==2100 else ('TH0',):
   p=s.prediction_manifest(arm,seed);refs[str(p)]=s.sha(p);rr=json.loads(p.read_text())['records'];assert [r['epoch'] for r in rr]==s.PLAN['grid']
   for r in rr:assert s.sha(r['path'])==r['sha256'];refs[r['path']]=r['sha256']
   cp_folder=folder if arm=='TH0' else s.PREVIOUS/('%s_%d'%(arm,seed))
   for r in rr:
    p=cp_folder/'checkpoints'/('epoch_%03d.pth'%r['epoch']);refs[str(p)]=s.sha(p)
   if arm!='TH0':
    for n in ('config.json','history.jsonl','frozen_manifest.json'):
     p=cp_folder/n;parent[str(p)]=s.sha(p)
 s.dump(s.OUT/'parent_manifest.json',parent);s.dump(s.OUT/'baseline_manifest.json',refs);s.dump(s.OUT/'locked_config.json',s.PLAN);files={}
 for p in Path(__file__).parent.iterdir():
  if p.is_file():
   if p.suffix=='.py':ast.parse(p.read_text())
   q=s.OUT/'code_snapshot'/p.name;shutil.copy2(p,q);files[str(p.resolve())]=s.sha(p);files[str(q)]=s.sha(q)
 files[str(s.OUT/'locked_config.json')]=s.sha(s.OUT/'locked_config.json');s.dump(s.OUT/'code_manifest.json',files)
 s.dump(s.OUT/'history_audit.json',{'reuse_TH0_seeds':list(s.SEEDS),'reuse_TH1_TH2_seeds':[2100],'new_jobs':[[a,k] for k in s.NEW_SEEDS for a in s.ARMS],'new_training_count':4,'protocol_and_attention_unchanged':True,'no_existing_result_overwritten':True})
 s.dump(s.OUT/'status.json',{'status':'prepared_preflight_pending'});(s.ROOT/'results'/s.OUT.name).symlink_to(s.OUT,target_is_directory=True);s.verify();print('PREPARED_4_NEW_TRAININGS',flush=True)
if __name__=='__main__':main()

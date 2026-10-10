import ast,json,shutil
from pathlib import Path
import study as s

def main():
 s.env().verify_files();s.OUT.mkdir(exist_ok=False);(s.OUT/'code_snapshot').mkdir();files={};parent={}
 for n in ('code_manifest.json','baseline_manifest.json'):
  p=s.PREVIOUS/n;parent[str(p)]=s.sha(p)
  for f,h in json.loads(p.read_text()).items():assert s.sha(f)==h,f;parent[f]=h
 assert json.loads((s.PREVIOUS/'preflight.json').read_text())['passed'];parent[str(s.PREVIOUS/'preflight.json')]=s.sha(s.PREVIOUS/'preflight.json')
 assert s.sha(Path(__file__).parent/'attention.py')==s.sha(s.PREVIOUS/'code_snapshot/attention.py')
 for arm in s.ARMS:
  f=s.PREVIOUS/(arm+'_2100');config=json.loads((f/'config.json').read_text());assert config['plan']['epochs']==100 and config['plan']['seed']==2100 and config['plan']['loss']=='source CE ONLY; no cosine/scenecache/LMMD/MCC/Eq9/adapter/EMA/distillation/reconstruction/TalkingHeadsotherlayers';rows=[json.loads(x) for x in (f/'history.jsonl').read_text().splitlines()];assert len(rows)==100
  for p in (f/'config.json',f/'history.jsonl',f/'frozen_manifest.json',f/'offline_predictions_frozen.json'):parent[str(p)]=s.sha(p)
  for r in json.loads((f/'frozen_manifest.json').read_text())['checkpoints']:assert s.sha(r['path'])==r['sha256'];parent[r['path']]=r['sha256']
  for r in json.loads((f/'offline_predictions_frozen.json').read_text())['records']:assert s.sha(r['path'])==r['sha256'];parent[r['path']]=r['sha256']
 s.dump(s.OUT/'parent_manifest.json',parent)
 for p in Path(__file__).parent.iterdir():
  if p.suffix=='.py':ast.parse(p.read_text())
  if p.is_file():
   q=s.OUT/'code_snapshot'/p.name;shutil.copy2(p,q);files[str(p.resolve())]=s.sha(p);files[str(q)]=s.sha(q)
 s.dump(s.OUT/'locked_config.json',s.PLAN);files[str(s.OUT/'locked_config.json')]=s.sha(s.OUT/'locked_config.json');s.dump(s.OUT/'code_manifest.json',files);refs={}
 for p in (s.baseline()/'config.json',s.baseline()/'history.jsonl',s.baseline()/'frozen_manifest.json'):refs[str(p)]=s.sha(p)
 old=s.B/'houston_shared_adapter_2x2_v1/CE0_2100/offline_predictions_frozen.json';refs[str(old)]=s.sha(old);rr=json.loads(old.read_text())['records'];assert [r['epoch'] for r in rr]==s.PLAN['grid']
 for r in rr:assert s.sha(r['path'])==r['sha256'];refs[r['path']]=r['sha256'];p=s.baseline()/'checkpoints'/('epoch_%03d.pth'%r['epoch']);refs[str(p)]=s.sha(p)
 s.dump(s.OUT/'baseline_manifest.json',refs);s.dump(s.OUT/'history_audit.json',{'TH0_strict200epochbaseline_reuse':True,'oldTH1_TH2_only100_epochs':True,'oldcheckpoints_noRNGstate_directresume_notused':True,'new_complete200_trainings':2,'old100_results_preserved':True,'no_new_variant_or_hyperparameter':True});s.dump(s.OUT/'status.json',{'status':'prepared_200_preflight_pending'});(s.ROOT/'results'/s.OUT.name).symlink_to(s.OUT,target_is_directory=True);s.verify();print('TALKING_200_PREPARED',flush=True)
if __name__=='__main__':main()

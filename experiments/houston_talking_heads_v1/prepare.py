import ast,json,shutil
from pathlib import Path
import study as s

def main():
 s.env().verify_files();s.OUT.mkdir(exist_ok=False);(s.OUT/'code_snapshot').mkdir();files={}
 for p in Path(__file__).parent.iterdir():
  if p.suffix=='.py':ast.parse(p.read_text())
  if p.is_file():
   q=s.OUT/'code_snapshot'/p.name;shutil.copy2(p,q);files[str(p.resolve())]=s.sha(p);files[str(q)]=s.sha(q)
 s.dump(s.OUT/'locked_config.json',s.PLAN);files[str(s.OUT/'locked_config.json')]=s.sha(s.OUT/'locked_config.json');s.dump(s.OUT/'code_manifest.json',files)
 cfg=json.loads((s.baseline()/'config.json').read_text());assert cfg['epochs']==200 and cfg['loss']=='source-self CE only' and cfg['batch']==128 and cfg['patch']==13 and cfg['lr']==.01 and cfg['depth']==3 and cfg['parameters']==378011;refs={}
 for p in (s.baseline()/'config.json',s.baseline()/'history.jsonl',s.baseline()/'frozen_manifest.json'):refs[str(p)]=s.sha(p)
 old=s.B/'houston_shared_adapter_2x2_v1/CE0_2100/offline_predictions_frozen.json';refs[str(old)]=s.sha(old);rr=[r for r in json.loads(old.read_text())['records'] if r['epoch']<=100];assert [r['epoch'] for r in rr]==s.PLAN['grid']
 for r in rr:assert s.sha(r['path'])==r['sha256'];refs[r['path']]=r['sha256'];p=s.baseline()/'checkpoints'/('epoch_%03d.pth'%r['epoch']);refs[str(p)]=s.sha(p)
 s.dump(s.OUT/'baseline_manifest.json',refs);s.dump(s.OUT/'history_audit.json',{'matchingTH1_TH2_history_found':False,'TH0_originalC_strict_reuse':True,'new_training_limit':2,'TH1_TH2_not_equal_parameter_count':True});s.dump(s.OUT/'status.json',{'status':'prepared_preflight_pending'});(s.ROOT/'results'/s.OUT.name).symlink_to(s.OUT,target_is_directory=True);s.verify();print('TALKING_HEADS_PREPARED',flush=True)
if __name__=='__main__':main()

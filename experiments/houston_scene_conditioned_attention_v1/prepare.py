import ast,json,shutil
from pathlib import Path
import numpy as np,torch
import study as s

def main():
 s.env().verify_files();s.OUT.mkdir(exist_ok=False);(s.OUT/'code_snapshot').mkdir();files={}
 for p in Path(__file__).parent.glob('*.py'):
  ast.parse(p.read_text());q=s.OUT/'code_snapshot'/p.name;shutil.copy2(p,q);files[str(p.resolve())]=s.sha(p);files[str(q)]=s.sha(q)
 s.dump(s.OUT/'locked_config.json',s.PLAN);files[str(s.OUT/'locked_config.json')]=s.sha(s.OUT/'locked_config.json')
 cfg=json.loads((s.baseline()/'config.json').read_text());assert cfg['epochs']==200 and cfg['arm']=='C' and cfg['loss']=='source-self CE only' and cfg['batch']==128 and cfg['patch']==13 and cfg['lr']==.01 and cfg['depth']==3 and cfg['parameters']==378011
 torch.set_num_threads(2);c=s.env();m,ls,helper,classes,a=c.setup('houston','C',2100);assert a['complete_initial_hash']==cfg['initial_full_model_sha256'];coh={};rng=np.random.RandomState(2100+182000)
 for name,l in [('source',ls[0]),('target',ls[2])]:
  pos=np.sort(rng.choice(len(l.dataset.indices),size=1024,replace=False));idx=l.dataset.indices[pos];coh[name]={'positions':pos.tolist(),'coordinates':idx.tolist(),'indices_sha256':c.sha_array(idx),'sampling':'existing candidate coordinates,uniform private RNG,no classvalues'}
 s.dump(s.OUT/'summary_cohorts.json',coh);files[str(s.OUT/'summary_cohorts.json')]=s.sha(s.OUT/'summary_cohorts.json');s.dump(s.OUT/'code_manifest.json',files)
 refs={}
 for p in (s.baseline()/'config.json',s.baseline()/'history.jsonl',s.baseline()/'frozen_manifest.json'):refs[str(p)]=s.sha(p)
 old=s.B/'houston_shared_adapter_2x2_v1/CE0_2100/offline_predictions_frozen.json';refs[str(old)]=s.sha(old)
 rr=[r for r in json.loads(old.read_text())['records'] if r['epoch']<=100];assert [r['epoch'] for r in rr]==s.PLAN['grid']
 for r in rr:assert s.sha(r['path'])==r['sha256'];refs[r['path']]=r['sha256'];cp=s.baseline()/'checkpoints'/('epoch_%03d.pth'%r['epoch']);refs[str(cp)]=s.sha(cp)
 s.dump(s.OUT/'baseline_manifest.json',refs);s.dump(s.OUT/'history_audit.json',{'complete_matching_S1_S2_found':False,'S0_exact_formalC_reused':True,'new_training_count':2,'S0_parameters':378011,'S1_S2_parameters':384251,'target_candidate_mask_inherited':True,'target_class_GT_not_used_for_cohort':True})
 (s.ROOT/'results'/s.OUT.name).symlink_to(s.OUT,target_is_directory=True);s.dump(s.OUT/'status.json',{'status':'prepared_preflight_pending'});print('SCENE_ATTENTION_PREPARED',flush=True)
if __name__=='__main__':main()

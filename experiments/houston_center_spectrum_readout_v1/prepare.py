import json,ast,shutil
from pathlib import Path
import study as s
s.OUT.mkdir(exist_ok=False);(s.OUT/'code_snapshot').mkdir();files={}
for p in Path(__file__).parent.glob('*.py'):
 ast.parse(p.read_text());q=s.OUT/'code_snapshot'/p.name;shutil.copy2(p,q);files[str(p)]=s.sha(p);files[str(q)]=s.sha(q)
s.dump(s.OUT/'locked_config.json',s.PLAN);files[str(s.OUT/'locked_config.json')]=s.sha(s.OUT/'locked_config.json');s.dump(s.OUT/'code_manifest.json',files)
refs={};audit=[]
for seed in s.SEEDS:
 f=s.baseline(seed);cfg=json.loads((f/'config.json').read_text());h=[json.loads(x) for x in (f/'history.jsonl').read_text().splitlines()];mf=json.loads((f/'frozen_manifest.json').read_text());assert cfg['epochs']==200 and cfg['arm']=='C' and cfg['loss']=='source-self CE only' and cfg['batch']==128 and cfg['patch']==13 and cfg['lr']==.01 and cfg['depth']==3 and cfg['parameters']==378011;assert len(h)==200 and all(r['steps']==18 for r in h)
 for p in (f/'config.json',f/'history.jsonl',f/'frozen_manifest.json'):refs[str(p)]=s.sha(p)
 for r in mf['checkpoints']+[mf['prediction']]:assert s.sha(r['path'])==r['sha256'];refs[r['path']]=r['sha256']
 old=s.B/'houston_shared_adapter_2x2_v1'/('CE0_%d'%seed)/'offline_predictions_frozen.json';refs[str(old)]=s.sha(old)
 rs=json.loads(old.read_text())['records'];assert [r['epoch'] for r in rs]==s.PLAN['evaluation_grid']
 for r in rs:assert s.sha(r['path'])==r['sha256'];refs[r['path']]=r['sha256']
 audit.append({'seed':seed,'R0_reuse':str(f),'all_config_budget_initial_hash_data_RNG_references_available':True,'R1_R2_exact_existing_training_found':False,'known_prior_spectral_fusions':'houston_spectral_complementarity_v1 frozen source-only probes,not main-network training'})
for p in list((s.B/'tokenizer_mechanism_v2').glob('*.py'))+list((s.B/'tokenizer_mechanism_v2/snapshot').rglob('*.py')):refs[str(p)]=s.sha(p)
s.dump(s.OUT/'baseline_manifest.json',refs);s.dump(s.OUT/'history_audit.json',{'audits':audit,'new_training_limit':6,'no_old_probe_used_as_formal_R1':True});s.dump(s.OUT/'status.json',{'status':'prepared_awaiting_preflight','new_training_started':False});(s.ROOT/'results'/s.OUT.name).symlink_to(s.OUT,target_is_directory=True)
print('CENTER_READOUT_PREPARED',flush=True)

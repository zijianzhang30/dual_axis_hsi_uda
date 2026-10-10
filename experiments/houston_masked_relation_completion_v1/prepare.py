import ast,json,shutil
from pathlib import Path
import numpy as np,h5py
import study as s
s.OUT.mkdir(exist_ok=False);(s.OUT/'data').mkdir();(s.OUT/'code_snapshot').mkdir();files={}
for p in Path(__file__).parent.glob('*.py'):
 ast.parse(p.read_text());q=s.OUT/'code_snapshot'/p.name;shutil.copy2(p,q);files[str(p)]=s.sha(p);files[str(q)]=s.sha(q)
s.dump(s.OUT/'locked_config.json',s.PLAN);files[str(s.OUT/'locked_config.json')]=s.sha(s.OUT/'locked_config.json');s.dump(s.OUT/'code_manifest.json',files)
refs={};old=s.baseline();cfg=json.loads((old/'config.json').read_text());hist=[json.loads(x) for x in (old/'history.jsonl').read_text().splitlines()];assert cfg['epochs']==200 and cfg['arm']=='C' and cfg['loss']=='source-self CE only' and cfg['parameters']==378011 and len(hist)==200
for p in (old/'config.json',old/'history.jsonl',old/'frozen_manifest.json'):refs[str(p)]=s.sha(p)
for r in json.loads((old/'frozen_manifest.json').read_text())['checkpoints']:
 if r['epoch']<=100:assert s.sha(r['path'])==r['sha256'];refs[r['path']]=r['sha256']
f=s.B/'houston_shared_adapter_2x2_v1/CE0_2100/offline_predictions_frozen.json';refs[str(f)]=s.sha(f)
for r in json.loads(f.read_text())['records']:
 if r['epoch']<=100:refs[r['path']]=s.sha(r['path']);assert refs[r['path']]==r['sha256']
for domain,name in [('source','Houston13'),('target','Houston18')]:
 p=Path('/home/zhangzj26/TGRS_MLUDA-2024/datasets/Houston')/(name+'.mat');refs[str(p)]=s.sha(p)
 with h5py.File(p,'r') as f:x=np.asarray(f['ori_data']).transpose(2,1,0).astype(np.float32)
 assert x.shape==(210,954,48) and np.isfinite(x).all();x=np.pad(x,((6,6),(6,6),(0,0)),mode='reflect');q=s.OUT/'data'/(domain+'_raw.npy');np.save(q,x);refs[str(q)]=s.sha(q)
for p in list((s.B/'tokenizer_mechanism_v2').glob('*.py'))+list((s.B/'tokenizer_mechanism_v2/snapshot').rglob('*.py')):refs[str(p)]=s.sha(p)
s.dump(s.OUT/'baseline_manifest.json',refs);s.dump(s.OUT/'history_audit.json',{'E0_exact_seed2100_budget100_reuse':str(old),'E1_E2_no_matching_formal_training_found':True,'two_new_training_limit':True,'raw_augmented_companions_keep_same_main_tensors_and_workerRNG':True});s.dump(s.OUT/'status.json',{'status':'prepared_not_training'});(s.ROOT/'results'/s.OUT.name).symlink_to(s.OUT,target_is_directory=True);print('COMPLETION_PREPARED',flush=True)

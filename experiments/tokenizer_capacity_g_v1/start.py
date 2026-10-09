import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from common import OUT,BASE,dump,sha
assert json.loads((OUT/'verification.json').read_text())['passed']
assert not (OUT/'launch.json').exists()
for p in Path(__file__).parent.iterdir():
 if p.is_file():shutil.copy2(p,OUT/p.name)
provenance=json.loads((OUT/'provenance.json').read_text());provenance['study_sha256']={p.name:sha(p) for p in OUT.glob('*.py')};provenance['protocol_sha256']=sha(OUT/'PROTOCOL.md');dump(OUT/'provenance.json',provenance)
# Save available old C initial/data-order references without modifying them.
reference={}
for dataset in ['houston','pavia']:
 for seed in [2100,2101,2102]:
  folder=BASE/dataset/f'C_{seed}';cfg=folder/'config.json';history=folder/'history.jsonl'
  if not cfg.exists():continue
  c=json.loads(cfg.read_text());h=[json.loads(x) for x in history.read_text().splitlines()] if history.exists() else []
  reference[f'{dataset}_C_{seed}']={'initial_original':c['initial_original_model_sha256'],'initial_full':c['initial_full_model_sha256'],'initial_scorer':c['initial_scorer_sha256'],'indices_sha256':c['indices_sha256'],'config_sha256':sha(cfg),
    'available_epoch_order_hashes':[{k:row[k] for k in ['epoch','batch_stream_sha256','original_rng_stream_sha256']} for row in h]}
dump(OUT/'existing_C_initial_and_data_order_hashes.json',reference)
env=os.environ.copy();env.update(PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MPLBACKEND='Agg')
with (OUT/'launcher.log').open('x') as log:
 p=subprocess.Popen([sys.executable,'-u',str(OUT/'launch.py')],cwd=OUT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
dump(OUT/'launch.json',{'pid':p.pid,'started':time.time(),'waiting_for_A_F':True,'formal_G_started':False,'G_seeds':[2100,2101,2102],'epochs':200,'initial_physical_gpus':[0,4,5],'no_existing_task_interrupted':True})
print('Audit passed; G queue waiting for existing Pavia A-F, controller',p.pid)

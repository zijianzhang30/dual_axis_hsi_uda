"""Launch the sole predefined release repair, then report and stop. No Ours queue."""
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
import study as s

def main():
 assert json.loads((s.OUT/'preflight.json').read_text())['passed']
 assert all((s.OUT/('baseline_%d/full_evaluation.json'%seed)).exists() for seed in s.SEEDS)
 s.verify_originals()
 own=Path(__file__).parent;snapshot=s.OUT/'experiment_code';snapshot.mkdir(exist_ok=False)
 for p in own.glob('*.py'):shutil.copy2(p,snapshot/p.name)
 s.dump(s.OUT/'new_code_manifest.json',{str(p):s.sha(p) for p in list(own.glob('*.py'))+list((s.OUT/'runtime_snapshot').rglob('*.py'))+[s.OUT/'frozen_official_helper.py',s.OUT/'control_snapshot.py']})
 jobs=[]
 for seed,gpu in zip(s.SEEDS,(0,4,5)):
  env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(gpu),PYTHONDONTWRITEBYTECODE='1',
       OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
  log=(s.OUT/('repair_%d.log'%seed)).open('w')
  p=subprocess.Popen([sys.executable,'-u',str(own/'run.py'),'--seed',str(seed)],env=env,stdout=log,stderr=subprocess.STDOUT)
  jobs.append((seed,gpu,p,log));print('Started official BiDA MMD argument repair seed%d GPU%d'%(seed,gpu),flush=True)
 while True:
  rows=[{'seed':seed,'gpu':gpu,'pid':p.pid,'exit_code':p.poll()} for seed,gpu,p,log in jobs]
  s.dump(s.OUT/'launcher_status.json',{'status':'running_single_factor_release_repair','jobs':rows,'Ours_paused':True,'automatic_followup_factors':False})
  if all(row['exit_code'] is not None for row in rows):break
  time.sleep(10)
 for seed,gpu,p,log in jobs:log.close()
 assert all(row['exit_code']==0 for row in rows), 'No automatic retry; inspect failure log'
 with (s.OUT/'final_report.log').open('w') as log:
  subprocess.run([sys.executable,str(own/'audit_report.py')],stdout=log,stderr=subprocess.STDOUT,check=True)
 s.verify_originals()
 for path,h in json.loads((s.OUT/'new_code_manifest.json').read_text()).items():assert s.sha(path)==h,path
 s.dump(s.OUT/'launcher_status.json',{'status':'completed_repair_and_audit_no_followups','jobs':rows,'Ours_paused':True,'automatic_followup_factors':False})
 print('Single controlled repair complete. Audit reported. No Ours or other factors launched.',flush=True)

if __name__=='__main__':main()

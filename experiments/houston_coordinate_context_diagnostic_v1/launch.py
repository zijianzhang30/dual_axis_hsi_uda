import os,sys,subprocess,time,json
from pathlib import Path
import common as c

def main():
 c.verify();jobs=[];code=Path(__file__).parent
 for seed,gpu in [(2100,5),(2101,3),(2102,6)]:
  env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(gpu),PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2');f=(c.OUT/('seed_%d.log'%seed)).open('x');p=subprocess.Popen([sys.executable,'-u',str(code/'run.py'),'--seed',str(seed)],env=env,stdout=f,stderr=subprocess.STDOUT);jobs.append((seed,gpu,p,f));print('START_SOURCE_PROBE_DIAG',seed,'GPU',gpu,'PID',p.pid,flush=True)
 while True:
  rows=[{'seed':s,'gpu':g,'pid':p.pid,'exit_code':p.poll()} for s,g,p,_ in jobs];c.dump(c.OUT/'status.json',{'status':'frozen_feature_extraction_and_source_probe_fit','jobs':rows,'main_network_training':False})
  if all(r['exit_code'] is not None for r in rows):break
  time.sleep(10)
 for *_,f in jobs:f.close()
 if not all(r['exit_code']==0 for r in rows):c.dump(c.OUT/'status.json',{'status':'failed_no_automatic_experiment_retries','jobs':rows});return
 with (c.OUT/'report.log').open('w') as f:subprocess.run([sys.executable,'-u',str(code/'report.py')],stdout=f,stderr=subprocess.STDOUT,check=True)
 c.dump(c.OUT/'status.json',{'status':'completed_source_probe_diagnostic_no_followups','no_main_network_training':True})
if __name__=='__main__':main()

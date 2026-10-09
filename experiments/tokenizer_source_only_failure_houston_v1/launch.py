import os,sys,json,time,subprocess
from common import OUT,SEEDS,dump

def main():
 assert json.loads((OUT/'verification.json').read_text())['passed'];(OUT/'houston').mkdir(exist_ok=False)
 env=os.environ.copy();env.update(PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2');active=[]
 for seed,gpu in zip(SEEDS,[3,6,7]):
  e=env.copy();e['CUDA_VISIBLE_DEVICES']=str(gpu);log=(OUT/f'worker_{seed}.log').open('x');p=subprocess.Popen([sys.executable,'-u',str(OUT/'worker.py'),'--seed',str(seed)],cwd=OUT,env=e,stdout=log,stderr=subprocess.STDOUT);active.append((seed,gpu,p,log))
 while active:
  finished=[]
  for seed,gpu,p,log in active:
   if p.poll() is None:continue
   log.close()
   if p.returncode:dump(OUT/'status.json',{'state':'training_failed','seed':seed,'exit_code':p.returncode,'no_auto_retry':True});return
   finished.append(p.pid)
  active=[r for r in active if r[2].pid not in finished];dump(OUT/'status.json',{'state':'source_only_training','active':[{'seed':s,'gpu':g,'worker_pid':p.pid} for s,g,p,l in active],'updated':time.time()})
  if active:time.sleep(10)
 (OUT/'diagnostics').mkdir(exist_ok=False);dump(OUT/'all_training_frozen_before_target_access.json',{'all6_frozen':True,'no_target_during_training':True});active=[]
 for seed,gpu in zip(SEEDS,[3,6,7]):
  e=env.copy();e['CUDA_VISIBLE_DEVICES']=str(gpu);log=(OUT/f'diagnose_{seed}.log').open('x');p=subprocess.Popen([sys.executable,'-u',str(OUT/'diagnose.py'),'--seed',str(seed)],cwd=OUT,env=e,stdout=log,stderr=subprocess.STDOUT);active.append((seed,gpu,p,log))
 while active:
  finished=[]
  for seed,gpu,p,log in active:
   if p.poll() is None:continue
   log.close()
   if p.returncode:dump(OUT/'status.json',{'state':'diagnosis_failed','seed':seed,'exit_code':p.returncode,'no_auto_retry':True});return
   finished.append(p.pid)
  active=[r for r in active if r[2].pid not in finished];dump(OUT/'status.json',{'state':'read_only_diagnosis','active':[{'seed':s,'gpu':g,'pid':p.pid} for s,g,p,l in active],'updated':time.time()})
  if active:time.sleep(10)
 with (OUT/'report.log').open('x') as log:code=subprocess.call([sys.executable,'-u',str(OUT/'report.py')],cwd=OUT,env=env,stdout=log,stderr=subprocess.STDOUT)
 dump(OUT/'status.json',{'state':'completed' if code==0 else 'report_failed','exit_code':code,'new_runs':6,'seeds':list(SEEDS),'epochs':200,'no_auto_followup':True,'Pavia_modified':False,'updated':time.time()})
if __name__=='__main__':main()

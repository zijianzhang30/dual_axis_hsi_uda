import json,os,subprocess,sys,time
from pathlib import Path
import study as s

def main():
 assert json.loads((s.OUT/'preflight.json').read_text())['passed'];s.verify_originals()
 for p,h in json.loads((s.OUT/'code_manifest.json').read_text()).items():assert s.sha(p)==h,p
 jobs=[];pending=list(zip(s.SEEDS,(0,4,5)))
 while pending or any(p.poll() is None for _,_,p,_ in jobs):
  independent=json.loads((s.PRIOR/'launcher_status.json').read_text())
  for seed,gpu in list(pending):
   prior=next(r for r in independent['jobs'] if r['seed']==seed)
   if prior['exit_code'] is None:continue
   # Independent MMD job finished naturally; no restart, termination or combination.
   env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(gpu),PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
   log=(s.OUT/('B1_%d.log'%seed)).open('w')
   p=subprocess.Popen([sys.executable,'-u',str(Path(__file__).parent/'run.py'),'--seed',str(seed)],env=env,stdout=log,stderr=subprocess.STDOUT)
   jobs.append((seed,gpu,p,log));pending.remove((seed,gpu));print('B1 Eq9 seed%d GPU%d pid%d'%(seed,gpu,p.pid),flush=True)
  s.dump(s.OUT/'launcher_status.json',{'status':'running_or_waiting_for_independent_MMD_gpu_release',
   'jobs':[{'seed':seed,'gpu':gpu,'pid':p.pid,'exit_code':p.poll()} for seed,gpu,p,_ in jobs],
   'queued':[{'seed':seed,'gpu':gpu} for seed,gpu in pending], 'Ours_paused':True,'no_combined_factors':True})
  time.sleep(10)
 for seed,gpu,p,log in jobs:log.close()
 rows=[{'seed':seed,'gpu':gpu,'pid':p.pid,'exit_code':p.poll()} for seed,gpu,p,_ in jobs]
 if not all(r['exit_code']==0 for r in rows):
  s.dump(s.OUT/'launcher_status.json',{'status':'failed_no_automatic_retry','jobs':rows});raise RuntimeError('Inspect failing B1 log; no automatic retry')
 with (s.OUT/'report.log').open('w') as f:subprocess.run([sys.executable,str(Path(__file__).parent/'report.py')],stdout=f,stderr=subprocess.STDOUT,check=True)
 s.verify_originals();s.dump(s.OUT/'launcher_status.json',{'status':'completed_no_followups','jobs':rows})
if __name__=='__main__':main()

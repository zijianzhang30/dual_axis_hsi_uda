import json,os,subprocess,sys,time
from pathlib import Path
import study as s

def main():
 assert json.loads((s.OUT/'preflight.json').read_text())['passed'];s.verify()
 # Six workers on six cards with ample reported memory; no existing job is stopped.
 pending=[('bida_full_C',2100,5),('bida_full_C',2101,4),('bida_full_C',2102,0),
          ('frozen_ours',2100,2),('frozen_ours',2101,6),('frozen_ours',2102,7)]
 jobs=[]
 while pending or any(p.poll() is None for _,_,_,p,_ in jobs):
  for track,seed,gpu in list(pending):
   if gpu==0 and not (s.B1/'B1_2100/frozen_manifest.json').exists():continue
   env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(gpu),PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
   log=(s.OUT/('%s_%d.log'%(track,seed))).open('w')
   p=subprocess.Popen([sys.executable,'-u',str(Path(__file__).parent/'run.py'),'--track',track,'--seed',str(seed)],env=env,stdout=log,stderr=subprocess.STDOUT)
   jobs.append((track,seed,gpu,p,log));pending.remove((track,seed,gpu));print('START',track,seed,'GPU',gpu,'PID',p.pid,flush=True)
  rows=[{'track':track,'seed':seed,'gpu':gpu,'pid':p.pid,'exit_code':p.poll()} for track,seed,gpu,p,_ in jobs]
  s.dump(s.OUT/'launcher_status.json',{'status':'running_two_authorized_tracks','jobs':rows,
  'queued':[{'track':t,'seed':seed,'gpu':gpu} for t,seed,gpu in pending],'no_extra_variants':True})
  time.sleep(10)
 for _,_,_,_,log in jobs:log.close()
 if not all(r['exit_code']==0 for r in rows):
  s.dump(s.OUT/'launcher_status.json',{'status':'failed_no_automatic_retries','jobs':rows});raise RuntimeError('Inspect failed log')
 with (s.OUT/'report.log').open('w') as f:subprocess.run([sys.executable,str(Path(__file__).parent/'report.py')],stdout=f,stderr=subprocess.STDOUT,check=True)
 s.verify();s.dump(s.OUT/'launcher_status.json',{'status':'completed_no_followups','jobs':rows})
if __name__=='__main__':main()

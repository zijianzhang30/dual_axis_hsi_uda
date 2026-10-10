import subprocess,os,sys,time
import common as c

def main():
 pending=[(d,s) for d in ('houston','pavia') for s in c.SEEDS];running=[];free=[5,3,0]
 while pending or running:
  while pending and free:
   dataset,seed=pending.pop(0);gpu=free.pop(0);env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1');f=(c.OUT/('%s_%d.log'%(dataset,seed))).open('x');p=subprocess.Popen([sys.executable,'-u',str(c.OUT/'code_snapshot/extract.py'),'--dataset',dataset,'--seed',str(seed)],env=env,stdout=f,stderr=subprocess.STDOUT);running.append((dataset,seed,gpu,p))
  for job in list(running):
   d,s,g,p=job
   if p.poll() is not None:
    if p.returncode:c.dump(c.OUT/'status.json',{'status':'STOP_frozen_extraction_failure','dataset':d,'seed':s,'exit_code':p.returncode});return
    running.remove(job);free.append(g)
  c.dump(c.OUT/'status.json',{'status':'frozen_feature_extraction','running':[{'dataset':d,'seed':s,'gpu':g,'pid':p.pid} for d,s,g,p in running],'pending':pending});time.sleep(10)
 c.dump(c.OUT/'status.json',{'status':'offline_geometry_scoring'});env=os.environ.copy();env.update(OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1')
 with (c.OUT/'report.log').open('x') as f:p=subprocess.run([sys.executable,'-u',str(c.OUT/'code_snapshot/report.py')],env=env,stdout=f,stderr=subprocess.STDOUT)
 c.dump(c.OUT/'status.json',{'status':'completed_no_followups' if not p.returncode else 'STOP_report_failure','no_training':True})
if __name__=='__main__':main()

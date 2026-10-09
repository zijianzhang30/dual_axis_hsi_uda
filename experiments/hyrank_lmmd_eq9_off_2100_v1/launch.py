import json,os,subprocess,sys,time
from pathlib import Path
import study as s

def main():
 assert json.loads((s.OUT/'preflight.json').read_text())['passed'];s.verify()
 env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='0',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
 with (s.OUT/'C_2100.log').open('x') as f:
  p=subprocess.Popen([sys.executable,'-u',str(Path(__file__).parent/'run.py'),'--seed','2100','--arm','C'],env=env,stdout=f,stderr=subprocess.STDOUT)
  while True:
   code=p.poll();s.dump(s.OUT/'launcher_status.json',{'status':'running_single_Eq9_off_control','seed':2100,'gpu':0,'pid':p.pid,'exit_code':code,'training_count':1})
   if code is not None:break
   time.sleep(10)
 if code!=0:
  s.dump(s.OUT/'launcher_status.json',{'status':'failed_no_automatic_retry','pid':p.pid,'exit_code':code});return
 with (s.OUT/'report.log').open('w') as f:subprocess.run([sys.executable,str(Path(__file__).parent/'report.py')],stdout=f,stderr=subprocess.STDOUT,check=True)
 s.verify();s.dump(s.OUT/'launcher_status.json',{'status':'completed_single_control_no_followups','seed':2100,'training_count':1})
if __name__=='__main__':main()

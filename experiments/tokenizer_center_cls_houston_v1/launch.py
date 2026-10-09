import os,sys,json,time,subprocess
from common import OUT,dump

def main():
 assert json.loads((OUT/'verification.json').read_text())['passed'];(OUT/'houston').mkdir(exist_ok=False)
 base=os.environ.copy();base.update(PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2');active=[]
 for arm,gpu in [('CQ0',7),('CQ1',6)]:
  e=base.copy();e['CUDA_VISIBLE_DEVICES']=str(gpu);l=(OUT/f'{arm}_2100_e100.log').open('x');p=subprocess.Popen([sys.executable,'-u',str(OUT/'train.py'),'--dataset','houston','--arm',arm,'--seed','2100','--epochs','100'],cwd=OUT,env=e,stdout=l,stderr=subprocess.STDOUT);active.append((arm,gpu,p,l))
 while active:
  done=[]
  for arm,gpu,p,l in active:
   if p.poll() is None:continue
   l.close()
   if p.returncode:dump(OUT/'status.json',{'state':'training_failed','arm':arm,'exit_code':p.returncode,'no_auto_retry':True});return
   done.append(p.pid)
  active=[q for q in active if q[2].pid not in done];dump(OUT/'status.json',{'state':'training','active':[{'arm':a,'gpu':g,'pid':p.pid,'seed':2100,'epochs':100} for a,g,p,l in active],'updated':time.time()})
  if active:time.sleep(10)
 e=base.copy();e['CUDA_VISIBLE_DEVICES']='7';dump(OUT/'status.json',{'state':'evaluating_C_CQ0_CQ1_at_epoch100','updated':time.time()})
 with (OUT/'evaluate.log').open('x') as l:code=subprocess.call([sys.executable,'-u',str(OUT/'evaluate.py')],cwd=OUT,env=e,stdout=l,stderr=subprocess.STDOUT)
 dump(OUT/'status.json',{'state':'completed' if code==0 else 'evaluation_failed','exit_code':code,'new_training_runs':2,'seed':2100,'epochs':100,'Pavia_modified':False,'no_auto_followup':True,'updated':time.time()})
if __name__=='__main__':main()

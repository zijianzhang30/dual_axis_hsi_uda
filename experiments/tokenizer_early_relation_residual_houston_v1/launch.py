import json,os,sys,time,subprocess
from common import OUT,SEEDS,dump

def main():
 assert json.loads((OUT/'verification.json').read_text())['passed']
 (OUT/'houston').mkdir(exist_ok=False)
 base=os.environ.copy();base.update(PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
 def stage(epochs,seeds,gpus):
  active=[]
  for s,g in zip(seeds,gpus):
   e=base.copy();e['CUDA_VISIBLE_DEVICES']=str(g);l=(OUT/f'S1_{s}_e{epochs}.log').open('x');p=subprocess.Popen([sys.executable,'-u',str(OUT/'train.py'),'--dataset','houston','--arm','S1','--seed',str(s),'--epochs',str(epochs)],cwd=OUT,env=e,stdout=l,stderr=subprocess.STDOUT);active.append((s,g,p,l))
  while active:
   done=[]
   for s,g,p,l in active:
    if p.poll() is None:continue
    l.close()
    if p.returncode:raise RuntimeError(f'training failed {s} {epochs}: {p.returncode}')
    done.append(p.pid)
   active=[a for a in active if a[2].pid not in done]
   dump(OUT/'status.json',{'state':'training','stage':1 if epochs==100 else 2,'epochs':epochs,'active':[{'seed':s,'gpu':g,'pid':p.pid} for s,g,p,l in active],'updated':time.time()})
   if active:time.sleep(10)
  e=base.copy();e['CUDA_VISIBLE_DEVICES']='6';dump(OUT/'status.json',{'state':'evaluating_paired_endpoint','epochs':epochs,'updated':time.time()})
  with (OUT/f'evaluate_e{epochs}.log').open('x') as l:code=subprocess.call([sys.executable,'-u',str(OUT/'evaluate.py'),'--epochs',str(epochs)],cwd=OUT,env=e,stdout=l,stderr=subprocess.STDOUT)
  if code:raise RuntimeError(f'paired endpoint evaluation failed {epochs}: {code}')
  return json.loads((OUT/f'evaluation_e{epochs}/summary.json').read_text())['decision']
 try:
  decision=stage(100,[2100],[6])
  if decision['continue_stage2']:
   # Full200 fromscratch at identical init/order; first100 repeat is disclosed,
   # avoids unsafe reconstruction of loader/worker RNG from weight-only checkpoint.
   stage(200,list(SEEDS),[3,6,7])
   dump(OUT/'status.json',{'state':'completed_stage2','formal_runs':3,'fixed_epochs':200,'no_auto_followup':True,'updated':time.time()})
  else:dump(OUT/'status.json',{'state':'No-Go_after_stage1','decision':decision,'stage2_started':False,'no_auto_modules':True,'updated':time.time()})
 except Exception as exc:dump(OUT/'status.json',{'state':'failed','error':str(exc),'no_auto_retry':True,'updated':time.time()})
if __name__=='__main__':main()

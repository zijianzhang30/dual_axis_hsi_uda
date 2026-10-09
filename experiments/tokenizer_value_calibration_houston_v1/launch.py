import json,os,sys,time,subprocess
from common import OUT,SEEDS,dump

def main():
 assert json.loads((OUT/'verification.json').read_text())['passed'];(OUT/'houston').mkdir(exist_ok=False)
 base=os.environ.copy();base.update(PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
 def stage(epochs):
  queue=[('V1',2100)] if epochs==100 else [(a,s) for s in SEEDS for a in ['V0','V1']];available=[3] if epochs==100 else [2,3,6];active=[]
  while queue or active:
   while queue and available:
    a,s=queue.pop(0);g=available.pop(0);e=base.copy();e['CUDA_VISIBLE_DEVICES']=str(g);l=(OUT/f'{a}_{s}_e{epochs}.log').open('x');p=subprocess.Popen([sys.executable,'-u',str(OUT/'train.py'),'--dataset','houston','--arm',a,'--seed',str(s),'--epochs',str(epochs)],cwd=OUT,env=e,stdout=l,stderr=subprocess.STDOUT);active.append((a,s,g,p,l))
   keep=[]
   for a,s,g,p,l in active:
    if p.poll() is None:keep.append((a,s,g,p,l));continue
    l.close()
    if p.returncode:raise RuntimeError(f'training failed {a} {s} e{epochs}: {p.returncode}')
    available.append(g)
   active=keep;dump(OUT/'status.json',{'state':'training','stage':1 if epochs==100 else 2,'fixed_epochs':epochs,'active':[{'arm':a,'seed':s,'gpu':g,'pid':p.pid} for a,s,g,p,l in active],'queued':[{'arm':a,'seed':s} for a,s in queue],'updated':time.time()})
   if queue or active:time.sleep(10)
  e=base.copy();e['CUDA_VISIBLE_DEVICES']='3';dump(OUT/'status.json',{'state':'evaluating_paired_endpoint','epochs':epochs,'updated':time.time()})
  with (OUT/f'evaluate_e{epochs}.log').open('x') as l:code=subprocess.call([sys.executable,'-u',str(OUT/'evaluate.py'),'--epochs',str(epochs)],cwd=OUT,env=e,stdout=l,stderr=subprocess.STDOUT)
  if code:raise RuntimeError(f'endpoint evaluation failed {epochs}: {code}')
  return json.loads((OUT/f'evaluation_e{epochs}/summary.json').read_text())['decision']
 try:
  decision=stage(100)
  if decision['continue_stage2']:
   stage(200);dump(OUT/'status.json',{'state':'completed_stage2','matched_formal_runs':6,'fixed_epochs':200,'no_auto_followup':True,'Pavia_modified':False,'updated':time.time()})
  else:dump(OUT/'status.json',{'state':'No-Go_after_stage1','decision':decision,'stage2_started':False,'no_auto_WV_local_variants':True,'Pavia_modified':False,'updated':time.time()})
 except Exception as exc:
  for a,s,g,p,l in active if 'active' in locals() else []:
   if p.poll() is None:p.terminate()
  dump(OUT/'status.json',{'state':'failed','error':str(exc),'no_auto_retry':True,'updated':time.time()})
if __name__=='__main__':main()

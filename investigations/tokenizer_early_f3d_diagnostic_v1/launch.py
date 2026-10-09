import os,sys,time,subprocess
from common import OUT,SEEDS,dump
base=os.environ.copy();base.update(PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2');active=[]
for seed,gpu in zip(SEEDS,[3,6,7]):
 e=base.copy();e['CUDA_VISIBLE_DEVICES']=str(gpu);l=(OUT/f'seed_{seed}.log').open('x');p=subprocess.Popen([sys.executable,'-u',str(OUT/'run_seed.py'),'--seed',str(seed)],cwd=OUT,env=e,stdout=l,stderr=subprocess.STDOUT);active.append((seed,gpu,p,l))
while active:
 done=[]
 for seed,gpu,p,l in active:
  if p.poll() is None:continue
  l.close()
  if p.returncode:dump(OUT/'status.json',{'state':'diagnosis_failed','seed':seed,'code':p.returncode,'no_auto_retry':True});sys.exit(1)
  done.append(p.pid)
 active=[a for a in active if a[2].pid not in done];dump(OUT/'status.json',{'state':'read_only_diagnosis','active':[{'seed':s,'gpu':g,'pid':p.pid} for s,g,p,l in active],'updated':time.time()})
 if active:time.sleep(10)
with (OUT/'report.log').open('x') as l:code=subprocess.call([sys.executable,'-u',str(OUT/'report.py')],cwd=OUT,env=base,stdout=l,stderr=subprocess.STDOUT)
dump(OUT/'status.json',{'state':'completed' if not code else 'report_failed','code':code,'new_backbone_training':False,'new_module_design':False,'Pavia_modified':False,'updated':time.time()})

import os,subprocess,sys,json
from common import *
CODE=Path(__file__).parent;PY='/home/zhangzj26/TGRS_MLUDA-2024/.venv/bin/python';env=os.environ.copy();env.update(PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',CUDA_VISIBLE_DEVICES='');jobs=[]
for seed in SEEDS:
 f=(OUT/('seed_%d.log'%seed)).open('w');p=subprocess.Popen([PY,'-u',str(CODE/'run.py'),'--seed',str(seed)],env=env,stdout=f,stderr=subprocess.STDOUT);jobs.append((seed,p,f))
dump(OUT/'processes.json',[{'seed':s,'pid':p.pid,'CPU_only':True} for s,p,f in jobs]);codes={}
for seed,p,f in jobs:codes[seed]=p.wait();f.close()
if any(codes.values()):dump(OUT/'status.json',{'status':'failed_no_auto_feature_or_parameter_change','exit_codes':codes});sys.exit(1)
with (OUT/'report.log').open('w') as f:r=subprocess.run([PY,'-u',str(CODE/'report.py')],env=env,stdout=f,stderr=subprocess.STDOUT)
sys.exit(r.returncode)

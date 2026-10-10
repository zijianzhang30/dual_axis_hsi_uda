import os,sys,subprocess,json,time
from pathlib import Path
ROOT=Path('/home/zhangzj26/dual_axis_hsi_uda');CODE=ROOT/'experiments/houston_lmmd_scale_stage_diagnostic_v1';OUT=Path('/nas1/zhangzj26/dual_axis_hsi_uda/results/houston_lmmd_scale_stage_diagnostic_v1');PY='/home/zhangzj26/TGRS_MLUDA-2024/.venv/bin/python';jobs=[]
for seed,gpu in [(2100,1),(2101,6),(2102,7)]:
 env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(gpu),PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2');f=(OUT/('seed_%d.log'%seed)).open('w');p=subprocess.Popen([PY,'-u',str(CODE/'run.py'),'--seed',str(seed)],cwd=str(ROOT),env=env,stdout=f,stderr=subprocess.STDOUT);jobs.append((seed,gpu,p,f))
(OUT/'processes.json').write_text(json.dumps([{'seed':seed,'gpu':gpu,'pid':p.pid} for seed,gpu,p,f in jobs],indent=2))
codes={}
for seed,gpu,p,f in jobs:codes[seed]=p.wait();f.close()
if any(codes.values()):
 (OUT/'status.json').write_text(json.dumps({'status':'failed_no_auto_repair','exit_codes':codes}));sys.exit(1)
env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='1',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
with (OUT/'report.log').open('w') as f:r=subprocess.run([PY,'-u',str(CODE/'report.py')],cwd=str(ROOT),env=env,stdout=f,stderr=subprocess.STDOUT)
sys.exit(r.returncode)

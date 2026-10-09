import os,sys,subprocess,json,time,hashlib,shutil
from pathlib import Path
local=Path(__file__).parent;out=Path('/nas1/zhangzj26/dual_axis_hsi_uda/results/relation_class_risk_diagnostic_v2');env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='4',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1')
for script in ('run.py','score.py'):
 result=subprocess.run([sys.executable,'-u',str(local/script)],env=env,cwd=str(local))
 if result.returncode:print('STOP_DIAGNOSTIC_FAILURE',script,result.returncode,flush=True);sys.exit(result.returncode)
 if script=='run.py' and not (out/'GT_SCORING_BARRIER.json').exists():print('SOURCE_PREFLIGHT_NO_GO_NO_TARGET_SCORING',flush=True);sys.exit(0)
snap=out/'code_snapshot';snap.mkdir();hashes={}
for p in local.glob('*.py'):shutil.copy2(p,snap/p.name);hashes[str(snap/p.name)]=hashlib.sha256(p.read_bytes()).hexdigest()
(out/'code_hashes.json').write_text(json.dumps(hashes,indent=2)+'\n')

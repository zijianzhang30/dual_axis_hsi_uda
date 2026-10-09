import ast,os,sys,subprocess,time,shutil
from pathlib import Path
from common import OUT,PREVIOUS,BASE,SEEDS,sha,dump
assert not (OUT/'launch.json').exists()
for p in OUT.glob('*.py'):ast.parse(p.read_text())
dump(OUT/'provenance.json',{'scripts_sha256':{p.name:sha(p) for p in OUT.glob('*.py')},'previous_fixed_inputs_sha256':sha(PREVIOUS/'fixed_inputs.json'),'protocol_sha256':sha(OUT/'PROTOCOL.txt'),'checkpoint_sha256':{str(s):sha(BASE/'houston'/f'C_{s}/checkpoints/epoch_200.pth') for s in SEEDS}})
local=Path('/home/zhangzj26/dual_axis_hsi_uda/investigations')/OUT.name;local.mkdir(exist_ok=True)
for p in OUT.glob('*.py'):shutil.copy2(p,local/p.name)
shutil.copy2(OUT/'PROTOCOL.txt',local/'PROTOCOL.txt');link=Path('/home/zhangzj26/dual_axis_hsi_uda/results')/OUT.name
if not link.exists():link.symlink_to(OUT,target_is_directory=True)
else:assert link.resolve()==OUT.resolve()
env=os.environ.copy();env.update(PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
with (OUT/'launcher.log').open('x') as log:p=subprocess.Popen([sys.executable,'-u',str(OUT/'launch.py')],cwd=OUT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
dump(OUT/'launch.json',{'pid':p.pid,'seeds':SEEDS,'GPUs':[3,6,7],'epoch':200,'read_only_backbone':True,'Pavia_modified':False,'started':time.time()});print('Early F3D diagnostic controller',p.pid,flush=True)

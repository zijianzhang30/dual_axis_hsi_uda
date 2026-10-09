import ast,os,sys,subprocess,time,shutil
from pathlib import Path
from common import OUT,BASE,dump,sha
assert (OUT/'fixed_inputs.json').exists() and not (OUT/'launch.json').exists()
for p in OUT.glob('*.py'):ast.parse(p.read_text())
dump(OUT/'provenance.json',{'scripts_sha256':{p.name:sha(p) for p in OUT.glob('*.py')},'data_sha256':{p.name:sha(p) for p in (OUT/'data').glob('*.npy')},'protocol_sha256':sha(OUT/'PROTOCOL.txt'),'frozen_model_common_sha256':sha(BASE/'common.py'),'new_backbone_training':False})
local=Path('/home/zhangzj26/dual_axis_hsi_uda/investigations')/OUT.name;local.mkdir(exist_ok=False)
for p in OUT.glob('*.py'):shutil.copy2(p,local/p.name)
shutil.copy2(OUT/'PROTOCOL.txt',local/'PROTOCOL.txt');(Path('/home/zhangzj26/dual_axis_hsi_uda/results')/OUT.name).symlink_to(OUT,target_is_directory=True)
env=os.environ.copy();env.update(PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
with (OUT/'launcher.log').open('x') as log:p=subprocess.Popen([sys.executable,'-u',str(OUT/'launch.py')],cwd=OUT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
dump(OUT/'launch.json',{'pid':p.pid,'GPUs':[3,6,7],'seeds':[2100,2101,2102],'epoch':200,'new_backbone_training':False,'Pavia_modified':False,'started':time.time()});print('Read-only stem/value controller',p.pid,flush=True)

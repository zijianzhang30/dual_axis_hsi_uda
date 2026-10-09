import pathlib,ast,json,os,sys,time,subprocess,shutil
from common import OUT,BASE,SEEDS,sha,dump
assert json.loads((OUT/'verification.json').read_text())['passed'];assert not (OUT/'launch.json').exists()
for p in OUT.glob('*.py'):ast.parse(p.read_text())
dump(OUT/'provenance.json',{'study_sha256':{p.name:sha(p) for p in OUT.glob('*.py')},'snapshot_sha256':{str(p.relative_to(OUT/'snapshot')):sha(p) for p in (OUT/'snapshot').rglob('*.py')},'protocol_sha256':sha(OUT/'PROTOCOL.txt'),'verification_sha256':sha(OUT/'verification.json'),'seeds':list(SEEDS),'Pavia_modified':False})
local=pathlib.Path('/home/zhangzj26/dual_axis_hsi_uda/experiments')/OUT.name;local.mkdir(exist_ok=False)
for p in OUT.glob('*.py'):shutil.copy2(p,local/p.name)
shutil.copy2(OUT/'PROTOCOL.txt',local/'PROTOCOL.txt');(pathlib.Path('/home/zhangzj26/dual_axis_hsi_uda/results')/OUT.name).symlink_to(OUT,target_is_directory=True)
env=os.environ.copy();env.update(PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
with (OUT/'launcher.log').open('x') as log:p=subprocess.Popen([sys.executable,'-u',str(OUT/'launch.py')],cwd=OUT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
dump(OUT/'launch.json',{'pid':p.pid,'started':time.time(),'seeds':list(SEEDS),'epochs':200,'arms':['A','C'],'GPUs':[3,6,7],'Pavia_untouched':True,'no_target_during_training':True,'no_auto_followup':True});print('Source-only failure diagnosis controller',p.pid,flush=True)

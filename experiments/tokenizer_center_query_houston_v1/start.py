import pathlib,ast,json,os,sys,time,subprocess,shutil
from common import OUT,BASE,SEEDS,sha,dump
assert json.loads((OUT/'verification.json').read_text())['passed']
assert not (OUT/'launch.json').exists()
for p in OUT.glob('*.py'):ast.parse(p.read_text())
assert (BASE/'houston/C_2100/checkpoints/epoch_100.pth').exists()
dump(OUT/'provenance.json',{'study_sha256':{p.name:sha(p) for p in OUT.glob('*.py')},'snapshot_sha256':{str(p.relative_to(OUT/'snapshot')):sha(p) for p in (OUT/'snapshot').rglob('*.py')},'protocol_sha256':sha(OUT/'PROTOCOL.txt'),'verification_sha256':sha(OUT/'verification.json'),'paired_C_2100_epoch100_sha256':sha(BASE/'houston/C_2100/checkpoints/epoch_100.pth'),'stage2_seeds':list(SEEDS),'Pavia_modified':False})
local=pathlib.Path('/home/zhangzj26/dual_axis_hsi_uda/experiments/tokenizer_center_query_houston_v1');local.mkdir(exist_ok=False)
for p in OUT.glob('*.py'):shutil.copy2(p,local/p.name)
shutil.copy2(OUT/'PROTOCOL.txt',local/'PROTOCOL.txt')
pathlib.Path('/home/zhangzj26/dual_axis_hsi_uda/results/tokenizer_center_query_houston_v1').symlink_to(OUT,target_is_directory=True)
env=os.environ.copy();env.update(PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
with (OUT/'launcher.log').open('x') as l:p=subprocess.Popen([sys.executable,'-u',str(OUT/'launch.py')],cwd=OUT,env=env,stdout=l,stderr=subprocess.STDOUT,start_new_session=True)
dump(OUT/'launch.json',{'pid':p.pid,'started':time.time(),'seed':2100,'epochs':100,'arms':['Q0','Q1'],'GPUs':[3,6],'no_auto_followup':True,'Pavia_untouched':True})
print('Center Query matched Q0/Q1 startedGPU3/6,seed2100,fixed100; controller',p.pid)

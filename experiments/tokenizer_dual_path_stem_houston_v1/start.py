import pathlib,ast,json,os,sys,time,subprocess,shutil
from common import OUT,BASE,SEEDS,sha,dump
assert json.loads((OUT/'verification.json').read_text())['passed'];assert not (OUT/'launch.json').exists()
for p in OUT.glob('*.py'):ast.parse(p.read_text())
dump(OUT/'provenance.json',{'study_sha256':{p.name:sha(p) for p in OUT.glob('*.py')},'snapshot_sha256':{str(p.relative_to(OUT/'snapshot')):sha(p) for p in (OUT/'snapshot').rglob('*.py')},'protocol_sha256':sha(OUT/'PROTOCOL.txt'),'verification_sha256':sha(OUT/'verification.json'),'paired_C_2100_epoch100_sha256':sha(BASE/'houston/C_2100/checkpoints/epoch_100.pth'),'stage2_seeds':list(SEEDS),'Pavia_modified':False})
local=pathlib.Path('/home/zhangzj26/dual_axis_hsi_uda/experiments/tokenizer_dual_path_stem_houston_v1');local.mkdir(exist_ok=False)
for p in OUT.glob('*.py'):shutil.copy2(p,local/p.name)
shutil.copy2(OUT/'PROTOCOL.txt',local/'PROTOCOL.txt');pathlib.Path('/home/zhangzj26/dual_axis_hsi_uda/results/tokenizer_dual_path_stem_houston_v1').symlink_to(OUT,target_is_directory=True)
env=os.environ.copy();env.update(PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
with (OUT/'launcher.log').open('x') as l:p=subprocess.Popen([sys.executable,'-u',str(OUT/'launch.py')],cwd=OUT,env=env,stdout=l,stderr=subprocess.STDOUT,start_new_session=True)
dump(OUT/'launch.json',{'pid':p.pid,'started':time.time(),'method':'MatchedDualPathStem','stage1_seed':2100,'stage1_epochs':100,'stage1_arms':['N0','N1'],'stage1_GPUs':[6,7],'stage2':'only OA/AA both improve vs fixed C100','stage2_seeds':list(SEEDS),'stage2_epochs':200,'Pavia_untouched':True})
print('Dual path stem audit passed; Stage1 matched N0/N1 seed2100 fixed100 started, controller',p.pid)

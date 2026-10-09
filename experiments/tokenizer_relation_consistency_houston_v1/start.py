import pathlib,ast,json,os,sys,subprocess,time,shutil
from common import OUT,BASE,SEEDS,sha,dump
assert json.loads((OUT/'preflight.json').read_text())['RI_preflight_passed']
assert not (OUT/'launch.json').exists()
for p in OUT.glob('*.py'):ast.parse(p.read_text())
assert all((BASE/'houston'/f'C_{s}/frozen_manifest.json').exists() for s in SEEDS)
dump(OUT/'provenance.json',{'study_sha256':{p.name:sha(p) for p in OUT.glob('*.py')},'snapshot_sha256':{str(p.relative_to(OUT/'snapshot')):sha(p) for p in (OUT/'snapshot').rglob('*.py')},'protocol_sha256':sha(OUT/'PROTOCOL.txt'),'preflight_sha256':sha(OUT/'preflight.json'),'C0_config_hashes':{str(s):sha(BASE/'houston'/f'C_{s}/config.json') for s in SEEDS},'lambda_rel':.1,'seeds':list(SEEDS),'C2_nonzero_gradient_preflight_passed':False,'no_Pavia_mutation':True})
local=pathlib.Path('/home/zhangzj26/dual_axis_hsi_uda/experiments/tokenizer_relation_consistency_houston_v1');local.mkdir(exist_ok=True)
for p in OUT.glob('*.py'):shutil.copy2(p,local/p.name)
shutil.copy2(OUT/'PROTOCOL.txt',local/'PROTOCOL.txt')
link=pathlib.Path('/home/zhangzj26/dual_axis_hsi_uda/results/tokenizer_relation_consistency_houston_v1');
if not link.exists():link.symlink_to(OUT,target_is_directory=True)
e=os.environ.copy();e.update(PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
with (OUT/'launcher.log').open('a') as f:p=subprocess.Popen([sys.executable,'-u',str(OUT/'launch.py')],cwd=OUT,env=e,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
dump(OUT/'launch.json',{'pid':p.pid,'started':time.time(),'C0':'reuse official C','C1_runs':3,'C2_runs':0,'epochs':200,'GPUs':[0,4,5],'wait_for_repaired_Pavia_resources':True})
print('C1 preflight passed, queue controller',p.pid,'C2 blocked by intrinsic equivariance')

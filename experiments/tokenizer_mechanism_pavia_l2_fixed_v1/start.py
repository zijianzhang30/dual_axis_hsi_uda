import pathlib,json,os,sys,time,subprocess,shutil,ast
from common import OUT,BASE,SEEDS,ARMS,sha,dump
assert json.loads((OUT/'verification.json').read_text())['passed']
assert json.loads((OUT/'normalization_audit.json').read_text())['passed']
assert not (OUT/'launch.json').exists()
for p in OUT.glob('*.py'):ast.parse(p.read_text())
provenance={'seeds':list(SEEDS),'variants':list(ARMS),'epochs':200,'bands':102,'patch':13,'protocol':'repaired per-pixel spectral L2 / formal fixed epoch200','old_raw_diagnostic_base':str(BASE),'old_capacity_G':str(OUT.parent/'tokenizer_capacity_g_v1'),'original_provenance_sha256':sha(BASE/'provenance.json'),'study_sha256':{p.name:sha(p) for p in OUT.glob('*.py')},'snapshot_sha256':{str(p.relative_to(OUT/'snapshot')):sha(p) for p in (OUT/'snapshot').rglob('*.py')},'normalization_audit_sha256':sha(OUT/'normalization_audit.json'),'verification_sha256':sha(OUT/'verification.json'),'protocol_sha256':sha(OUT/'PROTOCOL.txt'),'no_HyRANK':True,'no_target_best':True}
dump(OUT/'provenance.json',provenance)
registry={'classification':'RAW-INPUT DIAGNOSTIC ONLY','applies_to':[str(BASE/'pavia'),str(OUT.parent/'tokenizer_capacity_g_v1/pavia')],'not_included_in_formal_averages':True,'original_artifacts_preserved':True}
dump(OUT/'old_raw_results_registry.json',registry)
local=pathlib.Path('/home/zhangzj26/dual_axis_hsi_uda/experiments/tokenizer_mechanism_pavia_l2_fixed_v1')
for p in OUT.glob('*.py'):shutil.copy2(p,local/p.name)
shutil.copy2(OUT/'PROTOCOL.txt',local/'PROTOCOL.txt')
env=os.environ.copy();env.update(PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
with (OUT/'launcher.log').open('x') as log:p=subprocess.Popen([sys.executable,'-u',str(OUT/'launch.py')],cwd=OUT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
dump(OUT/'launch.json',{'pid':p.pid,'started':time.time(),'seeds':list(SEEDS),'variants':list(ARMS),'runs':15,'epochs':200,'GPUs':[0,4,5],'wait_for_existing_G':True,'HyRANK':False,'no_auto_retry':True})
print('Audit passed; locked 15-run normalized Pavia queue, PID',p.pid)

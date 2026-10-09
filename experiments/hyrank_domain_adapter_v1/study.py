import importlib.util,json,hashlib,shutil
from pathlib import Path
import torch
ROOT=Path('/home/zhangzj26/dual_axis_hsi_uda');OUT=Path('/nas1/zhangzj26/dual_axis_hsi_uda/results/hyrank_domain_adapter_v1')
R0=OUT.parent/'hyrank_joint_lmmd_eq9_v1';SEEDS=(2100,);ARMS=('R1','R2')
def module(name,p):
 spec=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
parent=module('domain_adapter_R0_frozen',R0/'experiment_code/study.py')
BASE=parent.BASE;PRIOR=parent.PRIOR;B1=parent.B1;random=parent.random
PLAN={'stage':'single seed2100 fixed epoch100 exploratory; report then STOP, no automatic3seed',
 'R0':'reuse current unified C+JointBN+CE+LMMD+Eq9 seed2100 checkpoint100; do not retrain',
 'R1':'parallel shared adapter, bias-free64->32->GELU->64 on each of3 FFN blocks',
 'R2':'parallel separate source/target adapters, EACH bias-free64->16->GELU->64 on each block',
 'capacity':'R1/R2 each12288 extra TOTAL trainable params; ALL participate; R1 each domain uses12288, R2 each domain6144, so per-domain width differs and must be disclosed; no dummy parameters',
 'initialization':'up weight zero, baseline function unchanged; separate CPU RNG seed+73000; R2 source/target down starting weights identical',
 'routing':'source/target first two actual self FFN calls only; unused coupled products no adapters; eval target adapter only; no source scene required',
 'recipe':'EXACT parent unified C config and LMMD/Eq9 kernel; 80updates/epoch, patch13,batch128,SGD.01,FullJointBN.19; LMMD ramp denominator stays200 even though exploratory endpoint100',
 'no_new_loss_or_other_structure':True,'wait_for_current_six_runs':'run only after all unified A/C six 200epoch manifests complete; no interruption',
 'formal_exploratory_endpoint':100,'target_GT':'offline scoring only after prediction freezing; no target-best selection',
 'decision':'report R2-R1 and R2-R0 OA/AA, collapse and adapter gradients; NO automatic formal expansion',
 'novelty':'shared/private and domain-conditioned learning prior art exists; this is hypothesis control, not established second contribution',
 'references':['https://proceedings.neurips.cc/paper/2016/hash/45fbc6d3e05ebd93369ce542e8f2322d-Abstract.html','https://arxiv.org/abs/2410.10442']}
def dump(p,v):Path(p).write_text(json.dumps(v,indent=2)+'\n')
sha=parent.sha;shared_hash=parent.shared_hash;geometry=parent.geometry;environment=parent.environment;lmmd=parent.lmmd

def build(c,args,arm):
 m,unused,initial=parent.build(c,args,'C');module('frozen_domain_adapter',OUT/'adapter.py').install(m,arm,args.seed)
 return m,unused,initial

def verify():
 parent.verify()
 for p,h in json.loads((OUT/'code_manifest.json').read_text()).items():assert sha(p)==h,p

def prepare():
 OUT.mkdir(exist_ok=False);dump(OUT/'frozen_plan.json',PLAN)
 shutil.copy2(Path(__file__).parent/'adapter.py',OUT/'adapter.py')
 shutil.copy2(R0/'pairing_kernel.py',OUT/'pairing_kernel.py')
 code=OUT/'experiment_code';code.mkdir()
 for p in Path(__file__).parent.glob('*.py'):shutil.copy2(p,code/p.name)
 dump(OUT/'code_manifest.json',{str(p):sha(p) for p in list(code.glob('*.py'))+[OUT/'adapter.py',OUT/'pairing_kernel.py']})
 (ROOT/'results'/OUT.name).symlink_to(OUT,target_is_directory=True)
 dump(OUT/'historical_check.json',{'same_architecture_completed_controls_found':False,'R0_only_reused':str(R0/'C_2100/checkpoints/epoch_100.pth'),'do_not_mix_prior_CE_only_or_BiDAfull_R0':True})
if __name__=='__main__':prepare()

import hashlib,importlib.util,json
from pathlib import Path
ROOT=Path('/home/zhangzj26/dual_axis_hsi_uda');B=Path('/nas1/zhangzj26/dual_axis_hsi_uda/results');OUT=ROOT/'results/houston_relation_identity_dual_evidence_3seed_v1';CODE=ROOT/'experiments/houston_relation_identity_dual_evidence_3seed_v1';SEEDS=(2100,2101,2102)
PLAN={'seeds':list(SEEDS),'epochs':200,'grid':list(range(10,201,10)),'training':'fresh relation and spectral experts from original initialization, two disjoint optimizers, independent source CE; source raw center detached; no spectral gradients into relation','R0':'unchanged original C + Full Joint BN + original 3-layer Transformer;378011parameters','R1':'exact prior LayerNorm48 + Linear48to64 + GELU + Linear64to7;3687parameters,no dropout or BN','R2':'.5*softmax(rel)+.5*softmax(spec),T=1','R3':'normalize(logsoftmax(rel)+logsoftmax(spec)); same previous float64 stable implementation; equivalent softmax(rel_logits+spec_logits)','init':'same private spectral CPU fork_rng seed+146000, preserve original CPU/CUDA RNG','protocol':'source2393/val127/target52901 native95%split; bs128/domain,4workers,originalaugmentation/sharedloadergenerator;18 steps/epoch,3600total;bothSGD lr.01 momentum0 decay0 no scheduler;FullJointBN momentum.19 one paired relation forward eachstep;sourceval epoch1/every10 single native val loader iteration evaluating both models','equivalence':'each epoch data/augmentation/RNG hashes,BN/update/block/classifier calls match original C. Fresh relation every10 checkpoint bitwise matches archived C. seed2100 spectral epoch100 matches previous single-seed checkpoint.','freeze':'all3seed completed200, all20grid R0/R1/R2/R3 logits/predictions/coords/agreement hash-frozen before new-study target GT scoring','formal':'fixed200 only; three-seed sample std ddof1','Go':'R3 mean deltaOA>0, at least2/3 seed deltaOA>0, mean deltaAA>=0 and at least2/3 seed deltaAA>=0; R3 no collapse95(max predicted share>=.95) or zero-recall in any seed','AA_interpretation':'prelocked operationalization of not systematically decreasing: nonnegative average and majority same-seed changes, all per-class results reported','best':'same 20 checkpoints per arm per seed; max targetOA diagnostic only, AA/Kappa at same OA-best checkpoint; never used for formal gate','no_target_training':'no target pseudo-label/GT,LMMD,gate,temperature or fusion-weight selection. Target candidate mask remains original; target class arrays zero before batch iteration','no_automatic_followups':'no extra variants, no automatic Pavia/HyRANK; discuss formal/cross-dataset stage only after completed result','prior_metrics':'previous Houston100 and originalC200 scores known; new run config fixed beforetraining, target labels excluded from this training/evidence construction'}
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def dump(p,x):Path(p).write_text(json.dumps(x,indent=2)+'\n')
def module(n,p):
 spec=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def environment():return module('dual_3seed_frozen_protocol',B/'cross_benchmark_lmmd_v1/snapshot/common.py')
def baseline(seed):return B/'tokenizer_mechanism_v2/houston'/('C_%d'%seed)
def verify():
 environment().verify_files()
 for name in ('code_manifest.json','baseline_manifest.json'):
  for p,h in json.loads((OUT/name).read_text()).items():assert sha(p)==h,p

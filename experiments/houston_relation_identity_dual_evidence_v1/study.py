import hashlib,importlib.util,json
from pathlib import Path
ROOT=Path('/home/zhangzj26/dual_axis_hsi_uda');B=Path('/nas1/zhangzj26/dual_axis_hsi_uda/results');OUT=ROOT/'results/houston_relation_identity_dual_evidence_v1';CODE=ROOT/'experiments/houston_relation_identity_dual_evidence_v1';SEED=2100
PLAN={'seed':2100,'epochs':100,'one_new_training':'spectral identity expert only','R0':'reuse exact original C S0_2100 fixed epoch100 relation expert; no parameter,forward architecture,BN buffer or classifier change','R1':'center actual preprocessed/augmented48-band spectrum only; LayerNorm48 + Linear48to64 + GELU + Linear64to7; 3687 parameters; no dropout/BN','R2':'p=.5*p_rel+.5*p_spec, fixed T1','R3':'logp=logsoftmax(rel)+logsoftmax(spec), normalize with logsumexp; equivalent softmax(rel_logits+spec_logits),fixed T1','source_training':'source GT CE only; independent SGD lr.01 momentum0 decay0 no scheduler; detach center input; optimizer contains spectral parameters exclusively; relation expert eval,requires_grad=False','data':'original source2393/source-val127/target52901; paired128,18 updates/epoch; original shared loader generator/4workers/augmentations; native source-val epoch1/every10; target label values zero before iteration','RNG':'private spectral initialization CPU fork_rng seed+146000. No spectral stochastic layer. Replay frozen S0 per-step pre/post CPU/CUDA states to preserve original data/augmentation/global stream without running or updating relation expert. Assert all100 epoch batch/RNG hashes match S0.','formal':'epoch100 only; all training epochs completed; no target-best scoring/grid selection','freeze':'R0 reused frozen logits; R1 logits; R2/R3 normalized log probabilities; all predictions,coordinates,agreement/disagreement hashed together before loading target labels for this study','Go':'at least one of R2/R3 has fixed100 OA delta>=.5pp, AA delta>=-.5pp, no new zero-recall class, no new global collapse95(max prediction share>=.95)','per_class':'all7 present target classes reported; no excluding small C4; no extra 5pp class-damage gate added to user specified collapse rule','diagnostic':'R0 error recovery / original correct damage; agreement and disagreement expert accuracy,per class C1/C5/C3/C6/C7; class-specific predicted confusion. Oracle any-expert accuracy descriptive only,never used for fusion.','no_tuning':'weights,T,hidden dim,threshold,optimizer,architecture fixed before this training; no learned gate/MoE or new fusion','followups':'no automatic experimental extension; if No-Go stop this dual-evidence fusion direction','historical_target_metrics':'previous studies already scored R0 and other variants; do not claim untouched Houston validation. Target labels excluded from this training and fusion construction.'}
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def dump(p,x):Path(p).write_text(json.dumps(x,indent=2)+'\n')
def module(n,p):
 spec=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def environment():return module('dual_evidence_frozen_protocol',B/'cross_benchmark_lmmd_v1/snapshot/common.py')
def previous():return ROOT/'results/houston_persistent_spectral_anchor_v1'
def relation():return previous()/'S0_2100'
def verify():
 environment().verify_files()
 for name in ('code_manifest.json','baseline_manifest.json'):
  for p,h in json.loads((OUT/name).read_text()).items():assert sha(p)==h,p

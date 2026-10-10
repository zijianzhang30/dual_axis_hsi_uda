import hashlib,importlib.util,json
from pathlib import Path
ROOT=Path('/home/zhangzj26/dual_axis_hsi_uda'); B=Path('/nas1/zhangzj26/dual_axis_hsi_uda/results'); OUT=ROOT/'results/houston_final_cls_spectral_anchor_v1'; CODE=ROOT/'experiments/houston_final_cls_spectral_anchor_v1'; ARMS=('S3',); SEED=2100
PLAN={'seed':2100,'epochs':100,'arms':['S3'],'baseline':'reuse completed houston_persistent_spectral_anchor_v1 S0_2100; do not retrain','source_CE_only':True,'base':'Joint BN + C tokenizer + original 3-layer Transformer','S3':'once after final Transformer block, before original final LayerNorm; CLS += alpha*sigmoid(Linear128to1(cat(CLS,s)))*s; T1..T4 never directly receive anchor; no further self-attention after injection','encoder':'same Linear48to64 + LayerNorm64 and same initial weights as S2','gate':'same Linear128to1 form and initial first gate as S2; alpha single scalar zero','value_map':'identity, same as S2; no new Wv tuning','source_training':'main model trainable, source-self CE only; relation tokens are unchanged by direct injection, but backbone may learn differently through CE gradients','init':'independent CPU fork_rng seed+126000, preserve all original CPU/CUDA RNG','protocol':'unchanged source2393/val127/target52901; source/target128;18 SGD lr.01 updates/epoch;momentum0 decay0 no scheduler;4workers,original augmentation;source-val epoch1/every10; native4 training outputs','grid':list(range(10,101,10)),'formal':'fixed100 only','Go':'S3-S0 OA>0.5pp, AA>=-0.5pp; no new zero-recall/collapse95; no individual class recall drop>=5pp','class_damage_threshold':'predefined 5pp inclusive across all7 classes, including small C4; no post-hoc exclusion','best':'same ten checkpoints, target-best descriptive only, not used for Go','no_followups':'exactly one new S3 seed2100 training; no hyperparameter tuning or automatic 3seed200; if No-Go stop spectral gate/token variants','interpretation':'S2 gain does not prove spectral rescue or relation pollution; this is controlled injection-location screening, not causal confirmation'}
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def dump(p,x):Path(p).write_text(json.dumps(x,indent=2)+'\n')
def module(n,p):
 spec=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def environment():return module('anchor_frozen_protocol',B/'cross_benchmark_lmmd_v1/snapshot/common.py')
def previous():return ROOT/'results/houston_persistent_spectral_anchor_v1'
def baseline():return B/'tokenizer_mechanism_v2/houston/C_2100'
def verify():
 environment().verify_files()
 for name in ('code_manifest.json','baseline_manifest.json'):
  for p,h in json.loads((OUT/name).read_text()).items():assert sha(p)==h,p

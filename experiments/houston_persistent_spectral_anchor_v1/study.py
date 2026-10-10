import hashlib,importlib.util,json
from pathlib import Path
ROOT=Path('/home/zhangzj26/dual_axis_hsi_uda'); B=Path('/nas1/zhangzj26/dual_axis_hsi_uda/results'); OUT=ROOT/'results/houston_persistent_spectral_anchor_v1'; CODE=ROOT/'experiments/houston_persistent_spectral_anchor_v1'; ARMS=('S0','S1','S2'); SEED=2100
PLAN={'seed':2100,'epochs':100,'arms':list(ARMS),'source_CE_only':True,'base':'Joint BN + C tokenizer + original 3-layer Transformer','encoder':'Linear(48,64,bias=True)+LayerNorm(64); center actual augmented/preprocessed input[:,0,:,6,6]','S1':'append ordinary spectral token after [CLS,T1,T2,T3,T4]; preserve original five positional embeddings; spectral positional offset fixed zero','S2':'anchor encoder computed once; before each block tokens read anchor with x += alpha[l]*sigmoid(Linear128to1(cat(x,anchor)))*anchor; read-only anchor reused unchanged; alpha init zero; no new dropout or BN','init':'independent CPU fork_rng seed+126000; original base initialization unchanged; S1/S2 identical encoder initialization','protocol':'original source2393/val127/target52901; paired128,18 SGD updates/epoch; lr.01 no momentum/decay/schedule; native augmentations,4workers,source-val epoch1/every10; original4 training outputs,source-selfCE only','S1_RNG':'S0 records pre/post CPU+CUDA RNG for every main forward; S1 enters same pre-state, runs length6 forward, restores corresponding S0 post-state. Sequence-length dropout masks necessarily differ; global RNG/data streams remain exactly paired. No extra main forward or BN updates.','S2_RNG':'native length5 random consumption; no replay, match original C every epoch','grid':list(range(10,101,10)),'formal':'fixed100 only','Go':'S2-S0 fixed100 OA>=0.5pp; AA>=-0.5pp; no newly zero-recall class and no new collapse95 (max prediction share>=.95)','best':'same-grid target-best diagnostic only; never used for Go or tuning','prohibited':'target labels in loss/tuning,extra losses,extra experiments,automatic 3seed200 extension'}
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def dump(p,x):Path(p).write_text(json.dumps(x,indent=2)+'\n')
def module(n,p):
 spec=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def environment():return module('anchor_frozen_protocol',B/'cross_benchmark_lmmd_v1/snapshot/common.py')
def baseline():return B/'tokenizer_mechanism_v2/houston/C_2100'
def verify():
 environment().verify_files()
 for p,h in json.loads((OUT/'code_manifest.json').read_text()).items():assert sha(p)==h,p

import hashlib, importlib.util, json
from pathlib import Path
B=Path('/nas1/zhangzj26/dual_axis_hsi_uda/results')
ROOT=Path('/home/zhangzj26/dual_axis_hsi_uda')
OUT=B/'houston_scene_conditioned_attention_v1'
SEED=2100
ARMS=('S1','S2')
PLAN={'seed':2100,'epochs':100,'grid':list(range(10,101,10)),
 'S0':'reuse exact formal C seed2100 checkpoints and matching prediction grid through100',
 'S1':'first self-attention conditioned on source batch summary',
 'S2':'first self-attention conditioned on target batch summary',
 'model':'FullJointBN+C nonlinear normalized center-relative tokenizer+original3selfTransformer+original classifier',
 'loss':'source CE only; no LMMD/MCC/Eq9/distillation/EMA/reconstruction/adapter',
 'summary':'actual first-attention LayerNorm inputs, excludeCLS; population channel mean/std across batch and4token slots; 128dimensions; training retainsgradient',
 'modulator':'Linear128->32/ReLU/Linear32->64; privateCPUfork seed+181000; outputlayer weights/bias zero; sameinitials S1/S2',
 'metric':'m=1+0.1*tanh(modulator(summary)); reshape64 as8heads x8headchannels; scores=((Q*m)@K.T)/sqrt(8); same m both domains; V unchanged',
 'scope':'first block self paths only; native unused coupled attention unchanged and calls/dropout ordering retained for original RNG; no coupled loss',
 'protocol':'formal Houston normband per-pixelL2,48bands,13patch,batch128,SGD.01/momentum0/decay0,persistent,18updates/epoch; sourceval1/every10 unchanged',
 'evaluation_summary':'eval/frozenBN deterministic1024 actual source-train or existingtargetcandidate samples; noaugmentation; same locked source/target indices botharms; recompute current-model summary before sourceval/checkpoint; savefixedbuffer; no averaging historicmodel summaries',
 'summary_cohort_RNG':'private numpy RandomState(seed+182000), independent source/target choices; sortedindices; labels zero; inheritedtargetnonzero mask permission only',
 'evaluation':'fixed saved summary,original native target,target test pairing; target-onlyTransformer; same10:10:100 grid; sourceval loader unchanged',
 'diagnostics':'firstbatch summary/modulation/attentionentropy/conditiongradient; first128 evalinputs fixed-vs-batch logits comparison; zero-init expected target-summary gradient zero firststep then nonzero afteroutputlayer learns',
 'formal':'fixed100 exploratory single seed, not formal3seed result; target-best oracle only AA/Kappa sameOA-best checkpoint',
 'go':'S2 must improve OA andAA vs both S0 andS1 atfixed100, reviewclasscoverage; best-only improvements reported separately, noautomaticfollowup',
 'new_trainings':2,'automatic_sweep':False,'automatic_benchmark_expansion':False}
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for x in iter(lambda:f.read(1048576),b''):h.update(x)
 return h.hexdigest()
def dump(p,v):Path(p).write_text(json.dumps(v,indent=2)+'\n')
def module(n,p):
 spec=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def env():return module('scene_frozen_common',B/'cross_benchmark_lmmd_v1/snapshot/common.py')
def baseline():return B/'tokenizer_mechanism_v2/houston/C_2100'
def verify():
 env().verify_files()
 for name in ('code_manifest.json','baseline_manifest.json'):
  for p,h in json.loads((OUT/name).read_text()).items():assert sha(p)==h,p

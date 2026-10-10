import hashlib,importlib.util,json
from pathlib import Path
B=Path('/nas1/zhangzj26/dual_axis_hsi_uda/results')
ROOT=Path('/home/zhangzj26/dual_axis_hsi_uda')
OUT=B/'houston_talking_heads_v1'
ARMS=('TH1','TH2')
PLAN={'seed':2100,'epochs':100,'grid':list(range(10,101,10)),
 'TH0':'reuse formalC2100 native dotproduct selfattention same100epoch endpoint/grid',
 'TH1':'firstselfattention diagonal pre-softmax andpost-softmax headscaling;16 trainable scalars initializedones; noheadmixing',
 'TH2':'firstselfattention full pre-softmax andpost-softmax8x8headlinearmatrices;128trainableweights initializedidentity; no bias',
 'formula':'L=QK.T/sqrt(8); Lpre[o]=sum_h Ppre[o,h]*L[h]; A=softmax_key(Lpre); Apost[o]=sum_h Ppost[o,h]*A[h]; output=Apost*V (originalsameheadV),originaloutputprojection follows',
 'TH1_formula':'Lpre[h]=dpre[h]*L[h]; Apost[h]=dpost[h]*softmax_key(Lpre[h]); dpre,dpost startones; unconstrainedlinear scales,not exponentiated',
 'postsoftmax':'no secondsoftmax,renormalization,positivityconstraint or extraheadactivation; weights can become negative/rowsums not1; report explicitly; Shannonentropy onlyBEFOREpostheadmixing',
 'scope':'firstlayer source/targetselfpaths only,shared pre/postparameters; allCLS/tokenpairs included5x5; nativeunusedcoupled attention unchanged/dropoutorder retained; no coupled objective',
 'network':'FullJointBN+C normalizedcenter-relative nonlinear tokenizer+original3selfTransformer+originalclassifier; QKV/V/FFN/pos/CLS unchanged',
 'loss':'source CE ONLY; no cosine/scenecache/LMMD/MCC/Eq9/adapter/EMA/distillation/reconstruction/TalkingHeadsotherlayers',
 'protocol':'Houston48bands normband per-pixelL2,13patch,batch128,persistentSGD.01/momentum0/weightdecay0,100x18=1800updates; native sourceval1/every10',
 'pairing':'baseinit/sourcepool/dataindices/augment/mainCPU-CUDA RNG matchnativeC each100epochs; deterministiceye/ones init consumesnoRNG',
 'capacity':'TH1adds16,TH2adds128;112extraoffdiagonals TH2. This isolatespresenceofheadmixing withDIFFERENTtrainableparametercounts,not strictcapacitymatched; no dummyparams',
 'reference':'CaiT officialAttention_talking_head headprojection order; no projectionbiases or randominit; noCaiTdepth/LayerScale/classattention borrowed',
 'evaluation':'fixed100exploratory; target-bestoracle same10:10:100grid,AA/Kappa sameOA-bestcheckpoint; targetGTonlyafterallpredictionfilesfrozen',
 'target_permissions':'inheritednative nonzeroGTcandidate samplingmask only, targetclassvalueszero fortrain/eval; nottargetfree',
 'diagnostics':'pre/postdiag andoffdiagnorms/gradient, matrices, beforepostmixheadentropy, postmixnegativeweightfraction/rowmass,sourceval,perclassrecall/collapse/time/params',
 'screen':'TH2 must improvefixed100OA/AA vsTH0 andTH1,reviewclasscoverage;best-onlysignals separate,single-seednotstabilityevidence',
 'automatic_followup':False,'new_training_count':2}
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for x in iter(lambda:f.read(1048576),b''):h.update(x)
 return h.hexdigest()
def dump(p,v):Path(p).write_text(json.dumps(v,indent=2)+'\n')
def module(n,p):
 spec=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def env():return module('talking_frozen_common',B/'cross_benchmark_lmmd_v1/snapshot/common.py')
def baseline():return B/'tokenizer_mechanism_v2/houston/C_2100'
def verify():
 env().verify_files()
 for name in ('code_manifest.json','baseline_manifest.json'):
  for p,h in json.loads((OUT/name).read_text()).items():assert sha(p)==h,p

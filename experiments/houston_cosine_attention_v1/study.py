import hashlib,importlib.util,json
from pathlib import Path
B=Path('/nas1/zhangzj26/dual_axis_hsi_uda/results')
ROOT=Path('/home/zhangzj26/dual_axis_hsi_uda')
OUT=B/'houston_cosine_attention_v1'
ARMS=('N1','N2')
PLAN={'seed':2100,'epochs':100,'grid':list(range(10,101,10)),
 'N0':'reuse exact formalC2100 same100epoch endpoint and10:10:100 predictiongrid',
 'N1':'first selfattention scores exp(a_h)*D_h, a_h=0; D=QK.T/sqrt(8)',
 'N2':'first selfattention scores s_h0*exp(a_h)*C_h, a_h=0; C=normalize(Q)normalize(K).T; NO additional sqrt(8) divisor',
 'calibration':'s_h0=globalRMS(D-rowmean_jD)/globalRMS(C-rowmean_jC); eachhead separately; FP64 squares/count across all1024source samples and5x5query/key slots includingCLS; NOT average batchratios',
 'calibration_source':'private RandomState2100+192000 picks1024 originalsourceTRAIN indices withoutreplacement sorted; noGTtarget, noaugmentation; originalinitialmodel eval; BNbuffers and mainstream RNG unchanged',
 'calibration_denominator_min':1e-8,'QK_normalization_eps':1e-12,
 'calibration_failure':'denominator<1e-8, nonpositive or nonfinite scale ->STOP,no clamp,reselection or alternateformula',
 'scale_parameterization':'8 trainable a_h eacharm initializedzeros; unconstrained exp(a_h),no cap,no temperature sweep; finitegradient/params/scale checked eachstep; N2 s_h0 fixedbuffer',
 'scope':'first layer source/targetselfattention only; QKV/V/FFN/CLS/classifier/tokenizer/CNN unchanged; original unusedcoupled calculation anddropoutorder preserved, no coupled objective',
 'model':'FullJointBN+C normalizedcenter-relative nonlinear tokenizer+original3SelfTransformer',
 'loss':'source CE ONLY; no scene summary,LMMD,MCC,Eq9,adapters,TalkingHeads,distillation,EMA,completion',
 'protocol':'formalHouston48bands,perpixelL2normband,13patch,batch128,persistentSGD.01/momentum0/decay0;100x18=1800updates; sourceval1/every10exactoriginalcallbudget',
 'pairing':'originalbaseinit/dataindices/augment/mainCPU-CUDA RNG hashes matchnativeC each100epochs; N1/N2sameaddedzeroa initialization; no new randomdraws',
 'evaluation':'originalnative target,targettestpair,evalrunningBN; grid10:10:100; targetGTonlyafterallpredictionfilesfrozen;bestAA/Kappa fromsameOA-bestcheckpoint',
 'target_mask':'existing nativeGT-nonzero candidate sampling mask inherited and disclosed; targetclassvalueszero beforetrain/eval',
 'diagnostics':'firsttrainingbatch headscale/logscale/gradient,QKnorms,scoreRMS,attentionentropy; sourceval/perclassrecall/zero-recall/collapse/time/params',
 'screen':'N2 must improve fixed100OA/AA vsN0 andN1,reviewclasscoverage;best-onlysignalreportedseparately;single-seednotstabilityevidence',
 'automatic_followup':False,'new_training_count':2}
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for x in iter(lambda:f.read(1048576),b''):h.update(x)
 return h.hexdigest()
def dump(p,v):Path(p).write_text(json.dumps(v,indent=2)+'\n')
def module(n,p):
 spec=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def env():return module('cosine_frozen_common',B/'cross_benchmark_lmmd_v1/snapshot/common.py')
def baseline():return B/'tokenizer_mechanism_v2/houston/C_2100'
def verify():
 env().verify_files()
 for name in ('code_manifest.json','baseline_manifest.json','calibration_manifest.json'):
  p=OUT/name
  if name=='calibration_manifest.json' and not p.exists():continue
  for f,h in json.loads(p.read_text()).items():assert sha(f)==h,f

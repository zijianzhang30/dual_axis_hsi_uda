import pathlib,json,hashlib,numpy as np,csv
O=pathlib.Path(__file__).parent

def stats(v):
 x=np.asarray(v,float);return {'values':x.tolist(),'mean':float(x.mean()),'sample_std':float(x.std(ddof=1))}
def js_per_sample(p,q):
 p=np.asarray(p,np.float64);q=np.asarray(q,np.float64);m=(p+q)/2
 return .5*(p*(np.log(p)-np.log(m))+q*(np.log(q)-np.log(m))).sum(-1).mean(-1)
def main():
 d=json.loads((O/'summary.json').read_text());seeds=d['seeds'];rows=d['runs'];stable={};paired={};assign={}
 # Stable FP64 postprocessing of already saved FP32 weights; no inference/retraining.
 for arm in 'AC':
  stable[arm]={}
  for dom in ['source','target']:
   stable[arm][dom]={}
   for perturb in ['mild','band_affine']:
    v=[]
    for seed in seeds:
     f=O/f'{arm}_{seed}';p=np.load(f/f'{dom}_W.npy');q=np.load(f/f'{dom}_W_{perturb}.npy');v.append(float(js_per_sample(p,q).mean()))
    stable[arm][dom][perturb]=stats(v)
  assign[arm]={dom:{k:stats([rows[f'{arm}_{s}']['domains'][dom]['assignment'][k] for s in seeds]) for k in ['entropy_normalized_mean','token_cosine_mean','token_JS_mean','peak_mean','top10_mass_mean']} for dom in ['source','target']}
  assign[arm]['source_target_mean_map_JS']=stats([float(js_per_sample(np.load(O/f'{arm}_{s}/source_W.npy').mean(0)[None],np.load(O/f'{arm}_{s}/target_W.npy').mean(0)[None])[0]) for s in seeds])
 for dom in ['source','target']:
  paired[dom]={}
  for perturb in ['mild','band_affine']:
   v=[];ci=[]
   for seed in seeds:
    f=O/f'C_{seed}';a=O/f'A_{seed}'
    c=js_per_sample(np.load(f/f'{dom}_W.npy'),np.load(f/f'{dom}_W_{perturb}.npy'));aa=js_per_sample(np.load(a/f'{dom}_W.npy'),np.load(a/f'{dom}_W_{perturb}.npy'));diff=c-aa;v.append(float(diff.mean()));rng=np.random.default_rng(71000);boot=[]
    for _ in range(1000):boot.append(float(diff[rng.integers(0,len(diff),len(diff))].mean()))
    ci.append(np.percentile(boot,[2.5,97.5]).tolist())
   paired[dom][perturb]={**stats(v),'paired_patch_bootstrap95CI_by_seed':ci}
 gap={}
 for arm in 'AC':
  gap[arm]={}
  for stage in ['X_center','V_center','V_mean','V_local','T','Z']:
   gap[arm][stage]={k:stats([rows[f'{arm}_{s}']['domain_gaps'][stage][k] for s in seeds]) for k in ['relative_mean_distance','relative_covariance_gap','MMD_unbiased_RBF_multiscale','mean_cosine','mean_spectral_angle_radians','CORAL_loss']}
 mild_worse=all(x>0 for dom in ['source','target'] for x in paired[dom]['mild']['values']);affine_better=all(x<0 for dom in ['source','target'] for x in paired[dom]['band_affine']['values'])
 decision={'case':'C: stability is perturbation-specific; generic transferability explanation unsupported' if mild_worse else 'Mixed/undetermined: do not infer causal value bottleneck','mild_spectral_noise_C_less_stable_all3seeds_bothdomains':mild_worse,'band_affine_C_more_stable_all3seeds_bothdomains':affine_better,'CaseA_premise_supported':False,'new_value_module_authorized_by_diagnosis':False,'no_training_or_module_design':True,'interpretation':'C is approximately invariant to positive perband patch-shared affine change by construction; this does not confer robustness to pixelwise noise. Marginal V/T/Z domain gaps alone cannot locate a causal bottleneck; class-prior/content confounding and different trainedCNN weights remain.'}
 result={'stability_JS_FP64':stable,'paired_C_minus_A_JS':paired,'assignment_statistics':assign,'domain_gaps':gap,'decision':decision,'raw_summary_sha256':hashlib.sha256((O/'summary.json').read_bytes()).hexdigest(),'bootstrap_caveat':'Patch bootstrap CIs are descriptive; overlapping patches are spatially correlated, not independent scene replications. Only three trained seeds.'}
 (O/'REPORT_DATA.json').write_text(json.dumps(result,indent=2)+'\n')
 lines=['Houston A/C W-V-T-Z read-only bottleneck diagnosis; existing epoch200,three seeds','Fixed2048patches/domain; source complete train+val union, targetnative eval coordinates; identical probes/perturbations allmodels.','Eval runningBN,selfpairedimages;Z=normalized finalCLS before classifier. No optimizer,target-label metrics or new training.','Perturbations: mild alpha(.98,1.02)+pixel/bandGaussian beta.005; perband affine gain(.9,1.1)+offset(-.01,.01) shared acrosspatch. No renormalization after perturbation.','W JS from FP64 postprocessing; tiny negative rawFP32 JS rounding retained inrawdiagnostic files.','', 'W stability JS (lowerbetter),mean+/-sampleSD:']
 for dom in ['source','target']:
  for perturb in ['mild','band_affine']:
   z=stable['A'][dom][perturb];q=stable['C'][dom][perturb];lines.append(f"{dom} {perturb}: A {z['mean']:.8g}+/-{z['sample_std']:.8g}; C {q['mean']:.8g}+/-{q['sample_std']:.8g}; C-Aseedvalues={paired[dom][perturb]['values']}")
 lines+=['','Assignment source/target entropy,diversity,peak,meanmap difference:',json.dumps(assign),'','Dimensionless marginal domain gaps: Vmean64,Tflatten256,Z64. Rawunits acrossstages notdirectly comparable.']
 for stage in ['V_center','V_mean','V_local','T','Z']:
  for metric in ['relative_mean_distance','relative_covariance_gap','MMD_unbiased_RBF_multiscale']:
   a=gap['A'][stage][metric];c=gap['C'][stage][metric];lines.append(f"{stage} {metric}: A {a['mean']:.6f}+/-{a['sample_std']:.6f}; C {c['mean']:.6f}+/-{c['sample_std']:.6f}")
 lines+=['','Mean-distance normalization: mean shift dividedbypooledcenteredfeature RMS; relativeCORAL=||Cs-Ct||F/(||Cs||F+||Ct||F).','MMD unbiased512/domain,threeRBF scales(.5,1,2),pooledmedian squared distance per representation,100permutationnull; not target-accuracy tuned.','Input Xcenter sameallmodels. Source/target classpriors/content differ. Marginal gap evidence is not a causal claim of valuebottleneck.','Patch bootstrap isdescriptive withoverlap/spatialdependence; no independentbenchmark validation yet.','', 'Decision: '+json.dumps(decision),'Classification gains ofC remain established onHouston; revise mechanism wording to center-conditioned,locally scaled contrast andspecific affine invariance ratherthan generic perturbation/domain robustness.','No candidate Innovation2 built ortrained. Pavia repaired unchanged.']
 (O/'REPORT.txt').write_text('\n'.join(lines)+'\n')
 with (O/'domain_gap_table.csv').open('w') as f:
  w=csv.writer(f);w.writerow(['stage','metric','Amean','AsampleSD','Cmean','CsampleSD'])
  for stage in ['V_center','V_mean','V_local','T','Z']:
   for metric in gap['A'][stage]:
    a=gap['A'][stage][metric];c=gap['C'][stage][metric];w.writerow([stage,metric,a['mean'],a['sample_std'],c['mean'],c['sample_std']])
 # Standalone scientific figure, no generated bitmap assets.
 import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
 fig,ax=plt.subplots(1,3,figsize=(13,3.8));x=np.arange(3)
 for arm,color in [('A','#64748b'),('C','#2563eb')]:
  ax[0].plot(x,stable[arm]['target']['mild']['values'],'o-',label=arm,color=color)
  ax[1].plot(x,np.maximum(stable[arm]['target']['band_affine']['values'],1e-16),'o-',label=arm,color=color)
  stages=['V_mean','T','Z'];means=[gap[arm][s]['relative_mean_distance']['mean'] for s in stages];err=[gap[arm][s]['relative_mean_distance']['sample_std'] for s in stages];ax[2].errorbar(range(3),means,yerr=err,marker='o',label=arm,color=color,capsize=3)
 for a in ax[:2]:a.set_xticks(x,[str(s) for s in seeds]);a.set_xlabel('Seed');a.set_ylabel('Target assignment JS (nats)');a.legend()
 ax[0].set_title('Mild pixelwise spectral noise');ax[1].set_title('Patch-shared band affine');ax[1].set_yscale('log')
 ax[2].set_xticks(range(3),['V spatial mean','T','Z']);ax[2].set_ylabel('Relative marginal mean distance');ax[2].set_title('Domain gap (mean ± seed SD)');ax[2].legend();fig.tight_layout();fig.savefig(O/'diagnostic_figure.png',dpi=180);fig.savefig(O/'diagnostic_figure.svg');plt.close(fig)
 print(json.dumps(decision,indent=2));print('target noise JS',stable['A']['target']['mild'],stable['C']['target']['mild'])
if __name__=='__main__':main()

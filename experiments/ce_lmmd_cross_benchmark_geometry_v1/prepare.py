import pathlib,shutil,json,numpy as np
import common as c

def main():
 c.OUT.mkdir(exist_ok=False);files={};audit=[];plan={'datasets':['houston','pavia'],'seeds':list(c.SEEDS),'objectives':['CE','CE+LMMD,nootheradaptation'],'epochs':{'houston':[20,200],'pavia':[10,100]},'selection':'10percent budget andfixedlast,not target-best','stages':list(c.STAGES),'CNN_descriptor':'F3Dcenter flattenchannels/spectral,Vcenter64,andVspatialmean64;sameactualCNNvalueusedbytokenizer','T':'actualpreTransformer4x64flatten','Z':'actualnormalizedCLS64 used by LMMD/classifier','cohorts':'allnative target pixels,eachseed originalsourceTRAINpool only;samecohorts withinCE/LMMD andphases','normalization_and_training':'ownfrozen protocols,do not compare absolute accuracies/distances acrossdatasets as protocol-controlled','source_prototypes':'mean of per-sampleL2 descriptors thennormalize,source labels only;source withinmargin leaveoneout','geometry':'source/target withinclassdispersion andclassmargin;target correct-source-centroidmargin;classcentroidcrosscosine;nearestsource-classconfusions;feature norms;globalmean distance secondary','target_GT':'only report afterALL24 featurecells frozen,no thresholds/rule fitting','existing_reuse':'HoustonCE200 F3D/V/T/Z caches;existing LMMD execution/gradientaudit and Pavia alignmentdiagnosis referenced,no newgradientqueries/probes','no_training_optimizer_backward':True,'eval_and_BN_buffers_immutable':True,'evaluation':'Houston128native tail;Pavia32repeatlasttailtrim,target,targetpair;originalviews only','no_auto_modules_or_new_experiments':True};c.dump(c.OUT/'frozen_plan.json',plan)
 for dataset in ('houston','pavia'):
  env=c.environment(dataset);env.verify_files()
  for seed in c.SEEDS:
   f0,f1=[c.folder(dataset,o,seed) for o in ('CE','LMMD')];cfg0,cfg1=[json.load(open(f/'config.json')) for f in (f0,f1)];h0,h1=[[json.loads(x) for x in (f/'history.jsonl').read_text().splitlines()] for f in (f0,f1)]
   if dataset=='houston':
    assert cfg0['initial_full_model_sha256']==cfg1['complete_initial_hash'] and cfg0['indices_sha256']==cfg1['indices_sha256']
    assert all(a['batch_stream_sha256']==b['batch_stream_sha256'] and a['original_rng_stream_sha256']==b['model_rng_sha256'] for a,b in zip(h0,h1));epochs=(20,200)
   else:
    assert cfg0['initial_complete_model_sha256']==cfg1['initial_complete_model_sha256'] and cfg0['parameters']==cfg1['parameters'];assert all(all(a[k]==b[k] for k in ('plan_sha256','all_input_views_sha256','model_rng_sha256')) for a,b in zip(h0,h1));epochs=(10,100)
   audit.append({'dataset':dataset,'seed':seed,'all_epochs_init_data_and_model_RNG_strictly_paired':True,'source_GT_budget_unchanged':True})
   for f in (f0,f1):
    for p in (f/'config.json',f/'history.jsonl',f/'frozen_manifest.json'):files[str(p)]=c.sha(p)
    for e in epochs:
     p=f/'checkpoints'/('epoch_%03d.pth'%e);assert p.exists();files[str(p)]=c.sha(p)
 if True:
  d=c.B/'tokenizer_stem_value_diagnostic_v1/data'
  for n in ('source_cube.npy','target_cube.npy','source_coords.npy','target_coords.npy','source_labels.npy','source_probe_train_mask.npy'):files[str(d/n)]=c.sha(d/n)
  p=c.B/'pavia_protocol_transfer_v1/inputs'
  for n in ('P2_source.npy','P2_target.npy','target_training_coords.npy')+tuple('source_selection_%d.npz'%s for s in c.SEEDS):files[str(p/n)]=c.sha(p/n)
 c.dump(c.OUT/'pairing_audit.json',audit);snap=c.OUT/'code_snapshot';snap.mkdir()
 for p in pathlib.Path(__file__).parent.glob('*.py'):shutil.copy2(p,snap/p.name)
 for p in snap.glob('*.py'):files[str(p)]=c.sha(p)
 c.dump(c.OUT/'input_and_code_hashes.json',files);pathlib.Path('/home/zhangzj26/dual_axis_hsi_uda/results/ce_lmmd_cross_benchmark_geometry_v1').symlink_to(c.OUT);print('STRICT_PAIRING_AUDIT_PASS',flush=True)
if __name__=='__main__':main()

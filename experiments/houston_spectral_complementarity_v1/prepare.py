import json,shutil,ast
from pathlib import Path
import numpy as np,h5py
from common import *
CODE=Path(__file__).parent
# Lock all definitions before target GT scoring. Target labels/GT arrays not opened here.
config={'checkpoint_seeds':list(SEEDS),'checkpoint_epoch':200,'main_network_forwards':0,'new_main_network_training':False,'source_train':2393,'source_val':127,'target_native_mask_candidates':52901,'target_mask_permission':'reuse historical native GT-nonzero candidate coordinates only; no expanded target-label use','raw':'48-dimensional center spectrum from existing per-pixel L2-normalized cube, no new preprocessing','structure':'index-local shape: (x_b-mean(x[b-2:b+3]))/(population_std(x[b-2:b+3])+1e-6), reflect at band-index ends;48dimensions','physical_wavelength_interpretation':False,'eps':1e-6,'index_window':5,'probe':{'StandardScaler':'fit source-train only; reuse source fit on source-val and target','C':1.0,'solver':'lbfgs','max_iter':2000,'tol':1e-6,'random_state':98000,'class_weight':None,'source_val_selection':'none; historical fixed configuration reused','hyperparameter_sweep':False},'full_fusions':'concat T256/Z64 with RAW48/STRUCT48; raw vs structure fusion dimension strictly matched','equal_total_dimension_controls':'StandardScaler on source-train backbone; source-only PCA retain D-48 components (T208,Z16); concatenate RAW48 or STRUCT48; final classifier input dimension equals original T256/Z64','source_projection':'PCA svd_solver full, whiten False, random_state98000; all fitted on source-train only','target_GT':'only report.py after every feature, fitted probe and prediction hash frozen','go_rule':'structure must improve target OA and AA vs raw, and both T/Z structure fusions must improve target OA and AA vs corresponding backbone-only and raw fusion across all3 seeds; equal-total-dimension comparison checked too. No automatic training regardless outcome.','limitations':'Deterministic spectral transform adds no new input information; tests representational accessibility/inductive bias only. Probe metrics not official network OA. Source-val127 and spatially overlapping patches; repeated checkpoint seeds share exactly same spectral data and split, standalone spectral probes not independent3trials.'}
dump(OUT/'locked_config.json',config)
meta={}
for name in ('Houston13','Houston18'):
 p=Path('/home/zhangzj26/TGRS_MLUDA-2024/datasets/Houston')/(name+'.mat')
 with h5py.File(p,'r') as f:
  meta[name]={'path':str(p),'sha256':sha(p),'keys':list(f),'root_attrs':{k:str(v) for k,v in f.attrs.items()},'datasets':{k:{'shape':list(v.shape),'attrs':{a:str(b) for a,b in v.attrs.items()}} for k,v in f.items()},'wavelength_or_band_selection_metadata_found':False}
meta['loader']='tokenizer_mechanism_v2/snapshot/BiDA/utils/dataset.py reads ori_data without band selection; upstream selection/resampling history unavailable.';meta['interpretation']='index order only; continuity and common wavelength registration cannot be independently verified.'
dump(OUT/'band_metadata_audit.json',meta)
manifest={};old=json.loads((BASE/'provenance.json').read_text());fixed=json.loads((BASE/'fixed_inputs.json').read_text())
(OUT/'data').mkdir()
for domain in ('source','target'):
 for fname in (domain+'_cube.npy',domain+'_coords.npy'):
  p=BASE/'data'/fname;assert sha(p)==old['data_sha256'][fname];manifest[str(p)]=sha(p)
 cube=np.load(BASE/'data'/(domain+'_cube.npy'),mmap_mode='r');coords=np.load(BASE/'data'/(domain+'_coords.npy'));x=np.asarray(cube[coords[:,0],coords[:,1]],np.float64);assert x.shape[1]==48 and np.isfinite(x).all();assert np.max(np.abs(np.linalg.norm(x,axis=1)-1))<1e-5
 pad=np.pad(x,((0,0),(2,2)),mode='reflect');window=np.stack([pad[:,i:i+48] for i in range(5)],axis=-1);mu=window.mean(-1);std=window.std(-1);shape=(x-mu)/(std+1e-6);assert np.isfinite(shape).all() and shape.shape==x.shape
 for kind,a in [('RAW',x),('STRUCT',shape)]:
  p=OUT/'data'/(domain+'_'+kind+'.npy');np.save(p,a);manifest[str(p)]=sha(p)
# Analytical/finite input tests without GT.
x=np.full((3,48),.2);p=np.pad(x,((0,0),(2,2)),mode='reflect');w=np.stack([p[:,i:i+48] for i in range(5)],-1);z=(x-w.mean(-1))/(w.std(-1)+1e-6);assert np.max(np.abs(z))<1e-9
for fname in ('source_labels.npy','source_probe_train_mask.npy'):
 p=BASE/'data'/fname;assert sha(p)==old['data_sha256'][fname];manifest[str(p)]=sha(p)
assert np.load(BASE/'data/source_probe_train_mask.npy').sum()==2393
for seed in SEEDS:
 cp=fixed['checkpoints'][str(seed)];assert sha(cp['path'])==cp['sha256'];manifest[cp['path']]=cp['sha256']
 for stage in ('T','Z'):
  for domain in ('source','target'):
   p=BASE/('seed_%d'%seed)/(domain+'_'+stage+'.npy');a=np.load(p,mmap_mode='r');assert len(a)==fixed['domains'][domain]['samples'] and np.isfinite(a).all();manifest[str(p)]=sha(p)
  p=BASE/('seed_%d'%seed)/(stage+'_source_probe.npz');manifest[str(p)]=sha(p)
for p in CODE.glob('*.py'):
 ast.parse(p.read_text());manifest[str(p)]=sha(p);q=OUT/'code_snapshot'/p.name;shutil.copy2(p,q);manifest[str(q)]=sha(q)
for fname in ('locked_config.json','band_metadata_audit.json'):
 p=OUT/fname;manifest[str(p)]=sha(p)
dump(OUT/'immutable_manifest.json',manifest);dump(OUT/'preflight.json',{'features_finite':True,'flat_spectrum_structure_is_zero':True,'RAW_STRUCT_dimensions_equal':48,'target_GT_not_read':True,'source_target_original_cube_and_coords_hashes_match':True,'network_checkpoints_hashes_match':True,'no_model_or_BN_forward':True});dump(OUT/'status.json',{'status':'prepared_config_locked_no_target_GT_scoring'});print('PREPARATION_COMPLETE',flush=True)

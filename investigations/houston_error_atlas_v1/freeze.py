import json,os
import numpy as np
import torch
from scipy.io import loadmat
from common import *

def link_checked(src,dest,expected):
 assert sha(src)==expected,src
 try:os.link(src.resolve(),dest)
 except OSError:dest.symlink_to(src.resolve())
 assert sha(dest)==expected

def read_raw(p):
 try:return loadmat(str(p),variable_names=['ori_data'])['ori_data']
 except NotImplementedError:
  import hdf5storage
  return hdf5storage.loadmat(str(p),variable_names=['ori_data'])['ori_data']

def descriptor_models(source,target,y,train):
 x=unit(np.asarray(source[train],float));cent=unit(np.stack([x[y[train]==c].mean(0) for c in range(7)]))
 scores=[]
 for i in range(0,len(target),512):scores.append(unit(np.asarray(target[i:i+512],float))@cent.T)
 return cent,np.concatenate(scores),unit(np.asarray(source,float))@cent.T

def main():
 torch.set_num_threads(2);OUT.mkdir(exist_ok=False);(OUT/'raw').mkdir()
 gate=json.loads((CACHE/'ALL_ROUTING_TRANSPORT_FROZEN.json').read_text());assert not gate['target_GT_opened_by_constructor']
 old=json.loads((CACHE/'provenance.json').read_text());hist=old['historical_metadata'];input_prov=json.loads((GEOM/'extraction_provenance.json').read_text())
 for name in ['source_labels.npy','source_train_mask.npy']:link_checked(CACHE/name,OUT/name,gate['files_sha256'][name])
 sy=np.load(OUT/'source_labels.npy');train=np.load(OUT/'source_train_mask.npy');assert train.sum()==2393
 for seed in SEEDS:
  folder=OUT/f'seed_{seed}';folder.mkdir();rec=old['seeds'][str(seed)];assert sha(rec['checkpoint'])==rec['checkpoint_sha256']
  names=[f'{d}_{s}.npy' for d in ['source','target'] for s in STAGES]
  names += [f'{d}_{key}.npy' for d in ['source','target'] for key in ['coords','sample_indices','logits','predictions']]
  names += ['frozen_head.npz']+[f'{r}_{suffix}' for r in ['R1','R3','R5'] for suffix in ['predictions.npy','centroid_scores.npz']]
  for name in names:rel=f'seed_{seed}/{name}';link_checked(CACHE/rel,folder/name,gate['files_sha256'][rel])
  for d in ['source','target']:assert ahash(np.load(folder/f'{d}_coords.npy'))==hist['domains'][d]['coords_sha256']
 raw_prov={};source_models={}
 for domain,scene in [('source','Houston13'),('target','Houston18')]:
  cube_path=DATA/f'{domain}_cube.npy';assert sha(cube_path)==input_prov['input_sha256'][str(cube_path)]
  cube=np.load(cube_path,mmap_mode='r');assert ahash(cube)==hist['domains'][domain]['cube_sha256']
  coords=np.load(OUT/f'seed_2100/{domain}_coords.npy');original=coords-6;assert np.all(original>=0)
  mat=Path('/home/zhangzj26/TGRS_MLUDA-2024/datasets/Houston')/f'{scene}.mat';before=sha(mat);raw=read_raw(mat)
  assert tuple(raw.shape[:2])==tuple(hist['domains'][domain]['shape']) and raw.shape[2]==48
  center=np.asarray(raw[original[:,0],original[:,1]]);assert np.isfinite(center).all()
  inp=np.asarray(cube[coords[:,0],coords[:,1]]);angle_error=float(abs(unit(center.astype(float))-unit(inp.astype(float))).max());assert angle_error<2e-6
  np.save(OUT/'raw'/f'{domain}_raw_center_spectrum.npy',center);np.save(OUT/'raw'/f'{domain}_input_center_spectrum.npy',inp)
  np.save(OUT/'raw'/f'{domain}_original_coords.npy',original)
  rel=np.lib.format.open_memmap(OUT/'raw'/f'{domain}_relation.npy',mode='w+',dtype=np.float32,shape=(len(coords),48*169))
  for i in range(0,len(coords),256):
   batch=coords[i:i+256];patch=np.stack([cube[r-6:r+7,c-6:c+7].transpose(2,0,1) for r,c in batch]);x=torch.from_numpy(patch.copy())
   with torch.inference_mode():r=(x-x[:,:,6:7,6:7])/(x.std(dim=(-2,-1),correction=0,keepdim=True)+1e-6)
   values=r.reshape(len(x),-1).numpy();assert np.isfinite(values).all() and not values[:,np.arange(48)*169+84].any();rel[i:i+len(x)]=values
   if i%(256*40)==0:print(domain,'relation extraction',i,'/',len(coords),flush=True)
  rel.flush();del rel,raw,cube
  assert sha(mat)==before;raw_prov[domain]={'unlabeled_mat':str(mat),'sha256':before,'cube_file_sha256':sha(cube_path),'raw_vs_input_unit_angle_max_error':angle_error,'relation_shape':[len(coords),8112]}
 for descriptor,name in [('S_RAW','raw_center_spectrum'),('R_FULL','relation')]:
  source=np.load(OUT/'raw'/f'source_{name}.npy',mmap_mode='r');target=np.load(OUT/'raw'/f'target_{name}.npy',mmap_mode='r')
  cent,ts,ss=descriptor_models(source,target,sy,train);np.savez_compressed(OUT/'raw'/f'{descriptor}_readout.npz',source_centroids=cent,target_scores=ts,source_scores=ss,target_predictions=ts.argmax(1),source_predictions=ss.argmax(1))
  print(descriptor,'source-only readout frozen',flush=True)
 # Consolidate all scores/predictions using original exactly replayed centroid readouts.
 sourceval=[]
 for seed in SEEDS:
  folder=OUT/f'seed_{seed}';scores={};preds={}
  for stage,router in [('V','R5'),('T','R3'),('Z','R1')]:
   rec=np.load(folder/f'{router}_centroid_scores.npz');score=rec['target_scores'];pred=np.load(folder/f'{router}_predictions.npy');assert np.array_equal(pred,score.argmax(1))
   scores[stage]=score;preds[stage]=pred
   # Source heldout is validation of frozen centroids, not fitting an extra readout.
   source=np.load(folder/f'source_{stage}.npy',mmap_mode='r');squery=unit(source[~train].astype(float));ss=squery@rec['source_centroids'].T
   sourceval.append({'seed':seed,'descriptor':stage,'heldout_n':int((~train).sum()),'heldout_OA':100*float((ss.argmax(1)==sy[~train]).mean())})
  for desc in ['S_RAW','R_FULL']:
   rec=np.load(OUT/'raw'/f'{desc}_readout.npz');scores[desc]=rec['target_scores'];preds[desc]=rec['target_predictions']
   sourceval.append({'seed':seed,'descriptor':desc,'heldout_n':int((~train).sum()),'heldout_OA':100*float((rec['source_predictions'][~train]==sy[~train]).mean())})
  native=np.load(folder/'target_logits.npy');assert np.array_equal(native.argmax(1),np.load(folder/'target_predictions.npy'))
  np.savez_compressed(folder/'ALL_FIXED_READOUTS.npz',**{f'{d}_scores':scores[d] for d in DESCRIPTORS},**{f'{d}_predictions':preds[d] for d in DESCRIPTORS})
 write('source_heldout_readout_accuracy.csv',sourceval)
 dump(OUT/'provenance.json',{'historical_metadata':hist,'seeds':old['seeds'],'raw_inputs':raw_prov,'input_links_never_mutated':True,
  'parent_preGT_manifest_sha256':sha(CACHE/'ALL_ROUTING_TRANSPORT_FROZEN.json'),'code_sha256':{str(p):sha(p) for p in HERE.iterdir() if p.is_file()}})
 files={str(p.relative_to(OUT)):sha(p) for p in OUT.rglob('*') if p.is_file()}
 dump(OUT/'ALL_ATLAS_FEATURES_READOUTS_FROZEN.json',{'target_GT_opened':False,'all3_seeds_complete':True,'source_only_centroids_no_readout_training':True,'no_tuning':True,'files_sha256':files})
 print('ALL ATLAS INPUTS/FEATURES/SCORES/PREDICTIONS HASH-FROZEN; target GT remains unopened.',flush=True)
if __name__=='__main__':main()

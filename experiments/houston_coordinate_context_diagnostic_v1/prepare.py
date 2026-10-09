import shutil,json,numpy as np
from pathlib import Path
import common as c

def main():
 c.OUT.mkdir(exist_ok=False);(c.OUT/'data').mkdir();fixed=json.loads((c.DIAG/'fixed_inputs.json').read_text());manifest={};meta={}
 plan={'seeds':list(c.SEEDS),'checkpoint':'formal C epoch200 source CE,not LMMD or adapter','mode':'all main parameters frozen;eval;BN buffers unchanged','offsets':c.OFFSETS.tolist(),'anchors':c.ANCHORS.tolist(),'offset_axis':'dx=row,dy=column','labels':'all contexts use label of original p,never window-center labels','probe':c.PROBE,'source_split':'same historical fixed source2393 train/127 val masks,identical native split across seeds','target_set':'historical diagnostic target coords;allfive valid original-scene windows only','target_GT':'not read until all3 predictions and probe configs frozen','fusion':'fixed arithmetic mean of five class probability vectors,no threshold or target tuning','Oracle':'any view correct,information only','scope':'Houston development diagnostic;probe fit is training,main network untouched;not formal model OA','budget':'five CNN+relation encodings;no Transformer or original CLS readout;union273 pixels,bounding17x17 diamond-window footprint','no_auto_followups':True,'no_Pavia':True}
 c.dump(c.OUT/'frozen_plan.json',plan)
 for d in ('source','target'):
  p=c.DIAG/'data'/('%s_coords.npy'%d);coords=np.load(p);physical=coords-6;shape=np.asarray(fixed['domains'][d]['shape']);valid=((physical>=8)&(physical<shape-8)).all(1);ids=np.flatnonzero(valid)
  for name,arr in [('coords_padded',coords[valid]),('coords_physical',physical[valid]),('historical_indices',ids),('window_centers_physical',physical[valid,None,:]+c.OFFSETS[None]),('window_anchors',np.broadcast_to(c.ANCHORS,(len(ids),5,2))),('excluded_historical_indices',np.flatnonzero(~valid))]:np.save(c.OUT/'data'/('%s_%s.npy'%(d,name)),arr)
  if d=='source':
   mask=np.load(c.DIAG/'data/source_probe_train_mask.npy')[valid];y=np.load(c.DIAG/'data/source_labels.npy')[valid];np.save(c.OUT/'data/source_train_mask.npy',mask);np.save(c.OUT/'data/source_labels.npy',y);assert set(y[mask])==set(range(7))
   meta[d]={'original':len(coords),'retained':len(ids),'excluded':int((~valid).sum()),'train_unique_pixels':int(mask.sum()),'val_unique_pixels':int((~mask).sum()),'train_counts':np.bincount(y[mask],minlength=7).tolist(),'val_counts':np.bincount(y[~mask],minlength=7).tolist()}
  else:meta[d]={'original':len(coords),'retained':len(ids),'excluded':int((~valid).sum()),'filter_uses_target_GT':False}
  for q in (c.DIAG/'data'/('%s_cube.npy'%d),p):manifest[str(q)]=c.sha(q)
 for s in c.SEEDS:
  ck=c.BASE/'houston'/('C_%d/checkpoints/epoch_200.pth'%s);assert c.sha(ck)==fixed['checkpoints'][str(s)]['sha256'];manifest[str(ck)]=c.sha(ck)
 for q in list(c.BASE.glob('*.py'))+list((c.BASE/'snapshot').rglob('*.py'))+[c.DIAG/'run_seed.py',c.DIAG/'fixed_inputs.json',c.DIAG/'data/source_labels.npy',c.DIAG/'data/source_probe_train_mask.npy']:manifest[str(q)]=c.sha(q)
 code=c.OUT/'experiment_code';code.mkdir()
 for p in Path(__file__).parent.glob('*.py'):shutil.copy2(p,code/p.name)
 for p in list(code.glob('*.py'))+list((c.OUT/'data').glob('*.npy'))+[c.OUT/'frozen_plan.json']:manifest[str(p)]=c.sha(p)
 c.dump(c.OUT/'cohort.json',meta);c.dump(c.OUT/'input_and_code_manifest.json',manifest);print('LOCKED_WITHOUT_TARGET_GT',meta,flush=True)
if __name__=='__main__':main()

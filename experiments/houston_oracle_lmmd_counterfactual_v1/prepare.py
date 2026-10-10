import pathlib,shutil,json,numpy as np,torch
import common as s

def main():
 s.OUT.mkdir(exist_ok=False);plan={'dataset':'Houston','seeds':list(s.SEEDS),'CE_checkpoint_epochs':list(s.PHASES),'queries_per_epoch':3,'updates_per_variant':1,'primary_variants':['A source CE','B CE+native predictedsoft LMMD','C CE+Oracle onehot LMMD'],'secondary_scale_control':'C_normmatch:oraclegradientdirection rescaled to predictedLMMD gradientnorm,not a new training method','lambda':'.3*(2/(1+exp(-10*checkpoint_epoch/200))-1),native frozen schedule','optimizer':'temporary fresh originalSGD .01/no momentum/no decay;equivalent persistent no-momentum SGD;discard eachvariant','gradients':'one identical trainmode jointforward at eachcheckpoint/input/rng;CE,predictedLMMD,OracleLMMD gradients same computational state','BN':'allvariants andbase eval use SAME post-single-forward buffers;no repeated train forward;restore originalcheckpoint allbuffers afterquery','input_batches':'same first3 epoch1 augmented pairedbatches as historical audit,trackednative targetindices;not actual checkpoint-phase historicalstep replay','reference_eval':'8192 native targetindices uniformlyselected excluding ALL3seed updateindices,private RNG99131;sourceval127;allvariants same eval pixels','Oracle_permission':'explicit user-authorized offline gradient counterfactual on updatebatch labels ONLY;never holdoutlabels in gradient,no formaltraining/modelselection/checkpoints','metrics':'holdoutNLL primary changes versusA,positive/negative tolerance1e-5;OA/AA/logitmargin secondary,report disagreements;gradient scale/cosines/modules/classactive sets','limits':'Oraclealso changes soft->onehot weights and nativeactiveclassset;effect not pure labelaccuracy;one-step association not longrunproof;target GT counterfactual not deployable','no_saved_temporary_models_optimizers':True,'no_sweeps_or_newnetwork':True};s.dump(s.OUT/'frozen_plan.json',plan);c=s.module();c.verify_files();files={};old=json.load(open(s.B/'lmmd_houston_pavia_execution_audit_v1/houston_gradient_queries.json'));unions=[];native_truth=None
 for seed in s.SEEDS:
  model,ls,helper,classes,audit=c.setup('houston','C',seed);source,val,target,ev=ls
  truth=np.asarray(target.dataset.labels).copy()
  if native_truth is None:native_truth=truth
  else:assert np.array_equal(native_truth,truth)
  target.dataset.label=np.zeros_like(target.dataset.label);target.dataset.labels=np.zeros_like(target.dataset.labels)
  indexloader=torch.utils.data.DataLoader(s.IndexedTarget(target.dataset),batch_size=128,shuffle=True,num_workers=target.num_workers,drop_last=False,generator=target.generator)
  batches=[]
  for (x,y),(t,ids) in zip(source,indexloader):
   assert len(x)==len(t)==128;h=__import__('hashlib').sha256(b''.join(v.numpy().tobytes() for v in (x,y,t))).hexdigest();ref=next(r for r in old if r['seed']==seed and r['epoch']==1 and r['batch']==len(batches));assert h==ref['input_sha256'],('indexedbatch drift',seed,len(batches));batches.append((x.numpy(),y.numpy(),t.numpy(),ids.numpy(),h))
   if len(batches)==3:break
  ids=np.concatenate([b[3] for b in batches]);unions.append(ids)
  path=s.OUT/('inputs_%d.npz'%seed);np.savez_compressed(path,source=np.stack([b[0] for b in batches]),source_labels=np.stack([b[1] for b in batches]),target=np.stack([b[2] for b in batches]),target_indices=np.stack([b[3] for b in batches]),oracle_target_labels=np.stack([truth[b[3]] for b in batches]));files[str(path)]=s.sha(path)
  s.dump(s.OUT/('input_audit_%d.json'%seed),{'reference_input_hashes':[b[4] for b in batches],'target_index_mapping_verified':True,'source_label_pool_unchanged':True,'target_GT_usage':'isolated Oraclegradient and offline evaluation only'})
  for epoch in s.PHASES:
   path=s.B/'tokenizer_mechanism_v2/houston'/('C_%d/checkpoints/epoch_%03d.pth'%(seed,epoch));assert path.exists();files[str(path)]=s.sha(path)
 updateids=np.unique(np.concatenate(unions));eligible=np.setdiff1d(np.arange(len(native_truth)),updateids);holdout=np.sort(np.random.default_rng(99131).choice(eligible,8192,replace=False));assert not np.intersect1d(holdout,updateids).size;np.save(s.OUT/'holdout_target_indices.npy',holdout);np.save(s.OUT/'holdout_target_labels_OFFLINE_ONLY.npy',native_truth[holdout]);np.save(s.OUT/'all_update_target_indices.npy',updateids);s.dump(s.OUT/'holdout_audit.json',{'heldout_n':len(holdout),'excluded_update_unique_target_n':len(updateids),'GT_used_to_select_holdout':False,'index_random_seed':99131,'zero_update_overlap':True,'heldout_GT_not_in_gradients':True})
 for p in (s.OUT/'holdout_target_indices.npy',s.OUT/'holdout_target_labels_OFFLINE_ONLY.npy',s.OUT/'all_update_target_indices.npy'):files[str(p)]=s.sha(p)
 snap=s.OUT/'code_snapshot';snap.mkdir()
 for p in pathlib.Path(__file__).parent.glob('*.py'):shutil.copy2(p,snap/p.name)
 for p in snap.glob('*.py'):files[str(p)]=s.sha(p)
 files[str(s.OUT/'frozen_plan.json')]=s.sha(s.OUT/'frozen_plan.json');s.dump(s.OUT/'immutable_manifest.json',files);pathlib.Path('/home/zhangzj26/dual_axis_hsi_uda/results/houston_oracle_lmmd_counterfactual_v1').symlink_to(s.OUT);print('COUNTERFACTUAL_INPUTS_MATCH_HISTORICAL_AND_HOLDOUT_LOCKED',flush=True)
if __name__=='__main__':main()

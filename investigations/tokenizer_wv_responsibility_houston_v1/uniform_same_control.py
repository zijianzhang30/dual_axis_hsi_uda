import numpy as np,torch,json
from common import OUT,BASE,CACHE,SEEDS,frozen,dump,sha
from run import Patches

@torch.inference_mode()
def run(seed):
 f=OUT/f'seed_{seed}';fixed=np.load(f/'fixed_subgroups.npz');ids=fixed['target_indices'];choose=fixed['safe_nonreflected']&(fixed['native_final_prediction']==4);rows=np.flatnonzero(choose);globalids=ids[choose];dataset=Patches();loader=torch.utils.data.DataLoader(dataset,batch_size=128,num_workers=2,shuffle=False);refs=np.load(f/'source_reference_prototypes.npz');ct=refs['source_T_centroids_unit'];cv=refs['source_V_center_centroids_unit'];gt=np.load(f/'GT_patch_labels_offline.npy')[choose];env=frozen.environment();env[4](seed);model,_=frozen.build(seed,'C','houston');cp=BASE/'houston'/f'C_{seed}/checkpoints/epoch_200.pth';cph=sha(cp);model.load_state_dict(torch.load(cp,map_location='cpu',weights_only=True)['model']);model.cuda().eval();before=frozen.state_hash(model);orig=model._forward_semantic_tokens;latest={}
 def hook(features):latest['V']=features.detach();return orig(features)
 model._forward_semantic_tokens=hook;start=0;nativecursor=0;global_mask=np.zeros(len(dataset),bool);global_mask[globalids]=True;allT=[];allV=[];max_logits=0.;mismatches=0;native=np.load(BASE/'houston'/f'C_{seed}/predictions_200.npz')['logits'];cachedC=np.load(CACHE/f'seed_{seed}/target_V.npy',mmap_mode='r');maxcenter=0.
 try:
  for x in loader:
   fulln=len(x);selected=global_mask[nativecursor:nativecursor+fulln];globalbatch=np.arange(nativecursor,nativecursor+fulln);nativecursor+=fulln
   if not selected.any():continue
   x=x.cuda();alllog=model(x,x)[1].cpu().numpy();log=alllog[selected];n=int(selected.sum());gi=globalbatch[selected];assert np.array_equal(gi,globalids[start:start+n]);max_logits=max(max_logits,float(abs(log-native[gi]).max()));mismatches+=int((log.argmax(1)!=native[gi].argmax(1)).sum());v=latest['V'][torch.from_numpy(selected).to(x.device)].flatten(2).transpose(1,2).cpu().numpy().astype(np.float64);maxcenter=max(maxcenter,float(abs(v[:,84]-cachedC[gi]).max()));known=gt[start:start+n]==5;mu=(v*known[:,:,None]).sum(1)/known.sum(1)[:,None];t=np.broadcast_to(mu[:,None,:],(n,4,64)).reshape(n,256);allT.append(t/np.maximum(np.linalg.norm(t,axis=1,keepdims=True),1e-12)@ct.T);allV.append(mu/np.maximum(np.linalg.norm(mu,axis=1,keepdims=True),1e-12)@cv.T);start+=n

 finally:model._forward_semantic_tokens=orig
 assert start==len(rows) and max_logits<2e-5 and maxcenter<1e-6 and mismatches==0,(max_logits,maxcenter,mismatches)
 assert frozen.state_hash(model)==before and sha(cp)==cph
 # Fixed learned-W knownsame counterfactual: also probe content with sourceV-center references.
 cftokens=np.load(f/'GT_only_counterfactual_tokens.npy',mmap_mode='r')[choose].astype(np.float64);learnedV=[]
 for j in range(2):
  mu=cftokens[:,j].mean(1);learnedV.append(mu/np.maximum(np.linalg.norm(mu,axis=1,keepdims=True),1e-12)@cv.T)
 uniformT=np.concatenate(allT);uniformV=np.concatenate(allV);learnedV=np.stack(learnedV,axis=1);cfT=np.load(f/'counterfactual_T_source_scores.npy')[choose];baset=np.load(f/'native_T_source_scores.npy')[choose];baseline5=baset.argmax(1)==4
 result={'seed':seed,'n_nonreflected_final_C6_to_C5':len(rows),'same_class_uniform_token_nearest_source_C6_fraction':float((uniformT.argmax(1)==5).mean()),'same_class_uniform_token_nearest_source_C5_fraction':float((uniformT.argmax(1)==4).mean()),'learned_same_class_token_nearest_source_C6_fraction':float((cfT[:,1].argmax(1)==5).mean()),'learned_same_class_token_nearest_source_C5_fraction':float((cfT[:,1].argmax(1)==4).mean()),'uniform_same_vs_learned_same_delta_C6_correspondence_pp':float(100*((uniformT.argmax(1)==5).mean()-(cfT[:,1].argmax(1)==5).mean())),'original_T_C5_to_uniform_same_T_C6_recovery_fraction':float(((uniformT.argmax(1)==5)&baseline5).sum()/baseline5.sum()),'uniform_same_mean_content_nearest_source_V_C6_fraction':float((uniformV.argmax(1)==5).mean()),'uniform_same_mean_content_nearest_source_V_C5_fraction':float((uniformV.argmax(1)==4).mean()),'learned_same_mean_content_nearest_source_V_C6_fraction':float((learnedV[:,1].argmax(1)==5).mean()),'learned_same_mean_content_nearest_source_V_C5_fraction':float((learnedV[:,1].argmax(1)==4).mean()),'native_logits_max_error':max_logits,'native_V_center_max_error':maxcenter,'native_prediction_mismatches':mismatches,'fixed_V_source_refs':True,'no_training':True,'GT_only_offline':True}
 np.savez_compressed(f/'uniform_same_control_scores.npz',subset_rows=rows,uniform_same_T_source_scores=uniformT,uniform_same_mean_V_source_scores=uniformV,learned_CF_mean_V_source_scores=learnedV);dump(f/'uniform_same_control.json',result);print(seed,json.dumps(result),flush=True);del model;torch.cuda.empty_cache()

def main():
 torch.set_num_threads(2)
 for s in SEEDS:run(s)
 dump(OUT/'uniform_same_control_summary.json',{'seeds':SEEDS,'primary_no_reflection':True,'no_training':True,'role':'within-known-same weighting sensitivity, source-V-reference sensitivity; does not alter primary projections or counterfactuals','records':[json.loads((OUT/f'seed_{s}/uniform_same_control.json').read_text()) for s in SEEDS]})
if __name__=='__main__':main()

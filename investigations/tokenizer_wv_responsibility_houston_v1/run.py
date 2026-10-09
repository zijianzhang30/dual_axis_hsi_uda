import sys,time,json
from pathlib import Path
import numpy as np,torch
from common import OUT,BASE,CACHE,SEEDS,frozen,dump,sha,hasharray
GEOM=BASE.parent/'tokenizer_class_correspondence_houston_v1';HEAD=BASE.parent/'tokenizer_head_diagnostic_houston_v1'
GROUPS=('center','known_same_noncenter','known_C5','other_known_different','unknown')
CFS=('drop_known_different_keep_unknown','only_known_same_C6')

def unit(a):return a/np.maximum(np.linalg.norm(a,axis=-1,keepdims=True),1e-12)
def divide(a,b):return np.divide(a,b,out=np.full_like(a,np.nan,dtype=np.float64),where=b>1e-30)
class Patches(torch.utils.data.Dataset):
 def __init__(self):self.cube=np.load(CACHE/'data/target_cube.npy',mmap_mode='r');self.coords=np.load(CACHE/'data/target_coords.npy')
 def __len__(self):return len(self.coords)
 def __getitem__(self,i):
  x,y=self.coords[i];return torch.from_numpy(np.array(self.cube[x-6:x+7,y-6:y+7],np.float32).transpose(2,0,1)).unsqueeze(0)
def native_init(seed):
 env=frozen.environment();env[4](seed);model,_=frozen.build(seed,'C','houston');cp=BASE/'houston'/f'C_{seed}/checkpoints/epoch_200.pth';h=sha(cp);model.load_state_dict(torch.load(cp,map_location='cpu',weights_only=True)['model']);model.cuda().eval();return model,cp,h

@torch.inference_mode()
def seed_run(seed):
 folder=OUT/f'seed_{seed}';folder.mkdir(exist_ok=False);model,cp,cp_sha=native_init(seed);before=frozen.state_hash(model);coords=np.load(CACHE/'data/target_coords.npy');labels=np.load(CACHE/'data/target_labels.npy');gt=np.load(CACHE/'data/target_gt.npy');idx=np.flatnonzero(labels==5);n=len(idx);assert n==32264
 sl=np.load(CACHE/'data/source_labels.npy');mask=np.load(HEAD/f'seed_{seed}/fixed_head_parameters.npz')['source_train_mask'];assert mask.sum()==2393
 cs=np.load(GEOM/f'seed_{seed}/C_T_source_centroid_scores.npz')['source_centroids'];sourceT=np.load(CACHE/f'seed_{seed}/source_T.npy');ct=np.stack([sourceT[mask&(sl==c)].astype(np.float64).mean(0) for c in range(7)]);assert np.max(abs(cs-ct))<1e-9;ct=unit(ct);directions=(ct[5]-ct[4]).reshape(4,64)
 sv=np.load(CACHE/f'seed_{seed}/source_V.npy');cv=unit(np.stack([sv[mask&(sl==c)].astype(np.float64).mean(0) for c in range(7)]))
 np.savez_compressed(folder/'source_reference_prototypes.npz',source_T_centroids_unit=ct,source_V_center_centroids_unit=cv,source_train_mask=mask,direction_C6_minus_C5=directions)
 gtpatch=np.stack([gt[x-6:x+7,y-6:y+7].reshape(-1) for x,y in coords[idx]]).astype(np.int8);assert (gtpatch[:,84]==5).all();np.save(folder/'GT_patch_labels_offline.npy',gtpatch)
 shape=json.loads((CACHE/'fixed_inputs.json').read_text())['domains']['target']['shape'];xy=coords[idx]-6;safe=(xy[:,0]>=6)&(xy[:,0]<shape[0]-6)&(xy[:,1]>=6)&(xy[:,1]<shape[1]-6);small=gtpatch.reshape(n,13,13)[:,5:8,5:8].reshape(n,9);boundary=((small>=0)&(small!=5)).any(1);full_same_3x3=(small==5).all(1)
 finalpred=np.load(BASE/'houston'/f'C_{seed}/predictions_200.npz')['predictions'][idx];np.savez_compressed(folder/'fixed_subgroups.npz',target_indices=idx,target_coordinates=coords[idx],safe_nonreflected=safe,boundary3x3=boundary,fully_labeled_same3x3=full_same_3x3,native_final_prediction=finalpred)
 nativeW=np.load(BASE/'houston'/f'C_{seed}/weights_200.npy',mmap_mode='r');nativeT=np.load(BASE/'houston'/f'C_{seed}/tokens_200.npy',mmap_mode='r');nativeLog=np.load(BASE/'houston'/f'C_{seed}/predictions_200.npz')['logits'];cachedCenter=np.load(CACHE/f'seed_{seed}/target_V.npy',mmap_mode='r')
 arrays={};perposition={name:np.lib.format.open_memmap(folder/(name+'.npy'),mode='w+',dtype=np.float32,shape=(n,169)) for name in ['position_signed_token_direction_contribution','position_negative_token_direction_magnitude','position_V_C6_minus_C5_cosine']};cf_tokens=np.lib.format.open_memmap(folder/'GT_only_counterfactual_tokens.npy',mode='w+',dtype=np.float32,shape=(n,2,4,64));latest={};original=model._forward_semantic_tokens
 def capture(features):
  patch=model._relation_pending[0];w=model.relation_scores(patch);t=original(features);latest.update(V=features.detach(),W=w.detach(),T=t.detach());return t
 model._forward_semantic_tokens=capture;loader=torch.utils.data.DataLoader(Patches(),batch_size=128,num_workers=2,shuffle=False);cursor=0;nativepos=0;prediction_mismatch_count=0;maxerrors={k:0. for k in ['W','T','logits','V_center','T_from_WV','projection_sum','cosine_margin_decomposition']};rfield=np.zeros((13,13),bool);rfield[2:11,2:11]=True;rfield=rfield.flatten();start=time.monotonic()
 def store(name,value,pos):
  if name not in arrays:arrays[name]=np.lib.format.open_memmap(folder/(name+'.npy'),mode='w+',dtype=value.dtype,shape=(n,*value.shape[1:]))
  arrays[name][pos:pos+len(value)]=value
 try:
  for batchid,x in enumerate(loader):
   x=x.cuda();out=model(x,x);b=len(x);lw=latest['W'].cpu().numpy();lt=latest['T'].cpu().numpy();log=out[1].cpu().numpy();prediction_mismatch_count+=int((log.argmax(1)!=nativeLog[nativepos:nativepos+b].argmax(1)).sum());maxerrors['W']=max(maxerrors['W'],float(abs(lw-nativeW[nativepos:nativepos+b]).max()));maxerrors['T']=max(maxerrors['T'],float(abs(lt-nativeT[nativepos:nativepos+b]).max()));maxerrors['logits']=max(maxerrors['logits'],float(abs(log-nativeLog[nativepos:nativepos+b]).max()));selected=labels[nativepos:nativepos+b]==5
   if not selected.any():nativepos+=b;continue
   v=latest['V'][torch.from_numpy(selected).to(x.device)].flatten(2).transpose(1,2).cpu().numpy().astype(np.float64);w=lw[selected].astype(np.float64);tnative=lt[selected].astype(np.float64);m=len(v);g=gtpatch[cursor:cursor+m];assert np.array_equal(idx[cursor:cursor+m],np.arange(nativepos,nativepos+b)[selected]);maxerrors['V_center']=max(maxerrors['V_center'],float(abs(v[:,84]-cachedCenter[nativepos:nativepos+b][selected]).max()));wbar=w.mean(1)
   reconstructed=np.einsum('nkp,npd->nkd',w,v,optimize=True);maxerrors['T_from_WV']=max(maxerrors['T_from_WV'],float(abs(reconstructed-tnative).max()));flat=reconstructed.reshape(m,256);tnorm=np.linalg.norm(flat,axis=1);scores=unit(flat)@ct.T;normmargin=scores[:,5]-scores[:,4]
   dots=np.einsum('npd,kd->nkp',v,directions,optimize=True);contrib=(dots*w).sum(1);neg=np.maximum(-contrib,0);pos=np.maximum(contrib,0);totalnegative=neg.sum(1);maxerrors['projection_sum']=max(maxerrors['projection_sum'],float(abs(contrib.sum(1)-flat@directions.reshape(-1)).max()));maxerrors['cosine_margin_decomposition']=max(maxerrors['cosine_margin_decomposition'],float(abs(contrib.sum(1)/tnorm-normmargin).max()))
   perposition['position_signed_token_direction_contribution'][cursor:cursor+m]=contrib;perposition['position_negative_token_direction_magnitude'][cursor:cursor+m]=neg
   vsim=unit(v)@cv.T;vm=vsim[:,:,5]-vsim[:,:,4];vp=vsim.argmax(2);valid=np.linalg.norm(v,axis=2)>1e-12;perposition['position_V_C6_minus_C5_cosine'][cursor:cursor+m]=vm
   center=np.zeros_like(g,dtype=bool);center[:,84]=True;same=(g==5)&~center;c5=g==4;other=(g>=0)&(g!=5)&(g!=4);unknown=g<0;gm=np.stack([center,same,c5,other,unknown],axis=1);assert (gm.sum(1)==1).all()
   mass=(gm*wbar[:,None,:]).sum(2);signed=(gm*contrib[:,None,:]).sum(2);negative=(gm*neg[:,None,:]).sum(2);positive=(gm*pos[:,None,:]).sum(2);weighted_v_neg=(gm*wbar[:,None,:]*(vm<0)[:,None,:]).sum(2);weighted_v_c5=(gm*wbar[:,None,:]*(vp==4)[:,None,:]).sum(2)
   knownnoncenter=same|c5|other;knownnum=knownnoncenter.sum(1);wmknown=(wbar*knownnoncenter).sum(1);uniform_c5=divide(c5.sum(1),knownnum);actual_c5=divide(mass[:,2],wmknown);uniform_same=divide(same.sum(1),knownnum);actual_same=divide(mass[:,1],wmknown)
   sameincl=(g==5)&valid;allvalue=divide((sameincl*(vm<0)).sum(1),sameincl.sum(1));samevaluec5=divide((sameincl*(vp==4)).sum(1),sameincl.sum(1));inside_same=sameincl&rfield[None,:];insidebad=divide((inside_same*(vm<0)).sum(1),inside_same.sum(1));insidec5=divide((inside_same*(vp==4)).sum(1),inside_same.sum(1))
   values={'native_T_source_scores':scores,'native_T_C6_minus_C5_margin':normmargin,'GT_group_weight_mass':mass,'GT_group_signed_direction_numerator':signed,'GT_group_normalized_direction_contribution':signed/tnorm[:,None],'GT_group_negative_direction_magnitude':negative,'GT_group_positive_direction_magnitude':positive,'GT_group_negative_budget_fraction':divide(negative,totalnegative[:,None]),'GT_group_V_C5_preferred_over_C6_weight_fraction':divide(weighted_v_neg,mass),'GT_group_V_nearest_C5_weight_fraction':divide(weighted_v_c5,mass),'known_neighbor_count':knownnum.astype(np.int16),'known_noncenter_weight_mass':wmknown,'uniform_known_C5_fraction':uniform_c5,'actual_known_C5_fraction':actual_c5,'C5_selection_excess_over_uniform':actual_c5-uniform_c5,'uniform_known_same_fraction':uniform_same,'actual_known_same_fraction':actual_same,'same_selection_excess_over_uniform':actual_same-uniform_same,'known_same_V_C5_preferred_over_C6_fraction':allvalue,'known_same_V_nearest_C5_fraction':samevaluec5,'RF_interior_known_same_V_C5_preferred_over_C6_fraction':insidebad,'RF_interior_known_same_V_nearest_C5_fraction':insidec5,'center_V_source_scores':vsim[:,84,:],'center_V_C6_minus_C5_margin':vm[:,84],'known_same_position_count':(g==5).sum(1).astype(np.int16),'sameclass_weight_before_CF':(wbar*(g==5)).sum(1),'negative_direction_total':totalnegative,'weight_on_CNN_RF_incomplete_rim':(wbar*(~rfield)[None,:]).sum(1)}
   allCF=[]
   for j,keep in enumerate([(g==5)|(g<0),g==5]):
    cw=w*keep[:,None,:];denom=cw.sum(2,keepdims=True);assert (denom>0).all();cw/=denom;assert abs(cw.sum(2)-1).max()<1e-12;cf=np.einsum('nkp,npd->nkd',cw,v,optimize=True);cf_tokens[cursor:cursor+m,j]=cf;allCF.append(unit(cf.reshape(m,256))@ct.T)
   values['counterfactual_T_source_scores']=np.stack(allCF,axis=1)
   for name,value in values.items():store(name,value,cursor)
   cursor+=m;nativepos+=b
   if batchid%100==0:print(seed,'native patches',nativepos,'C6 processed',cursor,'elapsed',round(time.monotonic()-start,1),flush=True)
 finally:model._forward_semantic_tokens=original
 assert nativepos==len(labels) and cursor==n
 assert max(maxerrors[k] for k in ['W','V_center'])<1e-6 and maxerrors['T']<5e-6 and maxerrors['logits']<2e-5 and prediction_mismatch_count==0,(maxerrors,prediction_mismatch_count)
 assert maxerrors['T_from_WV']<1e-5 and maxerrors['projection_sum']<1e-9 and maxerrors['cosine_margin_decomposition']<1e-12,maxerrors
 for arr in [*arrays.values(),*perposition.values(),cf_tokens]:arr.flush()
 assert frozen.state_hash(model)==before and sha(cp)==cp_sha
 dump(folder/'extraction_manifest.json',{'seed':seed,'checkpoint':str(cp),'checkpoint_sha256':cp_sha,'state_unchanged':True,'native_equivalence_max_errors':maxerrors,'native_prediction_mismatch_count':prediction_mismatch_count,'native_target_samples':len(labels),'analyzed_C6_samples':n,'GT_groups':GROUPS,'counterfactuals':CFS,'source_train_mask_sha256':hasharray(mask),'source_prototype_sha256':sha(folder/'source_reference_prototypes.npz'),'seconds':time.monotonic()-start,'Pavia_modified':False});del model;torch.cuda.empty_cache();print(seed,'DONE',json.dumps(maxerrors),flush=True)

def main():
 torch.set_num_threads(2)
 for seed in SEEDS:
  complete=OUT/f'seed_{seed}/extraction_manifest.json'
  if complete.exists():
   existing=json.loads(complete.read_text());assert sha(existing['checkpoint'])==existing['checkpoint_sha256'];assert existing['native_prediction_mismatch_count']==0;print('Retain completed seed',seed,flush=True)
  else:seed_run(seed)
 dump(OUT/'extraction_status.json',{'state':'all3seeds_extracted','no_training':True,'fixed_epoch':200,'seeds':SEEDS})
if __name__=='__main__':main()

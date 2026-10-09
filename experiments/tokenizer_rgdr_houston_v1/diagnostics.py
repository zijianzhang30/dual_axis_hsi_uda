import numpy as np,torch

def describe(details):
 capture,last=details;selected=last['selected'].cpu().numpy();candidate=last['sparse_candidate_ids'].cpu().numpy();similarity=last['similarity'].cpu().numpy();attn=last['attention'].cpu().numpy();valid=last['valid'].cpu().numpy();b,k,m=selected.shape;selected_flat=selected.reshape(-1,m);counts=np.bincount(selected.flatten(),minlength=28);classes=selected//4;classdiv=np.array([len(np.unique(r//4)) for r in selected_flat]);chosen=np.take_along_axis(similarity,selected,axis=-1);entropy=-(attn*np.log(np.maximum(attn,1e-30))).sum(-1)
 weights=capture['weights'].cpu().numpy();token=capture['raw'].cpu().numpy();refined=capture['refined'].cpu().numpy()
 return {'samples':b,'queries':b*k,'accessible_memory_entries':m,'valid_prototypes':int(valid.sum()),'selected_prototype_counts':counts.tolist(),'selected_class_counts':np.bincount(classes.flatten(),minlength=7).tolist(),'selected_slot_counts':np.bincount((selected%4).flatten(),minlength=4).tolist(),'per_query_slot_prototype_counts':[np.bincount(selected[:,slot].flatten(),minlength=28).tolist() for slot in range(4)],'top8_candidate_prototype_counts':np.bincount(candidate.flatten(),minlength=28).tolist(),'selected_similarity_quantiles':np.quantile(chosen,[0,.1,.5,.9,1]).tolist(),'all_valid_similarity_quantiles':np.quantile(similarity[:,:,valid],[0,.1,.5,.9,1]).tolist(),'attention_entropy_mean':float(entropy.mean()),'attention_entropy_normalized_mean':float((entropy/max(np.log(m),1e-30)).mean()),'selected_memory_class_diversity_mean':float(classdiv.mean()),'prototype_inclusion_fraction':(counts/(b*k)).tolist(),'most_selected_prototype_inclusion_fraction':float(counts.max()/(b*k)),'unique_selected_sets':len(set(tuple(sorted(v)) for v in selected_flat)),'relation_entropy_normalized_mean':float((-(weights*np.log(np.maximum(weights,1e-30))).sum(-1)/np.log(169)).mean()),'residual_norm_over_raw_token_norm_mean':float((np.linalg.norm(refined-token,axis=-1)/np.maximum(np.linalg.norm(token,axis=-1),1e-30)).mean())}

@torch.inference_mode()
def diagnostic(model,inputs):
 was=model.training;model.eval();mem=model.prototype_reasoning;mem.diagnostics=True
 logits,capture,last=model.target_only(inputs,return_details=True);stats=describe((capture,last));mem.diagnostics=False;model.train(was)
 return stats

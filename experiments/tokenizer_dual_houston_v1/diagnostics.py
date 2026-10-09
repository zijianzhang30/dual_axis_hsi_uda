import itertools,torch,numpy as np
PERMS=list(itertools.permutations(range(4)))
@torch.no_grad()
def describe(ws,wr,ts,tr):
 tiny=torch.finfo(ws.dtype).tiny
 cos=torch.nn.functional.normalize(ws,dim=-1)@torch.nn.functional.normalize(wr,dim=-1).transpose(1,2)
 p=ws[:,:,None,:];q=wr[:,None,:,:];mix=(p+q)/2
 js=.5*(p*(p.clamp_min(tiny).log()-mix.clamp_min(tiny).log())+q*(q.clamp_min(tiny).log()-mix.clamp_min(tiny).log())).sum(-1)
 idx=torch.arange(4,device=ws.device)
 costs=torch.stack([js[:,idx,list(perm)].mean(-1) for perm in PERMS],-1)
 best=costs.argmin(-1);mean_js=js.mean(0);global_cost=torch.stack([mean_js[idx,list(perm)].mean() for perm in PERMS]);global_best=int(global_cost.argmin())
 token_cos=torch.nn.functional.cosine_similarity(ts,tr,dim=-1).mean()
 token_pairs=torch.nn.functional.normalize(ts,dim=-1)@torch.nn.functional.normalize(tr,dim=-1).transpose(1,2)
 aligned=torch.stack([token_pairs[i,idx,list(PERMS[int(j)])].mean() for i,j in enumerate(best)]).mean()
 diag=js[:,idx,idx].mean();bestjs=costs.min(-1).values.mean()
 return {'pairwise_weight_cosine_matrix':cos.mean(0).cpu().tolist(),'pairwise_weight_JS_matrix':mean_js.cpu().tolist(),'best_global_slot_matching':list(PERMS[global_best]),'best_matching_histogram':torch.bincount(best,minlength=24).cpu().tolist(),'nonidentity_best_fraction':float((best!=0).float().mean()),'diagonal_JS':float(diag),'best_aligned_JS':float(bestjs),'diagonal_cosine':float(cos[:,idx,idx].mean()),'T_sem_rel_cosine':float(token_cos),'T_sem_rel_matched_cosine':float(aligned),'serious_slot_permutation':bool((best!=0).float().mean()>=.75 and diag-bestjs>=.01),'serious_slot_rule':'nonidentity best>=75% AND diagonal-minus-best JS>=.01 nats; diagnostic only'}

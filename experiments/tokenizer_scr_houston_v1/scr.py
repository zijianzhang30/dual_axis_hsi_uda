from types import MethodType
import torch
from torch import nn

def compatibility(values):
 assert values.ndim==4 and values.shape[1:]==(64,13,13)
 unit=torch.nn.functional.normalize(values.detach(),p=2,dim=1,eps=1e-6)
 center=unit[:,:,6,6][:,:,None,None]
 score=(unit*center).sum(1).flatten(1)
 assert score.shape==(len(values),169) and torch.isfinite(score).all()
 return score

def refined_weights(model,patch,values):
 assert patch.shape[-2:]==values.shape[-2:]==(13,13)
 u=model.relation_scores.relations(patch).permute(0,2,3,1)
 spectral=model.relation_scores.scorer(u).permute(0,3,1,2).flatten(2)
 sem=compatibility(values)
 final=spectral+model.scr_beta[None,:,None]*sem[:,None,:]
 return final.softmax(-1),sem

def semantic(model,values):
 patch=model._relation_pending.pop(0)
 weights,sem=refined_weights(model,patch,values)
 model._scr_weights=weights.detach();model._scr_sem=sem.detach()
 if model._scr_record:model._scr_last_sem=sem.detach()
 return torch.einsum('bln,bcn->blc',weights,values.flatten(2))

def install(model):
 model.scr_beta=nn.Parameter(torch.zeros(4));model._scr_record=False
 model._forward_semantic_tokens=MethodType(semantic,model)

def diagnostic(model):
 s=getattr(model,'_scr_last_sem',None)
 return {} if s is None else {'semantic_cosine_min':float(s.min()),'semantic_cosine_mean':float(s.mean()),'semantic_cosine_max':float(s.max())}

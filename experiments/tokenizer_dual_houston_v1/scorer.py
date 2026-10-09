from types import MethodType
import torch
from torch import nn
from relation_base import RelationScores,forward

def semantic_tokens(self,features):
 assert self._relation_pending
 patch=self._relation_pending.pop(0)
 assert patch.shape[-2:]==features.shape[-2:]==(13,13)
 values=features.flatten(2)
 ws=self.conv_a(features).flatten(2).softmax(-1)
 wr=self.relation_scores(patch)
 ts=torch.einsum('bln,bcn->blc',ws,values)
 tr=torch.einsum('bln,bcn->blc',wr,values)
 g=self.alpha.sigmoid();td=(1-g)*ts+g*tr
 assert ts.shape==tr.shape==td.shape==(len(patch),4,64)
 if getattr(self,'_record_dual',False):
  self._dual_diagnostic=(ws.detach(),wr.detach(),ts.detach(),tr.detach())
 return td

def install(model,arm,seed,bands):
 assert arm=='H'
 with torch.random.fork_rng(devices=[]):
  torch.set_rng_state(torch.Generator().manual_seed(seed+41000).get_state())
  model.relation_scores=RelationScores('C',bands,model.token_len)
 model.alpha=nn.Parameter(torch.zeros(()))
 model._relation_pending=[];model._relation_original_forward=model.forward.__func__
 model.forward=MethodType(forward,model);model._forward_semantic_tokens=MethodType(semantic_tokens,model)

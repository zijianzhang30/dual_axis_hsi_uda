"""Parameter-matched learned/center-conditioned query; existingfirstblock weights."""
from types import MethodType
import torch
from torch import nn

def tokens(self,features):
 assert features.shape[1:]==(64,13,13)
 self._query_centers.append(features[:,:,6,6])
 return self._query_original_tokens(self,features)

def readout(self,tokens,center):
 block=self.blocks[0];att=block.attn;b=len(tokens)
 query=self.cls_token.expand(b,-1,-1)+self.pos_embedding[:,:1]
 if self._query_arm=='Q1':query=query+center[:,None,:]
 context=tokens+self.pos_embedding[:,1:]
 query=self.dropout(query);context=self.dropout(context)
 q=att.qkv(block.norm1(query))[:,:,:64]
 allkv=att.qkv(block.norm1(context));k,v=allkv[:,:,64:128],allkv[:,:,128:]
 q=q.reshape(b,1,8,8).transpose(1,2);k=k.reshape(b,4,8,8).transpose(1,2);v=v.reshape(b,4,8,8).transpose(1,2)
 weights=((q@k.transpose(-1,-2))*att.scale).softmax(-1);att.attn=weights
 out=(att.attn_drop(weights)@v).transpose(1,2).reshape(b,1,64)
 out=att.proj_drop(att.proj(out));z=query+block.drop_path(out)
 z=z+block.drop_path(block.mlp(block.norm2(z)));z=self.norm(z)[:,0]
 self._query_last_attention=weights.detach()
 return self.nn1(z),z

def forward(self,source,target,inference_target_only=False,return_feat_prob=False):
 self._query_centers=[];self._relation_pending=[source,target]
 try:
  if self.training:
   ts,tt=self._query_joint.paired_tokens(self,source,target)
  else:
   ts=self._original_tokenize_func(self,source);tt=self._original_tokenize_func(self,target)
  assert len(self._query_centers)==2 and not self._relation_pending
  if self.training:
   os,zs=readout(self,ts,self._query_centers[0]);ot,zt=readout(self,tt,self._query_centers[1]);return os,ot,None,None
  ot,zt=readout(self,tt,self._query_centers[1])
  return (None,ot,None,zt) if return_feat_prob else (None,ot,None)
 finally:self._relation_pending=[]

def install(model,arm,joint):
 assert arm in ['Q0','Q1']
 model.blocks=nn.ModuleList([model.blocks[0]])
 model._query_arm=arm;model._query_joint=joint;model._query_centers=[]
 model._query_original_tokens=model._forward_semantic_tokens.__func__
 model._forward_semantic_tokens=MethodType(tokens,model);model.forward=MethodType(forward,model)

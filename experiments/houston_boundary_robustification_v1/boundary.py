import torch
import torch.nn.functional as F

def perturbed(z,y,head):
 # No gradient through the attack construction; z and classifier remain trainable in CE.
 with torch.no_grad():
  w=head.weight.detach();b=head.bias.detach();a=w[y]
  normal=a[:,None,:]-w[None,:,:]
  norm2=normal.square().sum(-1)
  norm2.scatter_(1,y[:,None],1.)
  assert (norm2>0).all()
  margin=(normal*z.detach()[:,None,:]).sum(-1)+b[y,None]-b[None,:]
  delta=-.5*margin[:,:,None]/norm2[:,:,None]*normal
 return z[:,None,:]+delta,margin,norm2.sqrt()

def loss(z,y,head,weights):
 zp,margin,norm=perturbed(z,y,head)
 logits=F.linear(zp,head.weight,head.bias)
 ce=F.cross_entropy(logits.reshape(-1,7),y[:,None].expand(-1,7).reshape(-1),reduction='none').reshape(-1,7)
 labels=F.one_hot(y,7).to(ce.dtype)
 numerator=labels.T@ce
 numerator=numerator+numerator.T
 counts=labels.sum(0);denominator=counts[:,None]+counts[None,:]
 # Each unordered pair averages its two true-class sides; sum weights=1.
 pairmeans=numerator/denominator.clamp_min(1)
 result=(torch.triu(weights,diagonal=1)*pairmeans).sum()
 return result,logits,margin,norm

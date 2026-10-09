"""Auxiliary relation-only branches; no model-forward or global RNG access."""
import torch
LAMBDA_REL=.1
ALPHA=(.98,1.02)
BETA=.005

def spectral(x,generator):
 # Mild version of native alpha*x + beta*pixelwise Gaussian noise.
 alpha=torch.rand((len(x),1,1,1,1),device=x.device,generator=generator)*.04+.98
 noise=torch.randn(x.shape,device=x.device,dtype=x.dtype,generator=generator)
 return alpha*x+BETA*noise

def flip(x,horizontal,vertical):
 dims=[]
 if horizontal:dims.append(-1)
 if vertical:dims.append(-2)
 return x.flip(dims) if dims else x

def kl(reference,prediction):
 # Batch/head mean of sum over spatial sites; no division by spatial length.
 p=reference.detach();tiny=torch.finfo(p.dtype).tiny
 return (p*(p.clamp_min(tiny).log()-prediction.clamp_min(tiny).log())).sum(-1).mean()

def auxiliary(scorer,target,arm,generator,weight=.1,geometry_generator=None):
 assert target.shape[-2:]==(13,13),'Houston odd13 anchor required'
 if weight==0 or arm=='C0':return target.new_zeros(()),{},None
 ref=scorer(target).detach();perturbed=scorer(spectral(target,generator));ri=kl(ref,perturbed);re=target.new_zeros(())
 if arm=='C2':
  flags=torch.randint(0,2,(2,),generator=geometry_generator if geometry_generator is not None else generator,device=target.device).tolist()
  altered=flip(target,*flags);w=scorer(altered).reshape(len(target),4,13,13)
  aligned=flip(w,*flags).flatten(2);re=kl(ref,aligned)
 return weight*(ri+re),{'ri':ri.detach(),'re':re.detach()},(ref,perturbed)

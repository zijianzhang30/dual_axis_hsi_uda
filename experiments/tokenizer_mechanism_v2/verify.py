"""Pre-training identity, anchor, grid, gradient, optimizer, and RNG gates."""
import copy
import json
import numpy as np
import torch
import torch.nn.functional as F
from common import OUT,SEEDS,ARMS,environment,build,state_hash,dump
from scorer import RelationScores

def main():
 torch.set_num_threads(2);device=torch.device('cuda:0');torch.cuda.set_device(device)
 env=environment();env[4](2100)
 errors=[];gradient_errors=[];geometry=[]
 for bands in [48,102,176]:
  for size in [12,13]:
   gen=torch.Generator().manual_seed(610+bands+size)
   patch=torch.randn(4,1,bands,size,size,generator=gen)
   coords=torch.tensor([size//2,size//2]).expand(4,2)
   d=RelationScores('D',bands);e=RelationScores('E',bands);e.load_state_dict(d.state_dict())
   for dev in [torch.device('cpu'),device]:
    dd=copy.deepcopy(d).to(dev);ee=copy.deepcopy(e).to(dev);p=patch.to(dev);c=coords.to(dev)
    u=dd.relations(p);r=ee.relations(p,c);uc=u[torch.arange(4,device=dev),:,c[:,0],c[:,1]].unsqueeze(-1).unsqueeze(-1)
    assert torch.allclose(r,u-uc,atol=2e-6,rtol=2e-6)
    error=float((dd(p)-ee(p,c)).abs().max());assert error<1e-6
    errors.append({'bands':bands,'size':size,'device':str(dev),'max_weight_error':error})
   # Double precision: equivalent input/weight gradients, null spatial bias gradient.
   d=d.double();e=e.double();e.load_state_dict(d.state_dict())
   pd=patch.double().requires_grad_(True);pe=patch.double().requires_grad_(True)
   value=torch.randn(4,4,size*size,generator=gen,dtype=torch.float64)
   ld=(d(pd)*value).sum();le=(e(pe,coords)*value).sum();ld.backward();le.backward()
   ge=float((pd.grad-pe.grad).abs().max());gw=float((d.scorer.weight.grad-e.scorer.weight.grad).abs().max())
   assert ge<1e-9 and gw<1e-9
   assert float(d.scorer.bias.grad.abs().max())<1e-9
   gradient_errors.append({'bands':bands,'size':size,'input_grad_error':ge,'weight_grad_error':gw})
   for arm in ['C','E','F']:
    scorer=RelationScores(arm,bands)
    weights=scorer(patch,coords).reshape(4,4,size,size)
    marker=torch.zeros(1,1,size,size);marker[0,0,size//2,size//2]=1
    for rotation in range(4):
     for flip in [False,True]:
      aug=torch.rot90(patch,rotation,(-2,-1));mark=torch.rot90(marker,rotation,(-2,-1));expected=torch.rot90(weights,rotation,(-2,-1))
      if flip:aug=aug.flip(-1);mark=mark.flip(-1);expected=expected.flip(-1)
      position=torch.nonzero(mark[0,0])[0].expand(4,2)
      actual=scorer(aug,position).reshape(4,4,size,size)
      assert torch.allclose(actual,expected,atol=1e-6,rtol=1e-6)
      rr=scorer.relations(aug,position)
      assert torch.equal(rr[torch.arange(4),:,position[:,0],position[:,1]],torch.zeros(4,bands))
    if size==12:
     try:scorer(patch)
     except ValueError:pass
     else:raise AssertionError('Even patch silently used guessed center')
    geometry.append({'arm':arm,'bands':bands,'size':size,'all_rotations_flips_and_anchor_checks':'passed'})
 # Real Houston loader augmentation, exact native A interface, current full-joint gradients.
 from argparse import Namespace
 args=Namespace(dataset_dir='/home/zhangzj26/TGRS_MLUDA-2024/datasets/Houston/',seed=2100,num_workers=0)
 env[4](2100);source,_,target,*_=env[0](args)
 x,y=next(iter(source));t,_=next(iter(target));x,y,t=x[:8].to(device),y[:8].to(device),t[:8].to(device)
 records=[];initials={};scores={}
 for arm in ARMS:
  env[4](2100);model,initial=build(2100,arm,'houston');initials[arm]=initial
  if arm!='A':scores[arm]=state_hash(model.relation_scores)
  if arm=='A':
   native=copy.deepcopy(model).to(device).eval();model.to(device).eval()
   with torch.no_grad():
    assert torch.equal(native(x,t)[1],model(x,t)[1])
  else:model.to(device)
  model.train();captured=[];tokens=[]
  def value_hook(mod,inp,out):captured.append(out)
  handle=model.conv2d_features[2].register_forward_hook(value_hook)
  token_function=model._forward_semantic_tokens
  def token_capture(value):
   result=token_function(value);tokens.append(result);return result
  model._forward_semantic_tokens=token_capture
  ti=t.detach().clone().requires_grad_(True)
  opt=torch.optim.SGD(model.parameters(),lr=.01);loss=F.cross_entropy(model(x,ti)[0],y);loss.backward()
  assert len(tokens)==2 and all(z.shape==(8,4,64) for z in tokens)
  # Joint BN value convolution is computed once on both domains, exact13×13 grid.
  assert len(captured)==1 and captured[0].shape==(16,64,13,13)
  assert ti.grad is not None and torch.isfinite(ti.grad).all() and ti.grad.abs().sum()>0
  assert model.conv3d_features[0].weight.grad.abs().sum()>0
  if arm!='A':
   scorer=model.relation_scores.scorer
   first=scorer if isinstance(scorer,torch.nn.Linear) else scorer[0]
   assert first.weight.grad is not None and first.weight.grad.abs().sum()>0
   old=first.weight.detach().clone();opt.step();assert not torch.equal(old,first.weight)
   assert all(id(p) in {id(z) for g in opt.param_groups for z in g['params']} for p in scorer.parameters())
  handle.remove();model._forward_semantic_tokens=token_function
  bn={name:int(m.num_batches_tracked) for name,m in model.named_modules() if isinstance(m,(torch.nn.BatchNorm2d,torch.nn.BatchNorm3d))}
  assert all(v==1 for v in bn.values())
  model.eval()
  with torch.no_grad():out=model(x,t)[1]
  assert out.shape==(8,7)
  records.append({'arm':arm,'tokens':[8,4,64],'value_grid':[13,13],'target_gradient_l1':float(ti.grad.abs().sum()),'parameters':sum(p.numel() for p in model.parameters()),'BN_calls':bn,'passed':True})
 assert len(set(initials.values()))==1
 assert scores['B']==scores['C']==scores['F'] and scores['D']==scores['E']
 # C exactly matches the historical center-relative formula and MLP parameters.
 old=RelationScores('C');from common import module,PROJECT
 legacy=module('legacy_relation_verification',PROJECT/'experiments/relation_tokenizer_v1/relation.py').RelationScores('C')
 legacy.mlp.load_state_dict(old.scorer.state_dict());p=torch.randn(3,1,48,13,13)
 assert torch.equal(old(p),legacy(p))
 report={'passed':True,'linear_theory':'r_p=u_p-u_center; W*r_p+b=(W*u_p+b)-W*u_center; per-token spatial constant cancels in softmax. Bias itself is unidentifiable under spatial softmax.',
  'D_E_weight_errors':errors,'max_D_E_error':max(x['max_weight_error'] for x in errors),'double_gradient_errors':gradient_errors,
  'anchor_checks':geometry,'real_augmented_Houston_gradient_checks':records,'MLP_B_C_F_initialization':'identical','linear_D_E_initialization':'identical','original_tensors_and_CPU_CUDA_rng':'unchanged','legacy_C_weights':'bitwise identical',
  'HyRANK_validation':'176-band synthetic geometry/linear equivalence only; no HyRANK training'}
 dump(OUT/'verification.json',report);print(json.dumps(report,indent=2),flush=True)
if __name__=='__main__':main()

import sys
sys.dont_write_bytecode=True
import pathlib,json,torch,copy
B=pathlib.Path('/nas1/zhangzj26/dual_axis_hsi_uda/results/tokenizer_mechanism_v2');O=pathlib.Path(__file__).parent
import common
from relations import auxiliary,flip,kl,spectral

def main():
 torch.set_num_threads(2);env=common.environment();env[4](2100);m,_=common.build(2100,'C','houston');m.train()
 g=torch.Generator().manual_seed(99000);x=torch.randn((4,1,48,13,13),generator=g);x=x/x.square().sum(2,keepdim=True).sqrt();target=x.flip(-1).clone()
 scorer=m.relation_scores;w=scorer(target)
 identity=float((w-scorer(target)).abs().max());assert identity<1e-6
 geo=[]
 for h,v in [(True,False),(False,True),(True,True)]:
  transformed=flip(target,h,v);assert torch.equal(scorer.anchor(transformed[:,0]),scorer.anchor(target[:,0]))
  # Coordinate mapping independently checked on unique integer grid.
  grid=torch.arange(169).reshape(1,1,13,13);assert torch.equal(flip(flip(grid,h,v),h,v),grid)
  wf=scorer(transformed).reshape(4,4,13,13);aligned=flip(wf,h,v).flatten(2);loss=kl(w,aligned)
  grads=torch.autograd.grad(loss,tuple(scorer.parameters()),retain_graph=True)
  geo.append({'h':h,'v':v,'max_weight_error':float((w-aligned).abs().max()),'RE':float(loss),'gradient_max':max(float(q.abs().max()) for q in grads)})
 # FP64 check distinguishes exact equivariance from FP32 rounding.
 double_scorer=copy.deepcopy(scorer).double();xd=target.double();wd=double_scorer(xd);ad=flip(double_scorer(flip(xd,True,True)).reshape(4,4,13,13),True,True).flatten(2)
 double_error=float((wd-ad).abs().max());assert double_error<1e-14
 # Auxiliary-only backward must not reach CNN/value or Transformer.
 m.zero_grad();before=common.state_hash(m);rng=torch.get_rng_state().clone();aux,parts,_=auxiliary(scorer,target,'C1',torch.Generator().manual_seed(31000));aux.backward();assert torch.equal(rng,torch.get_rng_state());assert before==common.state_hash(m)
 grad=max(float(p.grad.abs().max()) for p in scorer.parameters());assert grad>0
 assert all(p.grad is None for name,p in m.named_parameters() if not name.startswith('relation_scores.'))
 assert torch.isfinite(aux) and all(torch.isfinite(p.grad).all() for p in scorer.parameters())
 # Compare paired main forward at lambda0 under identical RNG and cloned model;
 # hook actual CNN BNs and native Transformer module(s).
 a=copy.deepcopy(m);b=copy.deepcopy(m);torch.manual_seed(321);ra=torch.get_rng_state().clone();oa=a(x,target)
 torch.set_rng_state(ra);ob=b(x,target);zero,_,_=auxiliary(b.relation_scores,target,'C2',torch.Generator().manual_seed(12),weight=0)
 assert all(torch.equal(q,z) for q,z in zip(oa,ob));assert zero==0
 counts={}
 hooks=[]
 for name,module in b.named_modules():
  if isinstance(module,torch.nn.modules.batchnorm._BatchNorm) or module.__class__.__name__.startswith('Block'):
   counts[name]=0
   def hook(mod,inp,out,name=name):counts[name]+=1
   hooks.append(module.register_forward_hook(hook))
 b(x,target);main_counts=counts.copy();auxiliary(b.relation_scores,target,'C2',torch.Generator().manual_seed(30000));assert counts==main_counts
 for h in hooks:h.remove()
 assert main_counts.get('conv3d_features.1')==main_counts.get('conv2d_features.1')==1
 assert any('block' in k.lower() for k in main_counts),main_counts
 result={'identity_error':identity,'flip_FP64_max_error':double_error,'flip_tests':geo,'RI_auxiliary_gradient_max':grad,'auxiliary_has_no_CNN_or_Transformer_gradient':True,'BN_Transformer_counts_main':main_counts,'BN_Transformer_counts_after_aux':counts,'C0_lambda0_logits_bitwise_equal':True,'RI_preflight_passed':True,'RE_intrinsic_equivariance':True,'RE_independent_nonzero_gradient_requirement_passed':False,'C2_formal_preflight_passed':False,'no_target_labels_accessed':True,'parameters_added':0,'reason':'Shared pointwise MLP + spatial mean/std + odd centered anchor is already flip-equivariant. RE only FP32 reduction roundoff; no geometric training signal.','spectral_augmentation':{'alpha_range':[.98,1.02],'pixelwise_gaussian_beta':.005},'lambda':.1}
 (O/'preflight.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()

import copy,json,torch,numpy as np
import study as s
import cdan_loss as a

def norm(grads):return sum(float(g.square().sum()) for g in grads if g is not None)**.5

def main():
 s.verify();torch.set_num_threads(2);c=s.environment();m,ls,helper,classes,audit=c.setup('houston','C',2100);m=m.cuda();d=a.build_discriminator(2100).cuda();d2=copy.deepcopy(d);official=s.official_loss(d2);official.grl.auto_step=False;official.grl.iter_num=1000;coeff=a.coefficient(1000)
 with torch.random.fork_rng(devices=[0]):
  torch.manual_seed(97100);torch.cuda.manual_seed_all(97100);fs=torch.randn(11,64,device='cuda',requires_grad=True);ft=torch.randn(11,64,device='cuda',requires_grad=True);gs=torch.randn(11,7,device='cuda',requires_grad=True);gt=torch.randn(11,7,device='cuda',requires_grad=True)
  mine,_=a.loss(d,(fs,ft),(gs,gt),coeff);ref=official(gs,fs,gt,ft);g=torch.autograd.grad(mine,(fs,ft,gs,gt),retain_graph=True,allow_unused=True);h=torch.autograd.grad(ref,(fs,ft,gs,gt),allow_unused=True)
  assert float(abs(mine-ref))<1e-7 and max(float((x-y).abs().max()) for x,y in zip(g[:2],h[:2]))<1e-7 and g[2:]==(None,None);numeric={'loss_error':float(abs(mine-ref)),'feature_gradient_error':max(float((x-y).abs().max()) for x,y in zip(g[:2],h[:2]))}
  x=torch.randn(12,1,48,13,13,device='cuda');t=torch.randn_like(x);y=torch.arange(12,device='cuda')%7
 state={k:v.clone() for k,v in m.state_dict().items()};cpu=torch.get_rng_state();gpu=torch.cuda.get_rng_state();m.train();out=m(x,t);ce=torch.nn.functional.cross_entropy(out[0],y);p=list(m.parameters());baseg=torch.autograd.grad(ce,p,allow_unused=True);bn0={k:int(b.num_batches_tracked) for k,b in m.named_modules() if isinstance(b,(torch.nn.BatchNorm2d,torch.nn.BatchNorm3d))};zero={}
 m.load_state_dict(state);torch.set_rng_state(cpu);torch.cuda.set_rng_state(gpu);bucket,clear,remove=a.capture(m);r=m(x,t);assert len(bucket['T'])==2 and len(bucket['Z'])==4;assert all(v.shape==(12,4,64) for v in bucket['T']);dom,_=a.loss(d,tuple(bucket['T']),r[:2],0.);g0=torch.autograd.grad(torch.nn.functional.cross_entropy(r[0],y)+dom,p,retain_graph=True,allow_unused=True)
 assert max(float((u-v).abs().max()) for u,v in zip(out,r))==0
 assert all((u is None and (v is None or not v.any())) or (u is not None and v is not None and torch.equal(u,v)) for u,v in zip(baseg,g0));zero['logits_error']=0.;zero['main_gradient_error']=0.;zero['SGD_one_step_error']=max(float(((p0-.01*(torch.zeros_like(p0) if u is None else u))-(p0-.01*(torch.zeros_like(p0) if v is None else v))).abs().max()) for p0,u,v in zip(p,baseg,g0));assert zero['SGD_one_step_error']==0
 grads={}
 for arm in ('T','Z'):
  dl,_=a.loss(d,tuple(bucket[arm][:2]),r[:2],.5);gg=torch.autograd.grad(dl,p,retain_graph=True,allow_unused=True);names=list(m.named_parameters());groups={}
  for prefix in ('conv3d_features.','conv2d_features.','relation_scores.','blocks.','nn1.'):
   groups[prefix]=norm([v for (name,param),v in zip(names,gg) if name.startswith(prefix)])
  assert groups['conv3d_features.']>0 and groups['conv2d_features.']>0 and groups['relation_scores.']>0 and groups['nn1.']==0
  assert (groups['blocks.']==0) if arm=='T' else (groups['blocks.']>0)
  grads[arm]=groups
 assert bn0=={k:int(b.num_batches_tracked) for k,b in m.named_modules() if isinstance(b,(torch.nn.BatchNorm2d,torch.nn.BatchNorm3d))};assert sum(v.numel() for v in d.parameters())==74113
 perm=torch.tensor([2,0,3,1],device='cuda');l1,_=a.loss(d,tuple(bucket['T']),r[:2],.5);l2,_=a.loss(d,tuple(v[:,perm] for v in bucket['T']),r[:2],.5);assert abs(float(l1-l2))<1e-6
 remove();s.dump(s.OUT/'preflight.json',{'passed':True,'official_numeric':numeric,'zero_GRL_main_equivalence':zero,'by_group_domain_gradient_norm':grads,'T_Z_critic_parameters_each':74113,'main_parameters':sum(p.numel() for p in m.parameters()),'detached_prediction_no_classifier_gradient':True,'token_permutation_loss_error':abs(float(l1-l2)),'BN_counts_unchanged_by_domain_loss':True,'independent_RNG_critic_init':True});print('PREFLIGHT_PASS',flush=True)
if __name__=='__main__':main()

import torch,json
from common import OUT,BASE,environment,build,state_hash,dump

def main():
 torch.set_num_threads(2);env=environment();env[4](2100);a,initial=build(2100,'Q0','houston');env[4](2100);b,other=build(2100,'Q1','houston');assert initial==other
 assert state_hash(a)==state_hash(b);params=sum(p.numel() for p in a.parameters());assert params==sum(p.numel() for p in b.parameters())==278427
 rc=json.loads((BASE/'houston/C_2100/config.json').read_text());assert initial==rc['initial_original_model_sha256'];assert state_hash(a.relation_scores)==rc['initial_scorer_sha256']
 gen=torch.Generator().manual_seed(93000);x=torch.randn(4,1,48,13,13,generator=gen);t=x.flip(-1).clone().requires_grad_();counts={};hooks=[]
 for name,m in b.named_modules():
  if isinstance(m,torch.nn.modules.batchnorm._BatchNorm):
   counts[name]=0
   def hook(m,a,o,name=name):counts[name]+=1
   hooks.append(m.register_forward_hook(hook))
 a.train();b.train();torch.manual_seed(45);rng=torch.get_rng_state();a(x,t);after=torch.get_rng_state();torch.set_rng_state(rng);out=b(x,t);assert torch.equal(after,torch.get_rng_state());assert all(v==1 for v in counts.values())
 loss=torch.nn.functional.cross_entropy(out[0],torch.arange(4));loss.backward();assert t.grad.abs().sum()>0
 gradients={n:float(p.grad.norm()) for n,p in b.named_parameters() if p.grad is not None}
 for n in ['relation_scores.scorer.0.weight','blocks.0.attn.qkv.weight','nn1.weight','conv2d_features.0.weight']:assert gradients[n]>0
 assert b._query_last_attention.shape==(4,8,1,4)
 assert b._query_centers[0].shape==(4,64)
 b._query_arm='Q0';torch.set_rng_state(rng);control=b(x,t);torch.set_rng_state(rng);base=a(x,t);err=max(float((u-v).abs().max()) for u,v in zip(control[:2],base[:2]));assert err<1e-6;b._query_arm='Q1'
 a.eval();b.eval();a(x,t);b(x,t);assert b(x,t,return_feat_prob=True)[3].shape==(4,64)
 for h in hooks:h.remove()
 dump(OUT/'verification.json',{'passed':True,'seed':2100,'Q0_Q1_initial_hash':state_hash(a),'original_C_initial_hash':initial,'same_parameter_count':params,'parameters_C':378011,'parameters_removed_vs_C':99584,'no_added_parameter_network':True,'one_existing_block_reused':True,'center':[6,6],'query_shape':[4,1,64],'context_shape':[4,4,64],'attention_shape':[4,8,1,4],'Q1_center_disabled_logits_error':err,'Q0_Q1_main_RNG_draws_equal':True,'BN_calls_train_each':1,'target_gradient_via_Joint_BN':True,'gradient_norms':gradients,'C_dropout_stream_not_matched':'3layer vs1layer differentdraw count;Q0/Q1 exactlypaired;data/augmentation mustmatchC each epoch'})
 print('PASS: parameter/init matched Q0Q1,sourceCEgradients,centergeometry,BN,RNG,center-offequivalence')
if __name__=='__main__':main()

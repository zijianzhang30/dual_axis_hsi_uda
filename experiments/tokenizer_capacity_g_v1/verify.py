"""Audit G only: no formal training, old files, or existing tasks changed."""
import json
import torch
import torch.nn.functional as F
from common import OUT,BASE,SEEDS,environment,build,state_hash,dump,module
from scorer import RelationScores

def main():
 torch.set_num_threads(2);torch.cuda.set_device(0);device=torch.device('cuda:0');env=environment();records=[]
 legacy=module('unchanged_baseline_scorer',BASE/'scorer.py')
 generator=torch.Generator().manual_seed(20261007)
 for dataset,bands in [('houston',48),('pavia',102)]:
  patch=torch.randn(4,1,bands,13,13,generator=generator)
  for seed in SEEDS:
   env[4](seed);c,ci=build(seed,'C',dataset)
   env[4](seed);g,gi=build(seed,'G',dataset)
   cs,gs=c.relation_scores,g.relation_scores
   assert ci==gi and state_hash(c)==state_hash(g)
   assert state_hash(cs)==state_hash(gs)
   assert sum(p.numel() for p in c.parameters())==sum(p.numel() for p in g.parameters())
   assert sum(p.numel() for p in cs.parameters())==sum(p.numel() for p in gs.parameters())
   assert [type(x) for x in gs.scorer]==[torch.nn.Linear,torch.nn.Identity,torch.nn.Linear]
   assert all(isinstance(x,(torch.nn.Sequential,torch.nn.Linear,torch.nn.Identity)) for x in gs.scorer.modules())
   assert torch.equal(cs.relations(patch),gs.relations(patch))
   old=legacy.RelationScores('C',bands);old.load_state_dict(cs.state_dict())
   assert torch.equal(old(patch),cs(patch))
   actual=BASE/dataset/f'C_{seed}/config.json'
   actual_match=None
   if actual.exists():
    cfg=json.loads(actual.read_text());assert cfg['initial_original_model_sha256']==gi
    assert cfg['initial_full_model_sha256']==state_hash(g)
    assert cfg['initial_scorer_sha256']==state_hash(gs)
    actual_match=True
   # G is an affine composition: verify no implicit activation with exact double reference.
   gd=RelationScores('G',bands).double();gd.load_state_dict(gs.state_dict());p64=patch.double()
   w1,b1=gd.scorer[0].weight,gd.scorer[0].bias;w2,b2=gd.scorer[2].weight,gd.scorer[2].bias
   u=gd.relations(p64).permute(0,2,3,1)
   direct=F.linear(u,w2@w1,w2@b1+b2).permute(0,3,1,2).flatten(2).softmax(-1)
   error=float((gd(p64)-direct).abs().max());assert error<1e-12
   c_count=sum(p.numel() for p in cs.parameters());g_count=sum(p.numel() for p in gs.parameters())
   before={k:v.clone() for k,v in g.state_dict().items()};g.to(device).train()
   x=patch.to(device);target=x.flip(-1).clone().requires_grad_(True);labels=torch.arange(4,device=device)%7
   tokens=[];original=g._forward_semantic_tokens
   def capture(values):
    assert values.shape==(4,64,13,13)
    z=original(values);tokens.append(z);return z
   g._forward_semantic_tokens=capture
   opt=torch.optim.SGD(g.parameters(),lr=.01);loss=F.cross_entropy(g(x,target)[0],labels);loss.backward()
   assert len(tokens)==2 and all(z.shape==(4,4,64) for z in tokens)
   assert target.grad is not None and torch.isfinite(target.grad).all() and target.grad.abs().sum()>0
   for p in g.relation_scores.parameters():assert p.grad is not None and torch.isfinite(p.grad).all()
   assert g.relation_scores.scorer[0].weight.grad.abs().sum()>0 and g.relation_scores.scorer[2].weight.grad.abs().sum()>0
   opt.step()
   assert not torch.equal(before['relation_scores.scorer.0.weight'],g.relation_scores.scorer[0].weight.cpu())
   weights=gs(x);assert weights.shape==(4,4,169) and torch.allclose(weights.sum(-1),torch.ones(4,4,device=device),atol=1e-6)
   g._forward_semantic_tokens=original
   row={'dataset':dataset,'seed':seed,'C_scorer_parameters':c_count,'G_scorer_parameters':g_count,'total_parameters':sum(p.numel() for p in g.parameters()),
    'original_initial_hash':gi,'C_G_initial_state_hash':state_hash(c),'C_G_initial_scorer_hash':state_hash(cs),
    'G_layers':['Linear(bands,32)','Identity','Linear(32,4)'],'relation_bitwise_equal':True,'legacy_C_bitwise_equal':True,
    'tokens':[4,4,64],'value_grid':[13,13],'affine_composition_error_FP64':error,'target_joint_gradient_l1':float(target.grad.abs().sum()),
    'both_scorer_weights_updated':True,'existing_C_initial_hash_match':actual_match}
   records.append(row);del c,g,cs,gs,gd,tokens,opt
 report={'passed':True,'records':records,'independent_RNG':'same seed+41000 strategy; original tensors/CPU+CUDA states unchanged in common.build',
  'data_order_proof':'Formal G compares exact indices/config and every one of200 batch/augmentation/pre-forward RNG hashes to fixed C. Existing C initial hashes checked when available.',
  'bias_gradient_note':'G affine biases cancel under spatial softmax; null/tiny bias gradients are expected. Both Linear weights and target Joint BN path have nonzero gradients.',
  'no_loss_BN_backbone_transformer_change':True,'no_existing_result_modified':True}
 dump(OUT/'verification.json',report);print(json.dumps(report,indent=2),flush=True)
if __name__=='__main__':main()

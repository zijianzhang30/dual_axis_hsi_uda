import json,torch,hashlib
from argparse import Namespace
from common import OUT,BASE,SEEDS,ARMS,environment,build,loaders,state_hash,dump
from audit import run
from scorer import RelationScores

def main():
 audit=run();env=environment();checks=[]
 torch.set_num_threads(2)
 for seed in SEEDS:
  models={};hashes={}
  for arm in ARMS:
   env[4](seed);m,initial=build(seed,arm,'pavia');models[arm]=m
   old=json.loads((BASE/'pavia'/f"{arm if arm!='G' else 'C'}_{seed}"/'config.json').read_text())
   assert initial==old['initial_original_model_sha256']
   hashes[arm]={'original':initial,'full':state_hash(m),'scorer':state_hash(m.relation_scores) if arm!='A' else None,'parameters':sum(p.numel() for p in m.parameters())}
  assert len({h['original'] for h in hashes.values()})==1
  assert len({hashes[a]['scorer'] for a in 'BCFG'})==1
  assert hashes['C']['full']==hashes['G']['full']
  assert hashes['C']['parameters']==hashes['G']['parameters']==628571
  assert [type(x).__name__ for x in models['C'].relation_scores.scorer]==['Linear','ReLU','Linear']
  assert [type(x).__name__ for x in models['G'].relation_scores.scorer]==['Linear','Identity','Linear']
  assert sum(p.numel() for p in models['C'].relation_scores.parameters())==3428
  assert models['C'].relation_scores.scorer[0].out_features==models['G'].relation_scores.scorer[0].out_features==32
  # Synthetic forward/backward on normalized patches; no target labels.
  torch.manual_seed(982);x=torch.randn(4,1,102,13,13);x=x/x.square().sum(2,keepdim=True).sqrt();t=x.flip(-1).clone().requires_grad_()
  for arm,m in models.items():
   m.train();m.zero_grad();out=m(x,t);assert out[0].shape==out[1].shape==(4,7)
   loss=torch.nn.functional.cross_entropy(out[0],torch.arange(4));loss.backward()
   assert t.grad is not None and t.grad.abs().sum()>0
   if arm!='A':
    assert m.relation_scores(x).shape==(4,4,169)
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.relation_scores.parameters())
    assert m.relation_scores.scorer[0].weight.grad.abs().sum()>0
   assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
   t.grad=None
  assert torch.equal(models['C'].relation_scores.relations(x),models['G'].relation_scores.relations(x))
  checks.append({'seed':seed,'initial_hashes':hashes,'C_G_capacity_match':True,'forward_backward_target_Joint_BN_path':True})
 # Loader order is independent of arm; compare actual augmented batches and RNG,
 # then check all seeds' indices against the frozen historical protocol.
 indices={};streams=[]
 for seed in SEEDS:
  args=Namespace(dataset='pavia',dataset_dir=__import__('audit').DATA,num_workers=0,seed=seed)
  env[4](seed);ls=loaders(args)
  hs={name:hashlib.sha256(l.dataset.indices.tobytes()).hexdigest() for name,l in zip(['source','val','target','target_eval'],ls)}
  assert hs==json.loads((BASE/'pavia'/f'A_{seed}'/'config.json').read_text())['indices_sha256'];indices[str(seed)]=hs
  del ls
 for arm in ARMS:
  env[4](2100);args=Namespace(dataset='pavia',dataset_dir=__import__('audit').DATA,num_workers=0,seed=2100);ls=loaders(args);m,_=build(2100,arm,'pavia');h=hashlib.sha256()
  for i,((x,y),(t,_)) in enumerate(zip(ls[0],ls[2])):
   for v in [x,y,t,torch.get_rng_state()]:h.update(v.numpy().tobytes())
   if i==1:break
  streams.append(h.hexdigest());del ls,m
 assert len(set(streams))==1
 dump(OUT/'verification.json',{'passed':True,'normalization_passed':True,'seeds':list(SEEDS),'models':checks,'indices_sha256':indices,'first_two_augmented_batch_and_rng_sha256':streams,'C_G_only_difference':'ReLU versus Identity; two Linear layers, hidden32, scorer3428; total628571'})
 print('PASS: real MAT + C/F normalization, initial hashes, C/G capacity, forward/backward and paired loaders',flush=True)
if __name__=='__main__':main()

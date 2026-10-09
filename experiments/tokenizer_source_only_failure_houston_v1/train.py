import argparse,json,time,hashlib
import torch,numpy as np
import torch.nn.functional as F
from common import OUT,BASE,SEEDS,ARMS,build,environment,source_loaders,sha,state_hash,dump
EPOCHS=[0,1,10,50,100,150,200]
def main():
 p=argparse.ArgumentParser();p.add_argument('--arm',choices=ARMS,required=True);p.add_argument('--seed',type=int,choices=SEEDS,required=True);args=p.parse_args();folder=OUT/'houston'/f'{args.arm}_{args.seed}';folder.mkdir(exist_ok=False);(folder/'checkpoints').mkdir()
 assert json.loads((OUT/'verification.json').read_text())['passed'];prov=json.loads((OUT/'provenance.json').read_text())
 for name,h in prov['study_sha256'].items():assert sha(OUT/name)==h
 for name,h in prov['snapshot_sha256'].items():assert sha(OUT/'snapshot'/name)==h
 torch.set_num_threads(2);env=environment();env[4](args.seed)
 import utils.dataset as dataset
 native_load=dataset.load_mat_hsi;access=[]
 def source_guard(name,*a,**kw):
  assert name=='Houston13','Target data forbidden in training';access.append(name);return native_load(name,*a,**kw)
 dataset.load_mat_hsi=source_guard
 (train,val),cube=source_loaders(args.seed);model,initial=build(args.seed,args.arm);model.cuda();opt=torch.optim.SGD(model.parameters(),lr=.01)
 config={'arm':args.arm,'seed':args.seed,'epochs':200,'patch':13,'batch':128,'steps_per_epoch':18,'optimizer':'SGD lr.01 no momentum/weight decay/scheduler','loss':'source CE only','BN':'ordinary source-only momentum.1','initial_original_model_sha256':initial,'initial_full_model_sha256':state_hash(model),'scorer_hash':state_hash(model.relation_scores) if args.arm=='C' else None,'source_normalized_cube_sha256':hashlib.sha256(cube.tobytes()).hexdigest(),'indices_sha256':{name:hashlib.sha256(loader.dataset.indices.tobytes()).hexdigest() for name,loader in [('source',train),('val',val)]},'parameters':sum(p.numel() for p in model.parameters()),'target_forward_count_training':0,'target_data_loaded_training':False,'forward':'single source CNN/tokenizer and single native self-attention path','partial_batch':'skip last partial;preserve18updates/epoch','code_hash':prov['study_sha256']}
 old=json.loads((BASE/'houston'/f'{args.arm}_{args.seed}'/'config.json').read_text());assert initial==old['initial_original_model_sha256'];assert config['indices_sha256']=={k:old['indices_sha256'][k] for k in ['source','val']};dump(folder/'config.json',config)
 reference=OUT/'houston'/f'A_{args.seed}'/'history.jsonl';ref=[json.loads(x) for x in reference.read_text().splitlines()] if args.arm=='C' else None
 checkpoints=[]
 def save(epoch):
  path=folder/'checkpoints'/f'epoch_{epoch:03d}.pth';torch.save({'model':model.state_dict(),'epoch':epoch,'config':config},path);checkpoints.append({'epoch':epoch,'path':str(path),'sha256':sha(path)})
 save(0);stats={};bn_modules={'bn3d':model.conv3d_features[1],'bn2d':model.conv2d_features[1]}
 def monitor(name):
  def hook(module,inputs):
   if not module.training:return
   x=inputs[0].detach();axes=(0,*range(2,x.ndim));mu=x.mean(axes);var=x.var(axes,correction=0);den=(module.running_var+module.eps).sqrt()
   stats[name].append({'input_var_min':float(var.min()),'input_var_mean':float(var.mean()),'input_running_mean_z_rms':float(((mu-module.running_mean)/den).square().mean().sqrt()),'input_running_var_ratio_mean':float((var/(module.running_var+module.eps)).mean())})
  return hook
 hooks=[m.register_forward_pre_hook(monitor(n)) for n,m in bn_modules.items()];start=time.monotonic()
 with (folder/'history.jsonl').open('x') as f:
  for epoch in range(1,201):
   model.train();stats={k:[] for k in bn_modules};datahash=hashlib.sha256();rnghash=hashlib.sha256();total=count=steps=correct=0
   for x,y in train:
    if len(x)!=128:continue
    datahash.update(x.numpy().tobytes());datahash.update(y.numpy().tobytes());rnghash.update(torch.get_rng_state().numpy().tobytes());rnghash.update(torch.cuda.get_rng_state().numpy().tobytes())
    x=x.cuda();y=y.cuda();opt.zero_grad(set_to_none=True);logits=model(x);loss=F.cross_entropy(logits,y);assert torch.isfinite(loss);loss.backward();opt.step();total+=float(loss)*len(y);count+=len(y);correct+=int((logits.argmax(1)==y).sum());steps+=1
   assert steps==18
   row={'epoch':epoch,'steps':steps,'source_ce':total/count,'source_train_OA':correct/count*100,'batch_stream_sha256':datahash.hexdigest(),'rng_stream_sha256':rnghash.hexdigest(),'BN':{}}
   for n,m in bn_modules.items():
    assert torch.isfinite(m.running_mean).all() and torch.isfinite(m.running_var).all();assert int(m.num_batches_tracked)==epoch*18
    row['BN'][n]={'running_mean':m.running_mean.tolist(),'running_var':m.running_var.tolist(),'running_var_min':float(m.running_var.min()),'num_batches_tracked':int(m.num_batches_tracked),'train_batch_summary':{k:float(np.mean([s[k] for s in stats[n]])) for k in stats[n][0]}}
   if ref:
    for k in ['batch_stream_sha256','rng_stream_sha256']:assert row[k]==ref[epoch-1][k],(args.seed,epoch,k)
   if epoch==1 or epoch%10==0:
    model.eval();good=n=0;vl=0
    with torch.inference_mode():
     for x,y in val:
      y=y.cuda();logits=model(x.cuda());good+=int((logits.argmax(1)==y).sum());n+=len(y);vl+=float(F.cross_entropy(logits,y))*len(y)
    row['source_val']={'oa_percent':good/n*100,'loss':vl/n};print(json.dumps({k:v for k,v in row.items() if k!='BN'}),flush=True)
   if epoch in EPOCHS:save(epoch)
   f.write(json.dumps(row)+'\n');f.flush()
 for h in hooks:h.remove()
 torch.cuda.synchronize();assert access==['Houston13'];dump(folder/'frozen_manifest.json',{'config':config,'checkpoints':checkpoints,'train_seconds':time.monotonic()-start,'source_val_at_final':row['source_val'],'data_access':access,'training_completed':True,'target_data_loaded_or_forwarded':False});print('Source-only200 frozen',flush=True)
if __name__=='__main__':main()

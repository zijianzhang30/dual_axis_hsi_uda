"""One fixed200 self-CE run; all labels for target scoring deferred."""
import argparse
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from common import OUT,BASE,PROJECT,SEEDS,ARMS,environment,loaders,build,state_hash,sha,dump

def main():
 p=argparse.ArgumentParser();p.add_argument('--dataset',choices=['houston'],required=True);p.add_argument('--arm',choices=ARMS,required=True);p.add_argument('--seed',type=int,choices=SEEDS,required=True);args=p.parse_args()
 args.num_workers=4;args.dataset_dir=(Path('/home/zhangzj26/TGRS_MLUDA-2024/datasets/Pavia') if args.dataset=='pavia' else '/home/zhangzj26/TGRS_MLUDA-2024/datasets/Houston/')
 folder=OUT/args.dataset/f'{args.arm}_{args.seed}';folder.mkdir(parents=True,exist_ok=False);(folder/'checkpoints').mkdir()
 assert json.loads((OUT/'preflight.json').read_text())['RI_preflight_passed']
 if args.arm=='C2':assert (OUT/'C2_redundant_control_authorized.json').exists(),'RE preflight has no nonzero signal; C2 needs explicit redundant-control authorization'
 provenance=json.loads((OUT/'provenance.json').read_text())
 for name,expected in provenance['study_sha256'].items():assert sha(OUT/name)==expected,(name,'changed study code')
 for name,expected in provenance['snapshot_sha256'].items():assert sha(OUT/'snapshot'/name)==expected,(name,'changed snapshot')
 torch.set_num_threads(2);device=torch.device('cuda:0');torch.cuda.set_device(device)
 env=environment();env[4](args.seed);source,val,target,target_eval=loaders(args)
 model,initial=build(args.seed,args.arm,args.dataset);model.to(device)
 helper=env[6];opt=torch.optim.SGD(model.parameters(),lr=.01)
 config={'dataset':args.dataset,'arm':args.arm,'seed':args.seed,'epochs':200,'patch':13,'batch':128,'lr':.01,'depth':3,'dropout':.1,'loss':'source-self CE + .1 target relation KL','BN':'unchanged Full Joint/momentum.19','primary':'fixed_epoch200','tokens':4,'dim':64,
  'initial_original_model_sha256':initial,'initial_full_model_sha256':state_hash(model),'initial_scorer_sha256':state_hash(model.relation_scores) if args.arm!='A' else None,
  'parameters':sum(p.numel() for p in model.parameters()),'samples':{name:len(loader.dataset) for name,loader in [('source',source),('val',val),('target',target),('target_eval',target_eval)]},
  'indices_sha256':{name:hashlib.sha256(loader.dataset.indices.tobytes()).hexdigest() for name,loader in [('source',source),('val',val),('target',target),('target_eval',target_eval)]},
  'provenance_sha256':sha(OUT/'provenance.json'),'code_sha256':provenance['study_sha256']}
 ref=BASE/args.dataset/f'C_{args.seed}'
 rc=json.loads((ref/'config.json').read_text())
 assert initial==rc['initial_original_model_sha256']
 assert config['initial_scorer_sha256']==rc['initial_scorer_sha256']
 assert config['initial_full_model_sha256']==rc['initial_full_model_sha256']
 assert config['parameters']==rc['parameters']
 assert config['indices_sha256']==rc['indices_sha256']
 assert config['samples']==rc['samples']
 reference=[json.loads(x) for x in (ref/'history.jsonl').read_text().splitlines()]
 config['exact_C0_reference']=str(ref)
 config['C_initial_state_and_indices_exact']=True
 config['lambda_rel']=.1
 config['spectral_augmentation']={'alpha_range':[.98,1.02],'pixelwise_gaussian_beta':.005}
 config['auxiliary_RNG_seed']=args.seed+91000
 config['parameters_added']=0
 dump(folder/'config.json',config);print(json.dumps(config),flush=True)
 from relations import auxiliary
 from diagnostics import describe
 auxgen=torch.Generator(device=device).manual_seed(args.seed+91000)
 geomgen=torch.Generator(device=device).manual_seed(args.seed+92000)
 checkpoints=[];start=time.monotonic();torch.cuda.synchronize()
 with (folder/'history.jsonl').open('x') as history:
  for epoch in range(1,201):
   model.train();total=count=steps=0;ri_total=re_total=0.;probe=None;batchhash=hashlib.sha256();rnghash=hashlib.sha256()
   for (x,y),(t,_) in zip(source,target):
    if len(x)!=len(t):continue
    for tensor in [x,y,t]:batchhash.update(tensor.numpy().tobytes())
    rnghash.update(torch.get_rng_state().numpy().tobytes());rnghash.update(torch.cuda.get_rng_state(device).numpy().tobytes())
    x,y,t=x.to(device),y.to(device),t.to(device);opt.zero_grad(set_to_none=True)
    ce=F.cross_entropy(model(x,t)[0],y)
    extra,parts,_=auxiliary(model.relation_scores,t,args.arm,auxgen,geometry_generator=geomgen)
    loss=ce+extra;assert torch.isfinite(loss);loss.backward();opt.step()
    ri_total+=float(parts['ri']);re_total+=float(parts['re'])
    if probe is None:probe=t.detach().clone()
    total+=ce.item()*len(y);count+=len(y);steps+=1
   expected=18 if args.dataset=='houston' else 290
   assert steps==expected,(steps,expected)
   row={'epoch':epoch,'steps':steps,'source_ce':total/count,'RI':ri_total/steps,'RE':re_total/steps,'batch_stream_sha256':batchhash.hexdigest(),'original_rng_stream_sha256':rnghash.hexdigest()}
   if reference is not None:
    for key in ['batch_stream_sha256','original_rng_stream_sha256']:assert row[key]==reference[epoch-1][key],(args.arm,args.seed,epoch,key)
   if epoch==1 or epoch%10==0:
    row['relation_diagnostics']=describe(model.relation_scores,probe,seed=77700)
    row['source_val']=helper.source_evaluate(model,val,device);print(json.dumps(row),flush=True)
   if epoch%10==0:
    path=folder/'checkpoints'/f'epoch_{epoch:03d}.pth';torch.save({'model':model.state_dict(),'epoch':epoch,'config':config},path);checkpoints.append({'epoch':epoch,'path':str(path),'sha256':sha(path)})
   history.write(json.dumps(row)+'\n');history.flush()
 torch.cuda.synchronize();seconds=time.monotonic()-start
 historical_replay=None
 # Diagnostic capture on final eval only; no extra random draws or training path.
 before=state_hash(model);model.eval();captures=[];original_tokenize=model._forward_semantic_tokens
 def capture(features):
  if args.arm=='A':weights=model.conv_a(features).flatten(2).softmax(-1)
  else:weights=model.relation_scores(model._relation_pending[0])
  tokens=original_tokenize(features);captures.append((weights.detach().cpu().numpy(),tokens.detach().cpu().numpy()))
  return tokens
 model._forward_semantic_tokens=capture
 all_logits=[];all_weights=[];all_tokens=[];equivalence_max=0.;start=time.monotonic()
 with torch.inference_mode():
  for images,_ in target_eval:
   captures=[];images=images.to(device);logits=model(images,images)[1]
   assert len(captures)==2
   all_logits.append(logits.cpu().numpy());all_weights.append(captures[1][0]);all_tokens.append(captures[1][1])
   if args.arm in 'DE':
    scorer=model.relation_scores
    equivalence_max=max(equivalence_max,float((scorer(images,arm='D')-scorer(images,arm='E')).abs().max()))
 model._forward_semantic_tokens=original_tokenize;assert before==state_hash(model)
 logits=np.concatenate(all_logits);weights=np.concatenate(all_weights);tokens=np.concatenate(all_tokens)
 np.savez_compressed(folder/'predictions_200.npz',logits=logits,predictions=logits.argmax(1))
 np.save(folder/'weights_200.npy',weights);np.save(folder/'tokens_200.npy',tokens)
 manifest={'config':config,'checkpoints':checkpoints,'prediction':{'path':str(folder/'predictions_200.npz'),'sha256':sha(folder/'predictions_200.npz')},
  'weights':{'path':str(folder/'weights_200.npy'),'sha256':sha(folder/'weights_200.npy')},'tokens':{'path':str(folder/'tokens_200.npy'),'sha256':sha(folder/'tokens_200.npy')},
  'train_seconds':seconds,'inference_seconds':time.monotonic()-start,'source_val_at_final':row['source_val'],'historical_A_replay':historical_replay,'same_parameters_D_E_max_error_real_patches':equivalence_max if args.arm in 'DE' else None,'no_target_scoring':True}
 dump(folder/'frozen_manifest.json',manifest);print('Final predictions frozen',flush=True)
if __name__=='__main__':main()

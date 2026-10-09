import argparse,json,csv,numpy as np,torch
from argparse import Namespace
from common import OUT,BASE,SEEDS,ARMS,environment,loaders,build,sha,dump,state_hash

def stats(v):
 a=np.asarray(v,float);return {'values':a.tolist(),'mean':float(a.mean()),'sample_std':float(a.std(ddof=1)) if len(a)>1 else None}
def main():
 p=argparse.ArgumentParser();p.add_argument('--epochs',type=int,choices=[100,200],required=True);epochs=p.parse_args().epochs;seeds=[2100] if epochs==100 else list(SEEDS)
 runarms=['V1'] if epochs==100 else list(ARMS);pairs=[('V1','C')] if epochs==100 else [('V0','C'),('V1','C'),('V1','V0')];root=OUT/f'evaluation_e{epochs}';root.mkdir(exist_ok=False);env=environment();torch.set_num_threads(2);device=torch.device('cuda:0');runs={};manifests={}
 # Score labels only after all matched experimental runs are complete and frozen.
 for seed in seeds:
  for arm in runarms:
   f=OUT/'houston'/f'{arm}_{seed}_e{epochs}';m=json.loads((f/'frozen_manifest.json').read_text());manifests[(arm,seed)]=m
   h=[json.loads(x) for x in (f/'history.jsonl').read_text().splitlines()];reference=[json.loads(x) for x in (BASE/'houston'/f'C_{seed}/history.jsonl').read_text().splitlines()];assert len(h)==epochs
   for row,ref in zip(h,reference):
    for key in ['batch_stream_sha256','original_rng_stream_sha256']:assert row[key]==ref[key]
   for item in [*m['checkpoints'],m['prediction'],m['weights'],m['tokens']]:assert sha(item['path'])==item['sha256']
  if epochs==200:assert manifests[('V0',seed)]['config']['initial_full_model_sha256']==manifests[('V1',seed)]['config']['initial_full_model_sha256']
 dump(root/'all_predictions_frozen.json',{'epochs':epochs,'seeds':seeds,'all_data_RNG_parameter_hash_checks_passed':True,'target_best':False})
 labels=np.load(BASE/'houston/evaluation_labels.npy')
 args=Namespace(dataset='houston',dataset_dir='/home/zhangzj26/TGRS_MLUDA-2024/datasets/Houston/',num_workers=0,seed=2100);env[4](2100);target_eval=loaders(args)[3]
 for seed in seeds:
  for arm in ['C',*runarms]:
   ref=BASE/'houston'/f'C_{seed}';f=ref if arm=='C' else OUT/'houston'/f'{arm}_{seed}_e{epochs}';cp=f/'checkpoints'/f'epoch_{epochs:03d}.pth'
   if arm=='C':
    env[4](seed);model,_=build(seed,'C','houston');model.load_state_dict(torch.load(cp,map_location='cpu',weights_only=True)['model']);model.to(device).eval();before=state_hash(model);chunks=[]
    with torch.inference_mode():
     for x,_ in target_eval: x=x.to(device);chunks.append(model(x,x)[1].cpu().numpy())
    assert before==state_hash(model);logits=np.concatenate(chunks);del model
    train_seconds=json.loads((f/'frozen_manifest.json').read_text())['train_seconds'] if epochs==200 else None
   else:logits=np.load(f/'predictions_final.npz')['logits'];train_seconds=manifests[(arm,seed)]['train_seconds']
   metrics=env[6].metrics(labels,logits.argmax(1),logits,7);metrics['collapse']=max(metrics['prediction_shares'])>=.95;metrics['zero_recall_classes']=[i+1 for i,v in enumerate(metrics['per_class_percent']) if v==0]
   history=[json.loads(x) for x in (f/'history.jsonl').read_text().splitlines()];run={'arm':arm,'seed':seed,'epoch':epochs,'metrics':metrics,'source_val':history[epochs-1]['source_val'],'parameters':378011 if arm=='C' else 382204,'train_seconds':train_seconds,'checkpoint_sha256':sha(cp)}
   if arm in ARMS:
    run['value_calibration_diagnostic']=history[-1]['branch_diagnostic'];run['gamma']=manifests[(arm,seed)]['final_gamma'];run['final_gradient_norms']={k:history[-1][k] for k in ['gamma_gradient_last_step','branch_weight_gradient_norm_last_step']}
    w=np.load(f/'weights_final.npy',mmap_mode='r');run['assignment_entropy_normalized']=float(-(w*np.log(np.maximum(w,1e-30))).sum(-1).mean()/np.log(169))
   runs[f'{arm}_{seed}']=run
 keys=['oa_percent','aa_percent','kappa'];comparisons={f'{a}-{b}':{k:stats([runs[f'{a}_{s}']['metrics'][k]-runs[f'{b}_{s}']['metrics'][k] for s in seeds]) for k in keys} for a,b in pairs};aggregate={arm:{k:stats([runs[f'{arm}_{s}']['metrics'][k] for s in seeds]) for k in keys} for arm in ['C',*runarms]}
 if epochs==100:
  decision={'continue_stage2':bool(comparisons['V1-C']['oa_percent']['mean']>0 and comparisons['V1-C']['aa_percent']['mean']>0),'rule':'fixed100 V1-C OA>0 AND AA>0, predefined before scoring; capacity control deferred to formal200 after Go','no_tuning':True}
 else:decision={'no_automatic_Pavia_or_new_variants':True,'V1_exceeds_C_OA_AA_seed_count':sum(runs[f'V1_{s}']['metrics']['oa_percent']>runs[f'C_{s}']['metrics']['oa_percent'] and runs[f'V1_{s}']['metrics']['aa_percent']>runs[f'C_{s}']['metrics']['aa_percent'] for s in seeds),'V1_exceeds_V0_OA_AA_seed_count':sum(runs[f'V1_{s}']['metrics']['oa_percent']>runs[f'V0_{s}']['metrics']['oa_percent'] and runs[f'V1_{s}']['metrics']['aa_percent']>runs[f'V0_{s}']['metrics']['aa_percent'] for s in seeds)}
 result={'fixed_epoch':epochs,'aggregate':aggregate,'paired_deltas':comparisons,'runs':runs,'decision':decision,'parameters_added':4193,'MLP_MACs_per_patch':169*(64*32+32*64),'Norm_residual_operations_not_counted':True,'insertion':'actual CNN value before unchanged relation tokenizer;gamma scalar0;MLP64->32ReLU->64','timing_limit':'shared GPU load, not controlled overhead benchmark; baseline100 runtime unavailable'};dump(root/'summary.json',result)
 with (root/'per_class_paired.csv').open('x',newline='') as f:
  w=csv.writer(f);w.writerow(['seed','comparison','class','target_count','target_share','baseline_recall_percent','method_recall_percent','delta_recall_pp','delta_OA_contribution_pp'])
  for seed in seeds:
   for a,b in pairs:
    ma=runs[f'{a}_{seed}']['metrics'];mb=runs[f'{b}_{seed}']['metrics']
    for i,(r,t,row) in enumerate(zip(mb['per_class_percent'],ma['per_class_percent'],mb['confusion_matrix'])):
     n=sum(row);w.writerow([seed,f'{a}-{b}',i+1,n,n/mb['n'],r,t,t-r,n/mb['n']*(t-r)])
 (root/'RESULTS.txt').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(decision),flush=True)
if __name__=='__main__':main()

import argparse,json,numpy as np,torch
from argparse import Namespace
from common import OUT,BASE,SEEDS,environment,loaders,build,sha,dump,state_hash

def stats(v):
 a=np.asarray(v,float);return {'values':a.tolist(),'mean':float(a.mean()),'sample_std':float(a.std(ddof=1)) if len(a)>1 else None}
def main():
 p=argparse.ArgumentParser();p.add_argument('--epochs',type=int,choices=[100,200],required=True);epochs=p.parse_args().epochs;seeds=[2100] if epochs==100 else list(SEEDS)
 root=OUT/f'evaluation_e{epochs}';root.mkdir(exist_ok=False);env=environment();torch.set_num_threads(2);device=torch.device('cuda:0');runs={}
 # Freeze/audit every trained run before loading evaluation labels.
 manifests={}
 for seed in seeds:
  f=OUT/'houston'/f'S1_{seed}_e{epochs}';m=json.loads((f/'frozen_manifest.json').read_text());manifests[seed]=m
  h=[json.loads(x) for x in (f/'history.jsonl').read_text().splitlines()];reference=[json.loads(x) for x in (BASE/'houston'/f'C_{seed}/history.jsonl').read_text().splitlines()];assert len(h)==epochs
  for row,ref in zip(h,reference):
   for key in ['batch_stream_sha256','original_rng_stream_sha256']:assert row[key]==ref[key]
  for item in [*m['checkpoints'],m['prediction'],m['weights'],m['tokens']]:assert sha(item['path'])==item['sha256']
 dump(root/'all_predictions_frozen.json',{'epochs':epochs,'seeds':seeds,'all_pairing_checks_passed':True,'target_best':False})
 labels=np.load(BASE/'houston/evaluation_labels.npy')
 args=Namespace(dataset='houston',dataset_dir='/home/zhangzj26/TGRS_MLUDA-2024/datasets/Houston/',num_workers=0,seed=2100);env[4](2100);target_eval=loaders(args)[3]
 for seed in seeds:
  for arm in ['C','S1']:
   ref=BASE/'houston'/f'C_{seed}'
   f=ref if arm=='C' else OUT/'houston'/f'S1_{seed}_e{epochs}'
   cp=f/'checkpoints'/f'epoch_{epochs:03d}.pth'
   if arm=='C':
    env[4](seed);model,_=build(seed,'C','houston');model.load_state_dict(torch.load(cp,map_location='cpu',weights_only=True)['model']);model.to(device).eval();before=state_hash(model);chunks=[]
    with torch.inference_mode():
     for x,_ in target_eval: x=x.to(device);chunks.append(model(x,x)[1].cpu().numpy())
    assert before==state_hash(model);logits=np.concatenate(chunks);del model
    train_seconds=json.loads((f/'frozen_manifest.json').read_text())['train_seconds'] if epochs==200 else None;gamma=0.
   else:
    logits=np.load(f/'predictions_final.npz')['logits'];train_seconds=manifests[seed]['train_seconds'];gamma=manifests[seed]['final_gamma']
   metrics=env[6].metrics(labels,logits.argmax(1),logits,7);metrics['collapse']=max(metrics['prediction_shares'])>=.95;metrics['zero_recall_classes']=[i+1 for i,v in enumerate(metrics['per_class_percent']) if v==0]
   history=[json.loads(x) for x in (f/'history.jsonl').read_text().splitlines()]
   run={'arm':arm,'seed':seed,'epoch':epochs,'metrics':metrics,'gamma':gamma,'source_val':history[epochs-1]['source_val'],'parameters':378011 if arm=='C' else 378036,'train_seconds':train_seconds,'checkpoint_sha256':sha(cp)}
   if arm=='S1':run['branch_diagnostic']=history[-1]['branch_diagnostic'];run['gamma_trajectory']=[{'epoch':r['epoch'],'gamma':r['gamma'],'weight_gradient_norm':r['branch_weight_gradient_norm_last_step']} for r in history if r['epoch']==1 or r['epoch']%10==0]
   runs[f'{arm}_{seed}']=run
 keys=['oa_percent','aa_percent','kappa'];paired={k:stats([runs[f'S1_{s}']['metrics'][k]-runs[f'C_{s}']['metrics'][k] for s in seeds]) for k in keys};aggregate={arm:{k:stats([runs[f'{arm}_{s}']['metrics'][k] for s in seeds]) for k in keys} for arm in ['C','S1']}
 decision={'continue_stage2':bool(paired['oa_percent']['mean']>0 and paired['aa_percent']['mean']>0),'rule':'fixed100 paired OA>0 AND AA>0, predefined before evaluation, no tuning'} if epochs==100 else {'no_automatic_Pavia_or_new_variants':True,'OA_AA_both_positive_seed_count':sum(runs[f'S1_{s}']['metrics']['oa_percent']>runs[f'C_{s}']['metrics']['oa_percent'] and runs[f'S1_{s}']['metrics']['aa_percent']>runs[f'C_{s}']['metrics']['aa_percent'] for s in seeds)}
 result={'fixed_epoch':epochs,'aggregate':aggregate,'paired_S1_minus_C':paired,'runs':runs,'decision':decision,'parameters_added':25,'extra_Conv3D_MACs_per_patch':8*48*13*13*3,'extra_relation_and_residual_ops_not_counted':True,'insertion':'original Conv3D output before unchanged original BN3D/ReLU','timing_limit':'shared GPU load; C100 baseline full200 runtime not used as matched100 runtime'};dump(root/'summary.json',result)
 (root/'RESULTS.txt').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(decision),flush=True)
if __name__=='__main__':main()

"""One paired entry point for fresh A/B/C/D fixed-endpoint runs."""
import argparse
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from shift import scene_shift,scene_stats,ALPHA,CE_WEIGHT,EPS

ROOT=Path(__file__).resolve().parents[2]
BIDA=Path('/home/zhangzj26/IEEE_TCSVT_BiDA')
sys.path.insert(0,str(BIDA));sys.path.insert(0,str(BIDA/'experiments/bida_memory_v1'))
from experiments.bida_memory_v1.run import make_loaders
from models.get_model import get_model
from utils.scheduler import load_scheduler
from utils.utils_HSI import seed_worker

def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result);return result
helper=module('shift_metrics',ROOT/'experiments/bida_self_dropout/train.py')
joint=module('shift_joint',ROOT/'experiments/bn_stabilization_v1/normalization.py')

def main():
    p=argparse.ArgumentParser();p.add_argument('--arm',choices=list('ABCD'),required=True)
    p.add_argument('--seed',type=int,choices=[2100,2101,2102],required=True)
    p.add_argument('--device',required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--num-workers',type=int,default=4)
    p.add_argument('--dataset-dir',default='/home/zhangzj26/TGRS_MLUDA-2024/datasets/Houston/')
    a=p.parse_args();torch.set_num_threads(2)
    if a.out.exists() and any(a.out.iterdir()):p.error('Output must be empty')
    a.model,a.lr,a.num_tokens,a.dim,a.depth='BiDA',.01,4,64,3
    a.loss_type,a.labelsmooth,a.epochs='softmax','off',200
    device=torch.device('cuda:'+a.device);enabled=a.arm in 'BD';joint_on=a.arm in 'CD'
    seed_worker(a.seed)
    train,val,target,target_eval,_,classes=make_loaders(a)
    model=get_model('BiDA','Houston13',13,a)
    initial=helper.state_hash(model)
    original=ROOT/'results/bida_self'/('diagnostic_2100' if a.seed==2100 else f'multiseed_{a.seed}')
    assert initial==json.loads((original/'config.json').read_text())['initial_model_sha256']
    ema=get_model('BiDA','Houston13',13,a,ema=True);del ema
    if joint_on:joint.install(model,'fixed_mixture')
    assert helper.state_hash(model)==initial
    model.to(device);optimizer,scheduler=load_scheduler('BiDA',model,a)
    assert scheduler is None
    prepare=time.monotonic();stats=scene_stats(train.dataset,target.dataset,device)
    preparation_seconds=time.monotonic()-prepare
    noise_rng=torch.Generator(device=device).manual_seed(a.seed+99173)
    auxiliary_rng=torch.Generator(device=device).manual_seed(a.seed+199173)
    auxiliary_state=auxiliary_rng.get_state()
    assert sum(p.numel() for p in model.parameters())==376567
    historical=Path('/home/zhangzj26/spectralflow_uda/experiments/round9/train_mluda_shift.py')
    cfg={'arm':a.arm,'seed':a.seed,'epochs':200,'initial_model_sha256':initial,
         'trainable_parameters':376567,'depth':3,'dropout':.1,'lr':.01,
         'bn':'Full Joint' if joint_on else 'Original sequential',
         'bn_momentum':.19 if joint_on else .1,'sceneshift':enabled,
         'shift_alpha':ALPHA,'shift_ce_weight':CE_WEIGHT,'shift_epsilon':EPS,
         'amplitude_sd':.04,'smooth_sd':.015,'smooth_kernel':5,'clamp':True,
         'loss':'CE_s + 0.5 CE_shift' if enabled else 'CE_s',
         'auxiliary_bn':'train batch stats; normal running updates' if enabled else 'none',
         'primary':'fixed_epoch_200','device':str(device),
         'source_n':len(train.dataset),'source_val_n':len(val.dataset),'target_n':len(target.dataset),
         'source_indices_sha256':hashlib.sha256(train.dataset.indices.tobytes()).hexdigest(),
         'target_indices_sha256':hashlib.sha256(target.dataset.indices.tobytes()).hexdigest(),
         'band_stats':[s.detach().cpu().reshape(-1).tolist() for s in stats],
         'protocol_sha256':helper.file_hash(Path(__file__).with_name('PROTOCOL.md')),
         'train_script_sha256':helper.file_hash(Path(__file__)),
         'shift_script_sha256':helper.file_hash(Path(__file__).with_name('shift.py')),
         'normalization_script_sha256':helper.file_hash(ROOT/'experiments/bn_stabilization_v1/normalization.py'),
         'historical_shift_script_sha256':helper.file_hash(historical)}
    a.out.mkdir(parents=True,exist_ok=True);(a.out/'checkpoints').mkdir()
    (a.out/'config.json').write_text(json.dumps(cfg,indent=2)+'\n')
    checkpoints=[];aux_events=[];clipped_zero=clipped_one=shift_elements=0
    torch.cuda.synchronize(device);start=time.monotonic()
    with (a.out/'history.jsonl').open('w') as history:
        for epoch in range(1,201):
            model.train();sum_ce=sum_shift=count=steps=0
            batch_hash=hashlib.sha256();rng_hash=hashlib.sha256()
            for (x,labels),(t,_) in zip(train,target):
                if len(x)!=len(t):continue
                for tensor in (x,labels,t):batch_hash.update(tensor.numpy().tobytes())
                rng_hash.update(torch.get_rng_state().numpy().tobytes())
                rng_hash.update(torch.cuda.get_rng_state(device).numpy().tobytes())
                x,labels,t=x.to(device),labels.to(device),t.to(device)
                optimizer.zero_grad(set_to_none=True)
                ce=F.cross_entropy(model(x,t)[0],labels);loss=ce
                if enabled:
                    begin=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
                    begin.record()
                    shifted=scene_shift(x,*stats,noise_rng)
                    with torch.random.fork_rng(devices=[device.index]):
                        torch.cuda.set_rng_state(auxiliary_state,device)
                        extra=F.cross_entropy(model(shifted,t)[0],labels)
                        auxiliary_state=torch.cuda.get_rng_state(device)
                    loss=ce+CE_WEIGHT*extra;end.record();aux_events.append((begin,end))
                    sum_shift+=extra.item()*len(labels)
                    clipped_zero+=int((shifted==0).sum());clipped_one+=int((shifted==1).sum());shift_elements+=shifted.numel()
                assert torch.isfinite(loss)
                loss.backward();optimizer.step()
                sum_ce+=ce.item()*len(labels);count+=len(labels);steps+=1
            assert steps==18
            row={'epoch':epoch,'steps':steps,'source_ce':sum_ce/count,
                 'shifted_ce':sum_shift/count if enabled else None,
                 'batch_stream_sha256':batch_hash.hexdigest(),'original_rng_stream_sha256':rng_hash.hexdigest()}
            if epoch==1 or epoch%10==0:
                row['source_val']=helper.source_evaluate(model,val,device);print(json.dumps(row),flush=True)
            if epoch%10==0:
                path=a.out/'checkpoints'/f'epoch_{epoch:03d}.pth'
                torch.save({'model':model.state_dict(),'epoch':epoch,'config':cfg},path)
                checkpoints.append({'epoch':epoch,'path':str(path),'sha256':helper.file_hash(path)})
            history.write(json.dumps(row)+'\n');history.flush()
    torch.cuda.synchronize(device);training_seconds=time.monotonic()-start
    aux_ms=sum(s.elapsed_time(e) for s,e in aux_events)
    infer=time.monotonic();logits,pred=helper.predict(model,target_eval,device)
    pred_path=a.out/'predictions_200.npz';np.savez_compressed(pred_path,logits=logits,predictions=pred)
    inference_seconds=time.monotonic()-infer
    manifest={'config':cfg,'checkpoints':checkpoints,'predictions':[{'epoch':200,'path':str(pred_path),'sha256':helper.file_hash(pred_path)}]}
    manifest_path=a.out/'frozen_manifest.json';manifest_path.write_text(json.dumps(manifest,indent=2)+'\n')
    manifest_hash=helper.file_hash(manifest_path)
    print(json.dumps({'frozen_manifest_sha256':manifest_hash}),flush=True)
    labels=np.concatenate([y.numpy() for _,y in target_eval])
    metrics=helper.metrics(labels,pred,logits,classes);metrics['class6_collapse']=metrics['prediction_shares'][5]>=.95
    reference=None
    if not enabled:
        ref_path=(ROOT/f'results/bn_stabilization_v1/fixed_mixture_{a.seed}/checkpoints/epoch_200.pth') if joint_on else original/'final.pth'
        ref=torch.load(ref_path,map_location='cpu',weights_only=True)['model']
        fresh={k:v.detach().cpu() for k,v in model.state_dict().items()}
        reference={'path':str(ref_path),'state_exact':all(torch.equal(v,ref[k]) for k,v in fresh.items()),
                   'max_abs_difference':max(float((v-ref[k]).abs().max()) for k,v in fresh.items())}
    result={'target_at_final':metrics,'source_val_at_final':row['source_val'],
            'frozen_manifest_sha256':manifest_hash,'parameters':376567,
            'train_seconds':training_seconds,'preparation_seconds':preparation_seconds,
            'inference_seconds':inference_seconds,'auxiliary_forward_loss_cuda_ms':aux_ms,
            'peak_gpu_allocated_bytes':torch.cuda.max_memory_allocated(device),
            'clamp_zero_share':clipped_zero/max(shift_elements,1),'clamp_one_share':clipped_one/max(shift_elements,1),
            'historical_replay':reference}
    (a.out/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    print('RESULT '+json.dumps(result),flush=True)

if __name__=='__main__':main()

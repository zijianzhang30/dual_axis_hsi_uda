"""Three prescribed complete-BiDA runs; no additional validation at epoch one."""
import argparse
import json
import platform
import time
import numpy as np
import torch
from control import c,OUT,ROOT,SEEDS,Trace,Loader,loaders,native,dump


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--seed',type=int,choices=SEEDS,required=True)
    parser.add_argument('--device',type=int,required=True);a=parser.parse_args()
    out=OUT/f'seed_{a.seed}';assert not out.exists(),'Refuse overwrite'
    old=json.loads((c.OUT/f'hyrank_A_{a.seed}'/'config.json').read_text())
    for path,h in old['code_sha256'].items():assert c.helper.file_hash(path)==h
    args=c.options(a.seed);args.device=str(a.device)
    assert {k:v for k,v in vars(args).items() if k!='device'}=={k:v for k,v in old['settings'].items() if k!='device'}
    torch.set_num_threads(2);device=torch.device(f'cuda:{a.device}');torch.cuda.set_device(device)
    c.seed_worker(a.seed) # One original startup seeding call; no later reseeding.
    values,data=loaders(args);source,val,target,target_eval=values
    assert source.generator is val.generator is target.generator is target_eval.generator
    assert data==old['data_protocol']
    model,ema,initial=c.build(args,'A')
    assert initial==old['initial_common_model_sha256']
    assert c.helper.state_hash(model)==old['initial_complete_state_sha256']
    model.to(device);ema.to(device)
    optimizer,scheduler=c.load_scheduler('BiDA',model,args);assert scheduler is None
    criterion,_=c.make_loss(args,num_classes=12)
    out.mkdir();(out/'checkpoints').mkdir();(out/'predictions').mkdir()
    config={'arm':'A_no_extra_val','interpretation':'remove extra epoch1 source validation workflow control, not pure RNG',
            'primary':'fixed epoch200','oracle':'diagnostic OA-best over epochs10:10:200; associated AA/Kappa',
            'settings':vars(args),'data_protocol':data,'old_A_config_sha256':c.helper.file_hash(c.OUT/f'hyrank_A_{a.seed}'/'config.json'),
            'initial_common_model_sha256':initial,'initial_complete_state_sha256':c.helper.state_hash(model),
            'initial_EMA_state_sha256':c.helper.state_hash(ema),'parameters':sum(p.numel() for p in model.parameters()),
            'environment':{'python':platform.python_version(),'torch':torch.__version__,'numpy':np.__version__,
                           'cuda':torch.version.cuda,'cudnn':torch.backends.cudnn.version()},
            'preserved_old_training_code_hashes':old['code_sha256'],
            'new_code_sha256':{str(p):c.helper.file_hash(p) for p in (ROOT/'experiments/hyrank_A_no_extra_val_v1').glob('*.py')},
            'no_added_train_eval_calls':True,'no_independent_generators':True,'no_rng_restore_or_consumption_simulation':True,
            'target_validation':'native validation call in original post-scheduler if e%log_interval==0 position',
            'old_vs_new_every10':'Old A replaced native target validation with source validation; new run uses requested native target validation, one shared-generator eval iterator per scheduled epoch; actual RNG states logged'}
    dump(out/'config.json',config)
    trace=Trace(model,ema,source.generator,device,out,source.dataset,target.dataset)
    hooks=[model.register_forward_pre_hook(trace.pre),model.register_forward_hook(trace.post),
           ema.register_forward_pre_hook(trace.teacher_pre)]
    with (out/'history.jsonl').open('w') as history,(out/'actual_batches.jsonl').open('w') as batch_log:
        trace.history=history;trace.batch_log=batch_log
        run=native(trace);torch.cuda.synchronize(device);start=time.monotonic()
        run(model,ema,optimizer,criterion,12,Loader(source,trace,'source'),val,
            Loader(target,trace,'target'),Loader(target_eval,trace,'eval'),args,str(out),device,scheduler,trace)
        torch.cuda.synchronize(device);seconds=time.monotonic()-start
    for h in hooks:h.remove()
    assert len(trace.predictions)==20
    dump(out/'frozen_manifest.json',{'config':config,'train_seconds':seconds,'evaluations':trace.predictions,
                                   'total_steps':16000,'completed_epochs':200,'additional_training_tasks':0})
    print(json.dumps({'seed':a.seed,'completed':True,'train_seconds':seconds}),flush=True)


if __name__=='__main__':main()

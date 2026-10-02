"""The only new cell: original native mixed BN + existing center relations."""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
import torch
import torch.nn.functional as F

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'experiments/relation_tokenizer_v1'))
import relation
import importlib.util
spec=importlib.util.spec_from_file_location('historical_relation_train',ROOT/'experiments/relation_tokenizer_v1/train.py')
base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
from models.BiDA import BiDAnet


def build(seed):
    args=argparse.Namespace(model='BiDA',lr=.01,num_tokens=4,dim=64,depth=3,loss_type='softmax',labelsmooth='off')
    model=base.get_model('BiDA','Houston13',13,args)
    initial=base.helper.state_hash(model)
    unused=base.get_model('BiDA','Houston13',13,args,ema=True);del unused
    assert model.forward.__func__ is BiDAnet.forward
    native=model.forward.__func__
    relation.install(model,'C',seed)
    assert model._relation_forward_func is native
    assert model._tokenize.__func__ is BiDAnet._tokenize
    assert not hasattr(model,'_bn_arm')
    for module in (model.conv3d_features[1],model.conv2d_features[1]):
        assert type(module) in (torch.nn.BatchNorm3d,torch.nn.BatchNorm2d) and module.momentum==.1
    return model,initial


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--seed',type=int,choices=(2100,2101,2102),required=True)
    p.add_argument('--device',type=int,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--num-workers',type=int,default=4)
    p.add_argument('--dataset-dir',default='/home/zhangzj26/TGRS_MLUDA-2024/datasets/Houston/')
    args=p.parse_args()
    if args.out.exists() and any(args.out.iterdir()):p.error('Output must be empty')
    assert json.loads((ROOT/'results/relation_bn_2x2_v1/reuse_audit.json').read_text())['eligible_for_reuse']
    torch.set_num_threads(2)
    device=torch.device(f'cuda:{args.device}');torch.cuda.set_device(device)
    args.model,args.lr,args.num_tokens,args.dim,args.depth='BiDA',.01,4,64,3
    args.loss_type,args.labelsmooth,args.epochs='softmax','off',200
    base.seed_worker(args.seed)
    source,val,target,target_eval,_,classes=base.make_loaders(args)
    reference=ROOT/f'results/relation_tokenizer_v1/C_{args.seed}'
    ref_config=json.loads((reference/'config.json').read_text())
    ref_history=[json.loads(s) for s in (reference/'history.jsonl').read_text().splitlines()]
    model,initial=build(args.seed)
    assert initial==ref_config['initial_original_model_sha256']
    assert base.helper.state_hash(model)==ref_config['initial_full_model_sha256']
    assert base.helper.state_hash(model.relation_scores)==ref_config['initial_relation_mlp_sha256']
    counts=dict(source_n=len(source.dataset),source_val_n=len(val.dataset),target_n=len(target.dataset))
    assert all(ref_config[k]==v for k,v in counts.items())
    model.to(device)
    optimizer,scheduler=base.load_scheduler('BiDA',model,args);assert scheduler is None
    assert sum(p.numel() for p in model.parameters())==378011
    config={**ref_config,**counts,'cell':'2','arm':'Original BN + center-relative tokenizer',
            'bn':'Original sequential','bn_momentum':.1,'device':str(device),
            'initial_full_model_sha256':base.helper.state_hash(model),
            'historical_reference':str(reference),
            'new_protocol_sha256':base.helper.file_hash(Path(__file__).with_name('PROTOCOL.md')),
            'new_train_script_sha256':base.helper.file_hash(Path(__file__)),
            'reuse_audit_sha256':base.helper.file_hash(ROOT/'results/relation_bn_2x2_v1/reuse_audit.json'),
            'native_forward':'unmodified models.BiDA.BiDAnet.forward; source then target',
            'secondary':'target-oracle diagnostic grid10:10:200; no selection'}
    args.out.mkdir(parents=True);(args.out/'checkpoints').mkdir()
    (args.out/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    print(json.dumps(config),flush=True)
    records=[]
    torch.cuda.synchronize(device);start=time.monotonic()
    with (args.out/'history.jsonl').open('w') as history:
        for epoch in range(1,201):
            model.train();total=count=steps=0
            batches,rng=hashlib.sha256(),hashlib.sha256()
            for (x,y),(t,_) in zip(source,target):
                if len(x)!=len(t):continue
                for tensor in (x,y,t):batches.update(tensor.numpy().tobytes())
                rng.update(torch.get_rng_state().numpy().tobytes());rng.update(torch.cuda.get_rng_state(device).numpy().tobytes())
                x,y,t=x.to(device),y.to(device),t.to(device)
                optimizer.zero_grad(set_to_none=True)
                loss=F.cross_entropy(model(x,t)[0],y);assert torch.isfinite(loss)
                loss.backward();optimizer.step()
                total+=loss.item()*len(y);count+=len(y);steps+=1
            assert steps==18
            row={'epoch':epoch,'steps':steps,'source_ce':total/count,
                 'batch_stream_sha256':batches.hexdigest(),'original_rng_stream_sha256':rng.hexdigest()}
            for key in ('batch_stream_sha256','original_rng_stream_sha256'):assert row[key]==ref_history[epoch-1][key]
            if epoch==1 or epoch%10==0:
                row['source_val']=base.helper.source_evaluate(model,val,device);print(json.dumps(row),flush=True)
            if epoch%10==0:
                path=args.out/'checkpoints'/f'epoch_{epoch:03d}.pth'
                torch.save({'model':model.state_dict(),'epoch':epoch,'config':config},path)
                records.append({'epoch':epoch,'path':str(path),'sha256':base.helper.file_hash(path)})
            history.write(json.dumps(row)+'\n');history.flush()
    torch.cuda.synchronize(device)
    manifest={'config':config,'checkpoints':records,'train_seconds':time.monotonic()-start,
              'source_val_at_final':row['source_val']}
    (args.out/'training_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('Training complete; target inference is deferred to independent grid workers.',flush=True)


if __name__=='__main__':main()

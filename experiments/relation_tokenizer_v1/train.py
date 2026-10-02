"""Paired A/B/C training; defer all target scoring until all runs freeze."""
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
import relation

ROOT = Path(__file__).resolve().parents[2]
BIDA = Path('/home/zhangzj26/IEEE_TCSVT_BiDA')
sys.path.insert(0, str(BIDA))
sys.path.insert(0, str(BIDA/'experiments/bida_memory_v1'))
from experiments.bida_memory_v1.run import make_loaders
from models.get_model import get_model
from utils.scheduler import load_scheduler
from utils.utils_HSI import seed_worker


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


helper = module('relation_metrics', ROOT/'experiments/bida_self_dropout/train.py')
joint = module('relation_joint', ROOT/'experiments/bn_stabilization_v1/normalization.py')


def build(seed, arm):
    args = argparse.Namespace(model='BiDA', lr=.01, num_tokens=4, dim=64, depth=3,
                              loss_type='softmax', labelsmooth='off')
    model = get_model('BiDA', 'Houston13', 13, args)
    initial = helper.state_hash(model)
    unused = get_model('BiDA', 'Houston13', 13, args, ema=True)
    del unused
    joint.install(model, 'fixed_mixture')
    if arm != 'A':
        relation.install(model, arm, seed)
    return model, initial


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--arm', choices=('A','B','C'), required=True)
    p.add_argument('--seed', type=int, choices=(2100,2101,2102), required=True)
    p.add_argument('--device', type=int, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--num-workers', type=int, default=4)
    p.add_argument('--dataset-dir', default='/home/zhangzj26/TGRS_MLUDA-2024/datasets/Houston/')
    args = p.parse_args()
    if args.out.exists() and any(args.out.iterdir()):
        p.error('Output must be empty')
    torch.set_num_threads(2)
    device = torch.device(f'cuda:{args.device}')
    torch.cuda.set_device(device)
    args.model,args.lr,args.num_tokens,args.dim,args.depth='BiDA',.01,4,64,3
    args.loss_type,args.labelsmooth,args.epochs='softmax','off',200
    seed_worker(args.seed)
    source,val,target,target_eval,_,classes = make_loaders(args)
    reference = ROOT/f'results/sceneshift_joint_v1/C_{args.seed}'
    reference_config = json.loads((reference/'config.json').read_text())
    reference_history = [json.loads(s) for s in (reference/'history.jsonl').read_text().splitlines()]
    model, initial = build(args.seed, args.arm)
    assert initial == reference_config['initial_model_sha256']
    assert reference_config['bn']=='Full Joint' and reference_config['bn_momentum']==.19
    assert not reference_config['sceneshift'] and reference_config['loss']=='CE_s'
    assert reference_config['epochs']==200 and reference_config['seed']==args.seed
    assert reference_config['lr']==.01 and reference_config['depth']==3 and reference_config['dropout']==.1
    for loader,key in ((source,'source_indices_sha256'),(target,'target_indices_sha256')):
        assert hashlib.sha256(loader.dataset.indices.tobytes()).hexdigest()==reference_config[key]
    model.to(device)
    optimizer,scheduler = load_scheduler('BiDA',model,args)
    assert scheduler is None
    parameters = sum(v.numel() for v in model.parameters())
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out/'checkpoints').mkdir()
    files = [Path(__file__),Path(__file__).with_name('relation.py'),Path(__file__).with_name('PROTOCOL.md'),
             ROOT/'experiments/bn_stabilization_v1/normalization.py', BIDA/'models/BiDA.py',
             BIDA/'utils/dataset.py',BIDA/'utils/scheduler.py',BIDA/'utils/utils_HSI.py',
             BIDA/'experiments/bida_memory_v1/run.py']
    config = {'arm':args.arm,'seed':args.seed,'epochs':200,'device':str(device),
              'initial_original_model_sha256':initial,'initial_full_model_sha256':helper.state_hash(model),
              'initial_relation_mlp_sha256':helper.state_hash(model.relation_scores) if args.arm!='A' else None,
              'parameters':parameters,'source_n':len(source.dataset),'source_val_n':len(val.dataset),
              'target_n':len(target.dataset),'eps':1e-6,'relation_hidden':32,'tokens':4,'dim':64,
              'depth':3,'dropout':.1,'bn':'Full Joint','bn_momentum':.19,'lr':.01,
              'loss':'source self CE only','primary':'fixed_epoch_200','historical_reference':str(reference),
              'code_sha256':{str(path):helper.file_hash(path) for path in files},
              'data_sha256':{path.name:helper.file_hash(path) for path in sorted(Path(args.dataset_dir).glob('*.mat'))}}
    (args.out/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    print(json.dumps(config),flush=True)
    records=[]
    torch.cuda.synchronize(device)
    start=time.monotonic()
    with (args.out/'history.jsonl').open('w') as history:
        for epoch in range(1,201):
            model.train()
            total=count=steps=0
            batches,rng=hashlib.sha256(),hashlib.sha256()
            for (x,y),(t,_) in zip(source,target):
                if len(x)!=len(t): continue
                for tensor in (x,y,t): batches.update(tensor.numpy().tobytes())
                rng.update(torch.get_rng_state().numpy().tobytes())
                rng.update(torch.cuda.get_rng_state(device).numpy().tobytes())
                x,y,t=x.to(device),y.to(device),t.to(device)
                optimizer.zero_grad(set_to_none=True)
                loss=F.cross_entropy(model(x,t)[0],y)
                assert torch.isfinite(loss)
                loss.backward()
                optimizer.step()
                total+=loss.item()*len(y);count+=len(y);steps+=1
            assert steps==18
            row={'epoch':epoch,'steps':steps,'source_ce':total/count,
                 'batch_stream_sha256':batches.hexdigest(),'original_rng_stream_sha256':rng.hexdigest()}
            ref=reference_history[epoch-1]
            for key in ('batch_stream_sha256','original_rng_stream_sha256'):
                assert row[key]==ref[key],(args.arm,args.seed,epoch,key)
            if args.arm=='A': assert abs(row['source_ce']-ref['source_ce'])<1e-7
            if epoch==1 or epoch%10==0:
                row['source_val']=helper.source_evaluate(model,val,device)
                print(json.dumps(row),flush=True)
            if epoch%10==0:
                path=args.out/'checkpoints'/f'epoch_{epoch:03d}.pth'
                torch.save({'model':model.state_dict(),'epoch':epoch,'config':config},path)
                records.append({'epoch':epoch,'path':str(path),'sha256':helper.file_hash(path)})
            history.write(json.dumps(row)+'\n');history.flush()
    torch.cuda.synchronize(device)
    seconds=time.monotonic()-start
    replay=None
    if args.arm=='A':
        ref=torch.load(reference/'checkpoints/epoch_200.pth',map_location='cpu',weights_only=True)['model']
        assert all(torch.equal(v.detach().cpu(),ref[k]) for k,v in model.state_dict().items())
        replay={'historical_state_exact':True,'path':str(reference/'checkpoints/epoch_200.pth')}
    start=time.monotonic()
    logits,predictions=helper.predict(model,target_eval,device)
    path=args.out/'predictions_200.npz'
    np.savez_compressed(path,logits=logits,predictions=predictions)
    manifest={'config':config,'checkpoints':records,'prediction':{'path':str(path),'sha256':helper.file_hash(path)},
              'train_seconds':seconds,'inference_seconds':time.monotonic()-start,
              'source_val_at_final':row['source_val'],'historical_replay':replay}
    (args.out/'frozen_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    # No target label collection or scoring here. The runner scores after all nine freeze.
    print(json.dumps({'frozen_manifest_sha256':helper.file_hash(args.out/'frozen_manifest.json')}),flush=True)


if __name__=='__main__': main()

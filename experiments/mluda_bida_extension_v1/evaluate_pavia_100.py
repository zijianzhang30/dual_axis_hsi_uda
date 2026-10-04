"""Independent once-only diagnostic; do not signal or modify training jobs."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
import common as c
summary_tools=c.module('pavia_extension_summary_tools',Path(__file__).with_name('summarize.py'))
profile,stats=summary_tools.profile,summary_tools.stats


def paths():
    return {(arm,seed):(c.ROOT/'results/pavia_joint_v1'/('joint_%d'%seed)
                       if arm=='B' else c.OUT/'pavia'/('C_%d'%seed))
            for arm in ('B','C') for seed in (2100,2101,2102)}


def ready(folders):
    for (arm,seed),folder in folders.items():
        checkpoint=folder/'checkpoints/epoch_100.pth'
        history=folder/'history.jsonl'
        lines=history.read_text().splitlines() if history.exists() else []
        try: epoch=json.loads(lines[-1])['epoch'] if lines else 0
        except json.JSONDecodeError: return False
        if epoch<100 or not checkpoint.exists():
            if arm=='C':
                log=c.OUT/('pavia_C_%d.log'%seed)
                if log.exists() and 'Traceback (most recent call last)' in log.read_text():
                    raise RuntimeError('Training failed; no automatic restart or changed endpoint')
            return False
    return True


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--device',type=int,default=7)
    args=parser.parse_args()
    out=c.OUT/'diagnostic_pavia_epoch100'
    if out.exists():raise RuntimeError('Refuse to overwrite existing diagnostic')
    folders=paths()
    print('Waiting for all three Pavia C completed epoch-100 checkpoints; training stays at 200.',flush=True)
    while not ready(folders):time.sleep(15)
    out.mkdir()
    torch.set_num_threads(2)
    device=torch.device('cuda:%d'%args.device);torch.cuda.set_device(device)
    c.pavia.seed_worker(2100)
    values=c.loaders('pavia',2100,4);loader=values[3]
    data=c.inputs_audit('pavia',values)
    records=[]
    for (arm,seed),folder in folders.items():
        checkpoint=folder/'checkpoints/epoch_100.pth'
        digest=c.helper.file_hash(checkpoint)
        state=torch.load(checkpoint,map_location='cpu',weights_only=True)
        assert state['epoch']==100
        assert state['config']['data_sha256']==data['data_sha256']
        assert state['config']['sample_counts']['target_eval']==len(loader.dataset)
        c.pavia.seed_worker(seed)
        model,initial=c.build('pavia',arm,seed)
        assert initial==state['config']['initial_model_sha256']
        c.historical('pavia',seed,initial,values)
        model.load_state_dict(state['model'],strict=True);del state
        model.to(device)
        logits,predictions=c.helper.predict(model,loader,device)
        path=out/('%s_%d.npz'%(arm,seed))
        np.savez_compressed(path,logits=logits,predictions=predictions)
        assert digest==c.helper.file_hash(checkpoint)
        records.append({'arm':arm,'seed':seed,'checkpoint':str(checkpoint),
                        'checkpoint_sha256':digest,'prediction':str(path),
                        'prediction_sha256':c.helper.file_hash(path)})
        print('Frozen %s %d epoch100'%(arm,seed),flush=True)
        del model,logits,predictions
    c.dump(out/'frozen_manifest.json',{'epoch':100,'records':records,'data':data,
           'addendum_sha256':c.helper.file_hash(Path(__file__).with_name('EPOCH100_ADDENDUM.md')),
           'script_sha256':c.helper.file_hash(Path(__file__))})
    labels=np.concatenate([y.numpy() for _,y in loader])
    details=[];by_arm={'B':[],'C':[]}
    for record in records:
        assert c.helper.file_hash(Path(record['prediction']))==record['prediction_sha256']
        with np.load(record['prediction']) as artifact:
            metric=c.helper.metrics(labels,artifact['predictions'],artifact['logits'],7)
        by_arm[record['arm']].append(metric)
        details.append({**record,**metric,**profile(metric)})
    paired=[]
    for seed in (2100,2101,2102):
        b=next(d for d in details if d['arm']=='B' and d['seed']==seed)
        new=next(d for d in details if d['arm']=='C' and d['seed']==seed)
        paired.append({'seed':seed,**{k:new[k]-b[k] for k in ('oa_percent','aa_percent','kappa')}})
    result={'epoch':100,'endpoint':'user-requested diagnostic only; fixed200 stays primary',
            'details':details,'aggregate':{a:stats(r) for a,r in by_arm.items()},
            'paired_C_minus_B':paired,'paired_aggregate':stats(paired),
            'frozen_manifest_sha256':c.helper.file_hash(out/'frozen_manifest.json')}
    c.dump(out/'summary.json',result)
    lines=['# Pavia epoch-100 diagnostic','',result['endpoint'],'',
           '| Arm | OA mean±SD | AA mean±SD | Kappa mean±SD | Single-class collapse |',
           '| --- | ---: | ---: | ---: | ---: |']
    for arm in ('B','C'):
        a=result['aggregate'][arm]
        text=['%.3f ± %.3f'%(a[k]['mean'],a[k]['sample_std']) for k in ('oa_percent','aa_percent','kappa')]
        lines.append('| %s | %s | %s | %s | %d/3 |'%(arm,*text,sum(d['single_class_collapse'] for d in details if d['arm']==arm)))
    lines += ['','| Seed | C−B OA (pp) | C−B AA (pp) | C−B Kappa |','| --- | ---: | ---: | ---: |']
    for row in paired:lines.append('| %d | %+.3f | %+.3f | %+.5f |'%(row['seed'],row['oa_percent'],row['aa_percent'],row['kappa']))
    lines += ['','Full per-class recalls, prediction distributions and zero-recall classes: summary.json.',
              'All six predictions frozen before scoring. No training process, model selection or protocol changed.']
    (out/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(result['aggregate']),flush=True)


if __name__=='__main__':main()

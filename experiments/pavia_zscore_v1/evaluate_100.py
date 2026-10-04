"""Once-only epoch100 diagnostics, isolated from both training processes."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
import common as c


def ready():
    for arm in ('B','C'):
        folder=c.OUT/(arm+'_2100');history=folder/'history.jsonl'
        lines=history.read_text().splitlines() if history.exists() else []
        try:epoch=json.loads(lines[-1])['epoch'] if lines else 0
        except json.JSONDecodeError:return False
        if epoch<100 or not (folder/'checkpoints/epoch_100.pth').exists():
            log=c.OUT/(arm+'_2100.log')
            if log.exists() and 'Traceback (most recent call last)' in log.read_text():
                raise RuntimeError('Training failed; do not change/restart from this diagnostic')
            return False
    return True


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--device',type=int,default=7)
    args=parser.parse_args()
    out=c.OUT/'diagnostic_epoch100'
    if out.exists():raise RuntimeError('Refuse to overwrite diagnostic')
    print('Waiting for both complete seed2100 epoch100 checkpoints; training continues to200.',flush=True)
    while not ready():time.sleep(15)
    out.mkdir()
    torch.set_num_threads(2);device=torch.device('cuda:%d'%args.device)
    torch.cuda.set_device(device)
    image,gt=c.load_scene('pavia')
    image=np.pad(image,((6,6),(6,6),(0,0)),mode='reflect')
    gt=np.pad(gt,((6,6),(6,6)),mode='reflect')
    dataset=c.pavia.HSIDataset(image,gt,13,data_aug=False)
    loader=torch.utils.data.DataLoader(dataset,batch_size=128,shuffle=False,
                                      drop_last=False,num_workers=4)
    indices_hash=hashlib.sha256(dataset.indices.tobytes()).hexdigest()
    old_out=c.base.OUT/'diagnostic_pavia_epoch100'
    old_manifest=json.loads((old_out/'frozen_manifest.json').read_text())
    assert old_manifest['data']['indices_sha256']['3']==indices_hash
    records=[]
    for arm in ('B','C'):
        folder=c.OUT/(arm+'_2100');checkpoint=folder/'checkpoints/epoch_100.pth'
        digest=c.helper.file_hash(checkpoint)
        state=torch.load(checkpoint,map_location='cpu',weights_only=True)
        assert state['epoch']==100 and state['config']['seed']==2100
        assert state['config']['indices_sha256']['3']==indices_hash
        assert state['config']['sample_counts']['target_eval']==len(dataset)
        assert 'sklearn.preprocessing.scale' in state['config']['norm']
        c.pavia.seed_worker(2100)
        model,initial=c.base.build('pavia',arm,2100)
        assert initial==state['config']['initial_model_sha256']
        model.load_state_dict(state['model'],strict=True);del state
        model.to(device)
        logits,predictions=c.helper.predict(model,loader,device)
        path=out/(arm+'_2100.npz')
        np.savez_compressed(path,logits=logits,predictions=predictions)
        assert digest==c.helper.file_hash(checkpoint)
        records.append({'arm':arm,'normalization':'zscore','seed':2100,
            'checkpoint':str(checkpoint),'checkpoint_sha256':digest,
            'prediction':str(path),'prediction_sha256':c.helper.file_hash(path)})
        old=next(r for r in old_manifest['records'] if r['arm']==arm and r['seed']==2100)
        assert c.helper.file_hash(Path(old['prediction']))==old['prediction_sha256']
        assert c.helper.file_hash(Path(old['checkpoint']))==old['checkpoint_sha256']
        records.append({**old,'normalization':'normband'})
        print('Frozen zscore %s epoch100; normband reference verified'%arm,flush=True)
        del model,logits,predictions
    c.dump(out/'frozen_manifest.json',{'epoch':100,'seed':2100,'records':records,
        'target_indices_sha256':indices_hash,'target_eval_count':len(dataset),
        'addendum_sha256':c.helper.file_hash(Path(__file__).with_name('EPOCH100_ADDENDUM.md')),
        'script_sha256':c.helper.file_hash(Path(__file__))})
    # All new predictions and old reference hashes are frozen before scoring.
    labels=np.asarray(dataset.labels,dtype=np.int64)
    rows={};deltas={}
    for record in records:
        with np.load(record['prediction']) as artifact:
            metric=c.helper.metrics(labels,artifact['predictions'],artifact['logits'],7)
        zeros=[i+1 for i,x in enumerate(metric['per_class_percent']) if x==0]
        metric.update(single_class_collapse=max(metric['prediction_shares'])>=.95,
            zero_recall_class_count=len(zeros),zero_recall_classes_one_based=zeros)
        rows[record['arm']+'_'+record['normalization']]=metric
    for arm in ('B','C'):
        deltas[arm]={k:rows[arm+'_zscore'][k]-rows[arm+'_normband'][k]
                    for k in ('oa_percent','aa_percent','kappa')}
    result={'epoch':100,'seed':2100,'endpoint':'mid-training diagnostic, fixed200 remains primary',
            'runs':rows,'zscore_minus_normband':deltas,
            'frozen_manifest_sha256':c.helper.file_hash(out/'frozen_manifest.json')}
    c.dump(out/'summary.json',result)
    lines=['# Pavia epoch100 normalization diagnostic, seed2100','',result['endpoint'],
           'No model selection/tuning/early stopping. Single-seed scores, no stability claims.','',
           '| Tokenizer | Norm | OA (%) | AA (%) | Kappa | Single-class collapse | Zero-recall classes |',
           '| --- | --- | ---: | ---: | ---: | --- | --- |']
    for arm in ('B','C'):
        for norm in ('normband','zscore'):
            m=rows[arm+'_'+norm]
            lines.append('| %s | %s | %.3f | %.3f | %.5f | %s | %s |'%(arm,norm,m['oa_percent'],
                m['aa_percent'],m['kappa'],m['single_class_collapse'],m['zero_recall_classes_one_based']))
    lines += ['','| Tokenizer | Z-score−normband OA (pp) | AA (pp) | Kappa |','| --- | ---: | ---: | ---: |']
    for arm,d in deltas.items():lines.append('| %s | %+.3f | %+.3f | %+.5f |'%(arm,d['oa_percent'],d['aa_percent'],d['kappa']))
    lines += ['','Per-class recalls and prediction distributions are preserved in summary.json.',
              'All predictions frozen/verified before scoring; training processes untouched.']
    (out/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()

"""Strict paired hash audit, fixed endpoints, deltas, cost and Go/No-Go."""
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/sceneshift_joint_v1'
SEEDS=[2100,2101,2102]

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for x in iter(lambda:f.read(1024*1024),b''):h.update(x)
    return h.hexdigest()

def stats(values):
    x=np.asarray(values,dtype=float)
    return {'mean':float(x.mean()),'sample_sd':float(x.std(ddof=1)),
            'values':x.tolist(),'positive_seeds':int((x>0).sum())}

def main():
    runs={};files_verified=0
    for arm in 'ABCD':
        runs[arm]={}
        for seed in SEEDS:
            folder=OUT/f'{arm}_{seed}'
            r=json.loads((folder/'results.json').read_text())
            c=json.loads((folder/'config.json').read_text())
            h=[json.loads(x) for x in (folder/'history.jsonl').read_text().splitlines()]
            assert len(h)==200 and [x['epoch'] for x in h]==list(range(1,201))
            assert all(x['steps']==18 for x in h)
            assert r['source_val_at_final']==h[-1]['source_val']
            assert r['parameters']==376567 and c['arm']==arm and c['seed']==seed
            manifest_path=folder/'frozen_manifest.json'
            assert sha(manifest_path)==r['frozen_manifest_sha256']
            manifest=json.loads(manifest_path.read_text())
            assert manifest['config']==c
            assert [v['epoch'] for v in manifest['checkpoints']]==list(range(10,201,10))
            assert [v['epoch'] for v in manifest['predictions']]==[200]
            for item in manifest['checkpoints']+manifest['predictions']:
                assert sha(item['path'])==item['sha256'];files_verified+=1
            runs[arm][seed]={'result':r,'config':c,'history':h}
    audit={}
    for seed in SEEDS:
        ref=runs['A'][seed]
        for arm in 'BCD':
            for k in ('initial_model_sha256','trainable_parameters','depth','dropout','lr',
                      'source_n','source_val_n','target_n','source_indices_sha256','target_indices_sha256',
                      'band_stats','protocol_sha256','train_script_sha256','shift_script_sha256','normalization_script_sha256'):
                assert runs[arm][seed]['config'][k]==ref['config'][k],(seed,arm,k)
            for x,y in zip(ref['history'],runs[arm][seed]['history']):
                for k in ('batch_stream_sha256','original_rng_stream_sha256'):
                    assert x[k]==y[k],(seed,arm,x['epoch'],k)
        audit[str(seed)]={'equal_initial_tensors':True,'equal_batch_stream_all_200_epochs':True,
                         'equal_original_rng_stream_all_200_epochs':True}
    aggregate={};rows=[]
    for arm in 'ABCD':
        rs=[runs[arm][s]['result'] for s in SEEDS]
        aggregate[arm]={k:stats([r['target_at_final'][k] for r in rs]) for k in ('oa_percent','aa_percent','kappa')}
        aggregate[arm].update({'source_val_oa':stats([r['source_val_at_final']['oa_percent'] for r in rs]),
                              'train_seconds':stats([r['train_seconds'] for r in rs]),
                              'preparation_seconds':stats([r['preparation_seconds'] for r in rs]),
                              'auxiliary_cuda_seconds':stats([r['auxiliary_forward_loss_cuda_ms']/1000 for r in rs]),
                              'class6_share_percent':stats([r['target_at_final']['prediction_shares'][5]*100 for r in rs]),
                              'collapse_count':sum(r['target_at_final']['class6_collapse'] for r in rs),
                              'per_class_mean':np.mean([r['target_at_final']['per_class_percent'] for r in rs],axis=0).tolist(),
                              'per_class_sample_sd':np.std([r['target_at_final']['per_class_percent'] for r in rs],axis=0,ddof=1).tolist()})
        for seed in SEEDS:
            r=runs[arm][seed]['result']
            rows.append({'arm':arm,'seed':seed,**r,
                         'zero_recall_classes':[i+1 for i,v in enumerate(r['target_at_final']['per_class_percent']) if v==0]})
    deltas={}
    for name,left,right in [('B-A','B','A'),('C-A','C','A'),('D-C','D','C')]:
        deltas[name]={}
        for k in ('oa_percent','aa_percent','kappa'):
            deltas[name][k]=stats([runs[left][s]['result']['target_at_final'][k]-runs[right][s]['result']['target_at_final'][k] for s in SEEDS])
        deltas[name]['train_seconds']=stats([runs[left][s]['result']['train_seconds']-runs[right][s]['result']['train_seconds'] for s in SEEDS])
        deltas[name]['train_time_ratio']=stats([runs[left][s]['result']['train_seconds']/runs[right][s]['result']['train_seconds'] for s in SEEDS])
    interaction={k:stats([x-y for x,y in zip(deltas['D-C'][k]['values'],deltas['B-A'][k]['values'])]) for k in ('oa_percent','aa_percent','kappa')}
    dc=deltas['D-C']
    gains={k:dc[k]['mean']>=1 and dc[k]['positive_seeds']>=2 for k in ('oa_percent','aa_percent')}
    collapse_ok=aggregate['D']['collapse_count']<=aggregate['C']['collapse_count']
    gate={'operational_material_threshold_pp':1,'majority_positive_same_metric':gains,
          'no_collapse_count_increase':collapse_ok,'go':any(gains.values()) and collapse_ok}
    output={'primary':'fixed_epoch_200','seeds':SEEDS,'sd':'sample ddof=1',
            'aggregate':aggregate,'rows':rows,'paired_deltas':deltas,'interaction':interaction,
            'gate':gate,'audit':audit,'artifact_files_verified':files_verified}
    dest=OUT/'summary.json'
    if dest.exists():raise RuntimeError('Refuse to overwrite existing summary')
    dest.write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps({'aggregate':aggregate,'paired_deltas':deltas,'interaction':interaction,'gate':gate,'audit':audit},indent=2))

if __name__=='__main__':main()

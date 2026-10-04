"""Strictly reuse matching frozen Pavia A/B/C; never select a model."""
import importlib.util
import json
from pathlib import Path
import shutil

import numpy as np
from scipy.io import loadmat

ROOT = Path('/home/zhangzj26/dual_axis_hsi_uda')
OUT = Path('/nas1/zhangzj26/dual_axis_hsi_uda/results/pavia_transfer_audit_v1')
spec = importlib.util.spec_from_file_location('reuse_transfer_common', ROOT / 'experiments/mluda_bida_extension_v1/common.py')
c = importlib.util.module_from_spec(spec)
spec.loader.exec_module(c)


def stats(values):
    return {'mean': float(np.mean(values)), 'sample_std': float(np.std(values, ddof=1))}


def main():
    assert json.loads((OUT / 'verification.json').read_text())['passed'], 'Implementation check must pass'
    assert not (OUT / 'summary.json').exists(), 'Do not overwrite'
    audit = json.loads((OUT / 'verification.json').read_text())['data_protocol']
    sources = {'historical_train': ROOT / 'experiments/pavia_joint_v1/train.py',
               'extension_train': ROOT / 'experiments/mluda_bida_extension_v1/train.py',
               'extension_common': ROOT / 'experiments/mluda_bida_extension_v1/common.py',
               'relation': ROOT / 'experiments/relation_tokenizer_v1/relation.py',
               'normalization': ROOT / 'experiments/bn_stabilization_v1/normalization.py',
               'model': c.pavia.BIDA / 'models/BiDA.py', 'dataset': c.pavia.BIDA / 'utils/dataset.py'}
    snapshot = OUT / 'source_snapshot'
    snapshot.mkdir()
    for name, path in sources.items(): shutil.copy2(str(path), str(snapshot / (name + '.py')))
    rows, traces, manifests = {}, {}, {}
    for seed in (2100,2101,2102):
        for arm in ('A','B','C'):
            if arm == 'C':
                folder = OUT.parent / 'mluda_bida_extension_v1/pavia' / ('C_%d' % seed)
                expected_script = sources['extension_train']
                expected_protocol = ROOT / 'experiments/mluda_bida_extension_v1/PROTOCOL.md'
            else:
                folder = ROOT / 'results/pavia_joint_v1' / (('%s_%d' % ('original' if arm=='A' else 'joint',seed)))
                expected_script = sources['historical_train']
                expected_protocol = ROOT / 'experiments/pavia_joint_v1/PROTOCOL.md'
            config = json.loads((folder / 'config.json').read_text())
            assert config['seed'] == seed and config['epochs'] == 200
            assert config['primary'] == 'fixed_epoch_200'
            assert config['sample_counts'] == dict(zip(('source_train','source_val','target_train','target_eval'), audit['sample_counts']))
            assert config['data_sha256'] == audit['data_sha256']
            assert config['script_sha256'] == c.helper.file_hash(expected_script)
            assert config['protocol_sha256'] == c.helper.file_hash(expected_protocol)
            for key, path in (('normalization_sha256', sources['normalization']),
                              ('model_code_sha256', sources['model']), ('dataset_code_sha256',sources['dataset'])):
                assert config[key] == c.helper.file_hash(path), (arm,seed,key)
            if arm == 'C':
                assert config['arm']=='C' and config['indices_sha256'] == audit['indices_sha256']
                assert config['relation_sha256'] == c.helper.file_hash(sources['relation'])
                assert config['bn_momentum']==.19 and config['patch']==13 and config['loss']=='source CE only'
            else:
                assert config['arm'] == ('original' if arm=='A' else 'joint')
            result = json.loads((folder / 'results.json').read_text())
            manifest = json.loads((folder / 'frozen_manifest.json').read_text())
            assert result['frozen_manifest_sha256'] == c.helper.file_hash(folder / 'frozen_manifest.json')
            assert manifest['config'] == config
            prediction = folder / 'predictions_epoch_200.npz'
            assert manifest['prediction']['sha256'] == c.helper.file_hash(prediction)
            for checkpoint in manifest['checkpoints']:
                assert checkpoint['sha256'] == c.helper.file_hash(folder / 'checkpoints' / ('epoch_%03d.pth' % checkpoint['epoch']))
            history = [json.loads(line) for line in (folder / 'history.jsonl').read_text().splitlines()]
            assert len(history)==200 and [r['epoch'] for r in history]==list(range(1,201))
            assert all(r['steps']==290 for r in history)
            key = '%s_%d' % (arm,seed)
            traces[key] = history
            manifests[key] = {'folder': str(folder.resolve()),
                              'config_sha256': c.helper.file_hash(folder / 'config.json'),
                              'frozen_manifest_sha256': result['frozen_manifest_sha256'],
                              'prediction_sha256': manifest['prediction']['sha256'],
                              'history_sha256': c.helper.file_hash(folder / 'history.jsonl')}
            rows[key] = {'seed': seed, 'arm': arm, 'config':config, 'result': result}
        assert len({rows['%s_%d' % (arm,seed)]['config']['initial_model_sha256'] for arm in ('A','B','C')})==1
        for left,right in (('A','B'),('B','C')):
            for a,b in zip(traces['%s_%d'%(left,seed)],traces['%s_%d'%(right,seed)]):
                for field in ('epoch','steps','batch_sha256','rng_sha256'):
                    assert a[field]==b[field], (seed,left,right,field,a['epoch'])
    # All predictions have been verified as already frozen before raw GT scoring.
    gt = loadmat(c.DATA / 'Pavia/pavia_gt_7.mat')['pavia_gt_7'].astype(int)-1
    indices = np.argwhere(gt>=0)
    indices = indices[(indices[:,0]>0)&(indices[:,1]>0)]
    labels = gt[indices[:,0],indices[:,1]]
    assert len(labels)==39348
    records = {}
    for key,row in rows.items():
        folder=Path(manifests[key]['folder'])
        with np.load(folder / 'predictions_epoch_200.npz') as prediction:
            score = c.helper.metrics(labels,prediction['predictions'],prediction['logits'],7)
        for field in ('oa_percent','aa_percent','kappa','per_class_percent','prediction_shares'):
            assert np.allclose(score[field], row['result']['target_at_final'][field],atol=1e-10,rtol=0), (key,field)
        score.update(single_class_collapse=max(score['prediction_shares'])>=.95,
                     zero_recall_classes=[i+1 for i,v in enumerate(score['per_class_percent']) if v==0])
        records[key]={'seed':row['seed'],'arm':row['arm'],'metrics':score,
                      'source_val':row['result']['source_val_at_final'],
                      'parameters':row['config']['trainable_parameters'],
                      'train_seconds':row['result']['train_seconds']}
    fields=('oa_percent','aa_percent','kappa')
    summary={arm:{field:stats([records['%s_%d'%(arm,seed)]['metrics'][field] for seed in (2100,2101,2102)])
                  for field in fields} for arm in ('A','B','C')}
    paired={}
    for name,left,right in (('B_minus_A','B','A'),('C_minus_B','C','B'),('C_minus_A','C','A')):
        differences={field:[records['%s_%d'%(left,s)]['metrics'][field]-records['%s_%d'%(right,s)]['metrics'][field]
                           for s in (2100,2101,2102)] for field in fields}
        paired[name]={'by_seed':differences, 'summary':{field:stats(values) for field,values in differences.items()}}
    result={'benchmark':'PaviaU -> PaviaC, frozen BiDA-style normband protocol',
            'epoch':200,'all_nine_runs_reused':True,'new_full_training_tasks':0,
            'all_200_epochs_actual_augmented_batches_and_model_RNG_match':True,
            'records':records,'summary':summary,'paired':paired,'provenance':manifests,
            'source_sha256':{name:c.helper.file_hash(path) for name,path in sources.items()}}
    c.dump(OUT / 'summary.json',result)
    lines=['# Pavia frozen three-arm transfer validation','',
           'All nine matching historical runs reused. No new full training, target-based selection or tuning.',
           'Implementation checks passed; see verification.json. This is not a full BiDA/EMA reproduction.',
           '', '| Arm | OA mean +/- sample std (%) | AA (%) | Kappa | Single-class collapse |',
           '| --- | ---: | ---: | ---: | ---: |']
    for arm in ('A','B','C'):
        s=summary[arm]
        vals=['%.3f +/- %.3f'%(s[f]['mean'],s[f]['sample_std']) for f in fields]
        count=sum(records['%s_%d'%(arm,seed)]['metrics']['single_class_collapse'] for seed in (2100,2101,2102))
        lines.append('| %s | %s | %s | %s | %d/3 |'%(arm,*vals,count))
    lines+=['','A: Original Mixed BN + original tokenizer. B: Full Joint BN + original tokenizer.',
            'C: Full Joint BN + center-relative tokenizer.', '',
            '| Seed | B-A OA / AA (pp) | C-B OA / AA (pp) | C-A OA / AA (pp) |',
            '| --- | ---: | ---: | ---: |']
    for i,seed in enumerate((2100,2101,2102)):
        vals=['%+.3f / %+.3f'%(paired[name]['by_seed']['oa_percent'][i],paired[name]['by_seed']['aa_percent'][i])
              for name in ('B_minus_A','C_minus_B','C_minus_A')]
        lines.append('| %d | %s | %s | %s |'%(seed,*vals))
    for key,row in records.items():
        m=row['metrics']
        lines+=['', '## '+key, '',
                'OA %.3f; AA %.3f; Kappa %.5f; source-val OA %.3f; parameters %d; train seconds %.1f.'%
                (m['oa_percent'],m['aa_percent'],m['kappa'],row['source_val']['oa_percent'],row['parameters'],row['train_seconds']),
                'Recall (classes 1-7): '+', '.join('%.2f'%v for v in m['per_class_percent']),
                'Prediction shares (%): '+', '.join('%.2f'%(v*100) for v in m['prediction_shares']),
                'Zero-recall classes: '+str(m['zero_recall_classes'])]
    lines+=['', 'Joint BN does not show a stable positive Pavia replication; tokenizer OA improves in all three seeds.',
            'Do not conflate this same-protocol finding with native MLUDA paper-score reproduction.']
    (OUT / 'RESULTS.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'summary':summary,'paired':paired},indent=2))


if __name__=='__main__':main()

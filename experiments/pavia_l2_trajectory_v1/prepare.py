"""Check frozen provenance and inventory all120 expected saved checkpoints."""
import json
from pathlib import Path
import shutil
import torch
import common as t


def main():
    t.OUT.mkdir(exist_ok=False)
    sources=list(Path(__file__).parent.glob('*.py'))+[Path(__file__).with_name('PROTOCOL.md'),
         t.ROOT/'experiments/pavia_l2_fixed_v1/common.py',
         t.ROOT/'experiments/pavia_l2_fixed_v1/train.py',
         t.ROOT/'experiments/pavia_l2_fixed_diagnostic_v1/stages.py',
         t.ROOT/'experiments/bn_stabilization_v1/normalization.py',
         t.ROOT/'experiments/relation_tokenizer_v1/relation.py',
         t.c.pavia.BIDA/'models/BiDA.py',t.c.pavia.BIDA/'utils/dataset.py']
    snapshot=t.OUT/'code_snapshot';snapshot.mkdir()
    hashes={}
    for p in sources:
        shutil.copy2(str(p),str(snapshot/str(p).replace('/','__')))
        hashes[str(p)]=t.c.helper.file_hash(p)
    data_hashes={p.name:t.c.helper.file_hash(p) for p in sorted((t.c.DATA/'Pavia').glob('*.mat'))}
    runs={};missing=[]
    for seed in t.SEEDS:
        for arm in t.ARMS:
            name='%s_%d'%(arm,seed);folder=t.c.OUT/name
            cfg=json.loads((folder/'config.json').read_text())
            manifest=json.loads((folder/'frozen_manifest.json').read_text())
            assert cfg['normalization']=='native spectral L2; repaired flat-to-cube return'
            assert cfg['seed']==seed and cfg['arm']==arm and cfg['epochs']==200
            assert cfg==manifest['config'] and cfg['data_sha256']==data_hashes
            for key,p in (('common_sha256',t.ROOT/'experiments/pavia_l2_fixed_v1/common.py'),
                          ('script_sha256',t.ROOT/'experiments/pavia_l2_fixed_v1/train.py'),
                          ('model_code_sha256',t.c.pavia.BIDA/'models/BiDA.py'),
                          ('dataset_code_sha256',t.c.pavia.BIDA/'utils/dataset.py'),
                          ('normalization_sha256',t.ROOT/'experiments/bn_stabilization_v1/normalization.py'),
                          ('relation_sha256',t.ROOT/'experiments/relation_tokenizer_v1/relation.py')):
                assert cfg[key]==t.c.helper.file_hash(p), (name,key)
            frozen={r['epoch']:r for r in manifest['checkpoints']}
            records=[]
            for epoch in t.EPOCHS:
                p=folder/'checkpoints'/('epoch_%03d.pth'%epoch)
                if not p.exists():
                    record={'epoch':epoch,'path':str(p),'status':'missing'};missing.append(dict(record,run=name))
                else:
                    assert epoch in frozen, (name,epoch,'not frozen in manifest')
                    sha=t.c.helper.file_hash(p);assert sha==frozen[epoch]['sha256']
                    checkpoint=torch.load(p,map_location='cpu')
                    assert checkpoint['epoch']==epoch and checkpoint['config']==cfg
                    record={'epoch':epoch,'path':str(p),'status':'present','sha256':sha}
                    del checkpoint
                records.append(record)
            runs[name]={'config':cfg,'checkpoints':records,
                        'frozen_manifest_sha256':t.c.helper.file_hash(folder/'frozen_manifest.json'),
                        'final_prediction_sha256':manifest['prediction']['sha256']}
    result={'only_repaired_L2_runs':True,'source_sha256':hashes,'data_sha256':data_hashes,
            'runs':runs,'missing_checkpoints':missing,'expected_checkpoint_n':120,
            'present_checkpoint_n':sum(r['status']=='present' for run in runs.values() for r in run['checkpoints'])}
    t.c.dump(t.OUT/'inventory.json',result)
    link=t.ROOT/'results/pavia_l2_trajectory_v1'
    assert not (link.exists() or link.is_symlink())
    link.symlink_to(t.OUT,target_is_directory=True)
    print(json.dumps({'present':result['present_checkpoint_n'],'missing':missing},indent=2),flush=True)


if __name__=='__main__':main()

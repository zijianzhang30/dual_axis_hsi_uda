"""Read-only eligibility audit for the three historical cells."""
import hashlib
import json
from pathlib import Path
import torch

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/relation_bn_2x2_v1'
SEEDS=(2100,2101,2102)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def folder(cell,seed):
    return ROOT/(f'results/sceneshift_joint_v1/A_{seed}' if cell=='1' else
                 f'results/relation_tokenizer_v1/{"A" if cell=="3" else "C"}_{seed}')


def read(path):return json.loads(Path(path).read_text())


def run():
    evidence={}
    for seed in SEEDS:
        cells={cell:folder(cell,seed) for cell in ('1','3','4')}
        configs={k:read(v/'config.json') for k,v in cells.items()}
        histories={k:[json.loads(s) for s in (v/'history.jsonl').read_text().splitlines()] for k,v in cells.items()}
        for cell,path in cells.items():
            c=configs[cell];h=histories[cell];r=read(path/'results.json');m=read(path/'frozen_manifest.json')
            assert c['seed']==seed and c['epochs']==200
            assert c['depth']==3 and c['dropout']==.1 and c['lr']==.01
            assert c['primary']=='fixed_epoch_200'
            assert (c['source_n'],c['source_val_n'],c['target_n'])==(2393,127,52901)
            assert [v['epoch'] for v in h]==list(range(1,201)) and all(v['steps']==18 for v in h)
            assert sha(path/'frozen_manifest.json')==r['frozen_manifest_sha256']
            for item in m['checkpoints']+(m['predictions'] if cell=='1' else [m['prediction']]):
                assert sha(item['path'])==item['sha256']
            if cell=='1':
                assert c['bn']=='Original sequential' and c['bn_momentum']==.1
                assert c['loss']=='CE_s' and not c['sceneshift']
                assert c['trainable_parameters']==376567
                assert sha(ROOT/'experiments/sceneshift_joint_v1/train.py')==c['train_script_sha256']
                assert r['historical_replay']['state_exact']
            else:
                assert c['bn']=='Full Joint' and c['bn_momentum']==.19
                assert c['loss']=='source self CE only'
                for code,hash_value in c['code_sha256'].items():assert sha(code)==hash_value
                assert c['parameters']==(376567 if cell=='3' else 378011)
                if cell=='4':assert c['eps']==1e-6 and c['relation_hidden']==32 and c['tokens']==4 and c['dim']==64
        assert configs['1']['initial_model_sha256']==configs['3']['initial_original_model_sha256']==configs['4']['initial_original_model_sha256']
        assert configs['3']['data_sha256']==configs['4']['data_sha256']
        for name,hash_value in configs['4']['data_sha256'].items():
            assert sha(Path('/home/zhangzj26/TGRS_MLUDA-2024/datasets/Houston')/name)==hash_value
        for epoch in range(200):
            for key in ('batch_stream_sha256','original_rng_stream_sha256'):
                assert len({histories[k][epoch][key] for k in cells})==1,(seed,epoch+1,key)
        # Cell 3 exactly reproduced a second historical Joint control whose full
        # source/target index hashes match cell 1. Verify that bridge explicitly.
        bridge=ROOT/f'results/sceneshift_joint_v1/C_{seed}'
        bc=read(bridge/'config.json')
        for key in ('source_indices_sha256','target_indices_sha256'):
            assert bc[key]==configs['1'][key]
        bh=[json.loads(s) for s in (bridge/'history.jsonl').read_text().splitlines()]
        assert all(x['source_ce']==y['source_ce'] for x,y in zip(bh,histories['3']))
        lhs=torch.load(cells['3']/'checkpoints/epoch_200.pth',map_location='cpu',weights_only=True)['model']
        rhs=torch.load(bridge/'checkpoints/epoch_200.pth',map_location='cpu',weights_only=True)['model']
        assert set(lhs)==set(rhs) and all(torch.equal(lhs[k],rhs[k]) for k in lhs)
        evidence[str(seed)]={cell:{'folder':str(path),'config_sha256':sha(path/'config.json'),
                          'history_sha256':sha(path/'history.jsonl'),'result_sha256':sha(path/'results.json'),
                          'manifest_sha256':sha(path/'frozen_manifest.json')} for cell,path in cells.items()}
    return {'eligible_for_reuse':True,'seeds':list(SEEDS),'all_200_epoch_batch_rng_streams_match':True,
            'historical_shared_initialization_and_config_match':True,'source_target_index_bridge_exact':True,
            'evidence':evidence,'new_training_tasks_authorized':3}


if __name__=='__main__':
    result=run()
    OUT.mkdir(parents=True,exist_ok=True)
    path=OUT/'reuse_audit.json'
    if path.exists():raise RuntimeError('Refuse overwrite')
    path.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))

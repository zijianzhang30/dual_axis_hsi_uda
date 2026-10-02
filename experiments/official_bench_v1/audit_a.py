"""Read-only reproduction checks and inference on already-frozen A checkpoints.

Writes only new audit artifacts. Never trains, alters a checkpoint, or selects
a formal endpoint. Oracle is diagnostic for the existing local trajectories.
"""
import ast
import difflib
import hashlib
import json
import subprocess
import types
import urllib.request
import numpy as np
import torch
import common as c

AUDIT=c.OUT/'audit_A_v1'
URL='https://raw.githubusercontent.com/YuxiangZhang-BIT/IEEE_TCSVT_BiDA/'


def write_json(path,value):
    path.write_text(json.dumps(value,indent=2)+'\n')


def sampling_simulation(seed,local):
    g=torch.Generator().manual_seed(seed)
    loaders=[torch.utils.data.DataLoader(torch.arange(n),batch_size=128,shuffle=i in (0,2),
                 generator=g,num_workers=0) for i,n in enumerate((19021,1002,10287,10287))]
    hashes=[]
    for epoch in range(1,3):
        h=hashlib.sha256()
        for source,target in zip(loaders[0],loaders[2]):
            if len(source)!=len(target):continue
            h.update(source.numpy().tobytes());h.update(target.numpy().tobytes())
        hashes.append(h.hexdigest())
        if local and (epoch==1 or epoch%10==0):list(loaders[1])
        if not local and epoch%10==0:list(loaders[3])
    return hashes


def main():
    assert not (AUDIT/'frozen_predictions.json').exists() and not (AUDIT/'protocol_checks.json').exists(), 'Refuse to overwrite completed audit'
    assert not (AUDIT/'predictions').exists() or not list((AUDIT/'predictions').iterdir()), 'Refuse to overwrite predictions'
    AUDIT.mkdir(exist_ok=True);(AUDIT/'upstream').mkdir(exist_ok=True);(AUDIT/'predictions').mkdir(exist_ok=True)
    torch.set_num_threads(2);device=torch.device('cuda:6');torch.cuda.set_device(device)
    commit=subprocess.check_output(['git','ls-remote','https://github.com/YuxiangZhang-BIT/IEEE_TCSVT_BiDA.git','refs/heads/main'],text=True).split()[0]
    comparison={};remote_dataset=None
    files=('main.py','train_pipeline.py','models/BiDA.py','models/get_model.py',
           'utils/dataset.py','utils/scheduler.py','utils/utils_HSI.py','loss/make_loss.py','loss/mmd_loss.py')
    for name in files:
        raw=urllib.request.urlopen(URL+commit+'/'+name,timeout=30).read()
        remote=raw.decode();local=(c.BIDA/name).read_text()
        p=AUDIT/'upstream'/name;p.parent.mkdir(parents=True,exist_ok=True)
        if p.exists():assert p.read_bytes()==raw
        else:p.write_bytes(raw)
        same=ast.dump(ast.parse(remote))==ast.dump(ast.parse(local))
        comparison[name]={'semantic_AST_equal':same,'upstream_sha256':hashlib.sha256(raw).hexdigest(),
                          'local_sha256':c.helper.file_hash(c.BIDA/name)}
        if not same:
            patch='\n'.join(difflib.unified_diff(remote.splitlines(),local.splitlines(),
                          fromfile='upstream/'+name,tofile='local/'+name,lineterm=''))+'\n'
            (AUDIT/(name.replace('/','_')+'.patch')).write_text(patch)
        if name=='utils/dataset.py':
            remote_dataset=types.ModuleType('upstream_dataset')
            exec(compile(remote,str(p),'exec'),remote_dataset.__dict__)
    numerical={};dataset_names=('Dioni','Loukia')
    for name in dataset_names:
        x,y,classes=c.load_mat_hsi(name,str(c.DATA)+'/',norm='normband')
        rx,ry,rc=remote_dataset.load_mat_hsi(name,str(c.DATA)+'/',norm='normband')
        assert np.array_equal(x,rx) and np.array_equal(y,ry) and classes==rc
        numerical[name]={'full_image_max_abs_error':float(np.abs(x-rx).max()),
                         'gt_and_class_mapping_exact':True,'raw_class_counts':np.bincount(y[y>=0],minlength=12).tolist()}
    c.seed_worker(2100);loaders,data=c.loaders(c.options())
    # Check native dataset indices and same original patch at first/middle/last centers.
    # Released constructor references self.data without assigning it. Document
    # the pre-existing local runtime repair; add only that assignment IN MEMORY
    # so the otherwise native boundary/patch logic can be compared numerically.
    source=(AUDIT/'upstream/utils/dataset.py').read_text()
    assert 'self.data = image' not in source and source.count('        self.label = gt')==1
    fixed=source.replace('        self.label = gt','        self.data = image\n        self.label = gt')
    diagnostic_dataset=types.ModuleType('upstream_dataset_runtime_repair')
    exec(compile(fixed,'upstream_dataset_runtime_repair','exec'),diagnostic_dataset.__dict__)
    patches={}
    for i,loader in enumerate(loaders):
        ds=loader.dataset
        rds=diagnostic_dataset.HSIDataset(ds.data,ds.label,13,data_aug=False)
        assert np.array_equal(ds.indices,rds.indices)
        ds.data_aug=False
        errors=[float((ds[k][0]-rds[k][0]).abs().max()) for k in (0,len(ds)//2,len(ds)-1)]
        assert max(errors)==0
        patches[str(i)]={'indices_exact':True,'sampled_patch_max_abs_error':max(errors)}
    sampling={str(seed):{'official':sampling_simulation(seed,False),'local':sampling_simulation(seed,True)}
              for seed in (2100,2101,2102)}
    assert all(r['official'][0]==r['local'][0] and r['official'][1]!=r['local'][1] for r in sampling.values())
    checks={'upstream_commit':commit,'upstream_comparison':comparison,'normalization':numerical,
            'upstream_constructor_runtime_defect':'self.data referenced without assignment; pre-existing local self.data=image repair',
            'dataset_patch_comparison_caveat':'upstream constructor tested with that single assignment added in memory; no training file modified',
            'dataset_patch_comparison':patches,'shared_loader_generator_cadence_probe':sampling,
            'data_protocol':data,'history':{}}
    frozen=[]
    for seed in (2100,2101,2102):
        folder=c.OUT/f'hyrank_A_{seed}'
        manifest=json.loads((folder/'frozen_manifest.json').read_text())
        for path,h in manifest['config']['code_sha256'].items():
            assert c.helper.file_hash(path)==h, f'Frozen training code changed: {path}'
        history=[json.loads(s) for s in (folder/'history.jsonl').read_text().splitlines()]
        assert len(history)==200 and all(r['steps']==80 for r in history)
        assert history[-1]['global_step']==16000
        checks['history'][str(seed)]={'epochs':200,'steps_each_epoch':80,'total_optimizer_steps':16000,
                                     'MMD_eligible_steps_epochs_101_200':8000}
        c.seed_worker(seed)
        model=c.get_model('BiDA','Dioni',13,c.options(seed)).to(device).eval()
        for record in manifest['checkpoints']:
            assert c.helper.file_hash(record['path'])==record['sha256']
            checkpoint=torch.load(record['path'],map_location='cpu',weights_only=False)
            assert checkpoint['epoch']==record['epoch']
            model.load_state_dict(checkpoint['model'])
            before=c.helper.state_hash(model)
            logits,pred=c.helper.predict(model,loaders[-1],device)
            assert c.helper.state_hash(model)==before, 'Inference mutated buffers'
            dest=AUDIT/'predictions'/f'A_{seed}_epoch_{record["epoch"]:03d}.npz'
            np.savez_compressed(dest,logits=logits,predictions=pred)
            frozen.append({'seed':seed,'epoch':record['epoch'],'path':str(dest),
                           'sha256':c.helper.file_hash(dest),'checkpoint_sha256':record['sha256']})
            if record['epoch']==200:
                with np.load(manifest['prediction']['path']) as old:
                    assert np.array_equal(pred,old['predictions'])
                    checks['history'][str(seed)]['epoch200_reinference_max_logit_error']=float(np.abs(logits-old['logits']).max())
            print(json.dumps({'seed':seed,'epoch':record['epoch'],'prediction_frozen':True}),flush=True)
        del model
    write_json(AUDIT/'protocol_checks.json',checks)
    write_json(AUDIT/'frozen_predictions.json',frozen)
    # All 60 predictions are frozen before label values enter target metrics.
    labels=np.asarray(loaders[-1].dataset.labels)
    rows=[]
    for record in frozen:
        assert c.helper.file_hash(record['path'])==record['sha256']
        with np.load(record['path']) as prediction:
            metrics=c.helper.metrics(labels,prediction['predictions'],prediction['logits'],12)
        rows.append({'seed':record['seed'],'epoch':record['epoch'],**metrics})
    best=[max((r for r in rows if r['seed']==seed),key=lambda r:r['oa_percent']) for seed in (2100,2101,2102)]
    final=[r for r in rows if r['epoch']==200]
    aggregate=lambda rr:{field:{'mean':float(np.mean([r[field] for r in rr])),
                               'sample_std':float(np.std([r[field] for r in rr],ddof=1))}
                         for field in ('oa_percent','aa_percent','kappa')}
    paper_recall=[34.15,80.19,63.62,40.55,69.20,36.02,81.22,57.52,50.38,50.68,100.,100.]
    result={'primary':'fixed epoch 200 unchanged','oracle_definition':'diagnostic only, local existing checkpoints 10:10:200',
            'trajectory':rows,'oracle_per_seed':best,'fixed200':aggregate(final),'oracle':aggregate(best),
            'paper':{'url':'https://arxiv.org/html/2507.02268v1#S3.T14','OA':68.89,'kappa':.627,
                     'AA_from_rounded_recall':float(np.mean(paper_recall)),
                     'seed_aggregation':'not established','checkpoint_rule':'not established from paper'}}
    write_json(AUDIT/'trajectory_summary.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('trajectory','oracle_per_seed')},indent=2),flush=True)


if __name__=='__main__':main()

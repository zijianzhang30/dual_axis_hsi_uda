"""Score only frozen epoch200 endpoints; CE controls remain immutable."""
import csv
import json
import numpy as np
import torch
import common as c

def main(dataset):
    records={};reuse=json.loads((c.OUT/'reused_CE_controls.json').read_text())
    for seed in c.SEEDS:
      for arm in ('A','C'):
        p=c.OUT/dataset/('%s_LMMD_%d'%(arm,seed));m=json.loads((p/'frozen_manifest.json').read_text())
        assert m['checkpoints'][-1]['epoch']==200 and m['no_target_scoring']
        assert c.sha(m['prediction']['path'])==m['prediction']['sha256']
        records[arm,seed]=m
    # Every new endpoint in this benchmark is frozen before reading evaluation label values.
    m,worker,helper=c.environment(dataset);worker(2100)
    if dataset=='houston':
        from argparse import Namespace
        loaders=m.loaders(Namespace(dataset=dataset,seed=2100,num_workers=0,dataset_dir='/home/zhangzj26/TGRS_MLUDA-2024/datasets/Houston/'))
        classes=7
    else:
        args=m.options(2100);args.num_workers=0;loaders,audit=m.loaders(args);classes=12
    truth=np.asarray(loaders[-1].dataset.labels);indices=loaders[-1].dataset.indices
    masks={'native':np.ones(len(truth),dtype=bool)}
    truths={'native':truth}
    if dataset=='hyrank':
        clean=c.clean_dataset(loaders[-1].dataset,keep_labels=True)
        truths['clean_original_scene']=np.asarray(clean.labels)
        masks['clean_original_scene']=np.ones(len(clean),dtype=bool)
    rows=[]
    for seed in c.SEEDS:
     for arm in ('A','C'):
      for objective in ('CE','LMMD'):
        if objective=='CE':info=reuse['%s_%s_%d'%(dataset,arm,seed)];path=info['prediction']['path'];cfg=info
        else:info=records[arm,seed];path=info['prediction']['path'];cfg=info['config']
        z=np.load(path);pred=z['predictions'];logits=z['logits'];assert len(pred)==len(truth)
        if objective=='LMMD':assert np.array_equal(np.load(c.OUT/dataset/('%s_LMMD_%d/evaluation_indices.npy'%(arm,seed))),indices)
        for name,mask in masks.items():
            pp,ll=pred,logits
            if name=='clean_original_scene':
                if objective=='LMMD':
                    item=info['clean_prediction'];assert c.sha(item['path'])==item['sha256']
                    zz=np.load(item['path']);pp,ll=zz['predictions'],zz['logits']
                    assert np.array_equal(np.load(c.OUT/dataset/('%s_LMMD_%d/clean_evaluation_indices.npy'%(arm,seed))),clean.indices)
                else:
                    # Read-only endpoint evaluation of existing CE control, with separately saved clean predictions.
                    base_model,base_loaders,hh,cc,aa=c.setup(dataset,arm,seed)
                    manifest=json.loads((c.baseline(dataset,arm,seed)/'frozen_manifest.json').read_text())
                    checkpoint=manifest['checkpoints'][-1]['path']
                    saved=torch.load(checkpoint,map_location='cpu',weights_only=False);assert saved['epoch']==200
                    base_model.load_state_dict(saved['model'],strict=True);base_model=base_model.cuda().eval()
                    before=c.state_hash(base_model)
                    cd=c.clean_dataset(base_loaders[-1].dataset)
                    loader=torch.utils.data.DataLoader(cd,batch_size=128,shuffle=False,num_workers=0,generator=torch.Generator().manual_seed(seed+85000))
                    ll,pp=helper.predict(base_model,loader,torch.device('cuda:0'));assert before==c.state_hash(base_model)
                    np.savez_compressed(c.OUT/dataset/('reused_%s_CE_%d_clean_predictions.npz'%(arm,seed)),logits=ll,predictions=pp)
                    assert np.array_equal(cd.indices,clean.indices)
                    del base_model,base_loaders
            metrics=helper.metrics(truths[name][mask],pp[mask],ll[mask],classes)
            zero=[k+1 for k,v in enumerate(metrics['per_class_percent']) if v==0]
            rows.append({'dataset':dataset,'eval_protocol':name,'arm':arm,'objective':objective,'seed':seed,**metrics,
                'zero_recall_classes':zero,'collapse':bool(zero or max(metrics['prediction_shares'])>=.95),
                'parameters':cfg['parameters'],'train_seconds':info['train_seconds'],
                'source_val_OA':info['source_val_at_final']['oa_percent']})
    aggregate=[];paired=[]
    for protocol in masks:
     for arm in ('A','C'):
      for objective in ('CE','LMMD'):
        rr=[r for r in rows if (r['eval_protocol'],r['arm'],r['objective'])==(protocol,arm,objective)]
        aggregate.append({'eval_protocol':protocol,'arm':arm,'objective':objective,
            **{key:{'mean':float(np.mean([r[key] for r in rr])),'sample_std':float(np.std([r[key] for r in rr],ddof=1))} for key in ['oa_percent','aa_percent','kappa','train_seconds']}})
     for seed in c.SEEDS:
      get=lambda arm,obj:next(r for r in rows if (r['eval_protocol'],r['seed'],r['arm'],r['objective'])==(protocol,seed,arm,obj))
      for label,left,right in [('C_LMMD-A_LMMD',get('C','LMMD'),get('A','LMMD')),('C_LMMD-C_CE',get('C','LMMD'),get('C','CE')),('A_LMMD-A_CE',get('A','LMMD'),get('A','CE'))]:
        paired.append({'eval_protocol':protocol,'comparison':label,'seed':seed,**{key:left[key]-right[key] for key in ['oa_percent','aa_percent','kappa']}})
    output=c.OUT/dataset
    c.dump(output/'summary.json',{'rows':rows,'aggregate':aggregate,'paired':paired,'formal':'fixed_epoch200','no_followups':True})
    for name,data in [('paired_deltas.csv',paired),('metrics.csv',[{k:r[k] for k in ['dataset','eval_protocol','arm','objective','seed','oa_percent','aa_percent','kappa','collapse','zero_recall_classes','parameters','train_seconds','source_val_OA']} for r in rows]),
                      ('per_class_recall.csv',[{'eval_protocol':r['eval_protocol'],'arm':r['arm'],'objective':r['objective'],'seed':r['seed'],'class':i+1,'recall_percent':rec} for r in rows for i,rec in enumerate(r['per_class_percent'])])]:
        with (output/name).open('w') as f:w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
    lines=['%s: fixed epoch200, seeds2100/2101/2102; no target-best.'%dataset,
           'A = original tokenizer + Full Joint BN + self Transformer; NOT complete released BiDA.',
           'Native/clean protocols are saved separately. LMMD hyperparameters frozen before training.',json.dumps(aggregate,indent=2),json.dumps(paired,indent=2)]
    (output/'RESULTS.txt').write_text('\n'.join(lines)+'\n')
    print('Scored all frozen '+dataset+' endpoints; no follow-up experiments launched.',flush=True)

if __name__=='__main__':
    import sys
    main(sys.argv[1])

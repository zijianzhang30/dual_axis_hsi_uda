"""Score only fixed endpoints after all six predictions pass the pairing barrier."""
import csv
import json
import numpy as np
import common as c


def stats(values):
    a=np.asarray(values,dtype=float)
    return {'values':a.tolist(),'mean':float(a.mean()),'sample_std':float(a.std(ddof=1))}


def csvfile(name, rows):
    with (c.OUT/name).open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def main():
    assert not (c.OUT/'summary.json').exists(),'Refuse overwrite'
    records={};audit={};runs={}
    for seed in c.SEEDS:
        histories={}
        for arm in c.ARMS:
            folder=c.OUT/('%s_%d'%(arm,seed));m=json.loads((folder/'frozen_manifest.json').read_text())
            assert m['epoch']==100 and c.sha(m['prediction']['path'])==m['prediction']['sha256']
            assert c.sha(m['checkpoints'][-1]['path'])==m['checkpoints'][-1]['sha256']
            h=[json.loads(line) for line in (folder/'history.jsonl').read_text().splitlines()]
            assert len(h)==100 and all(r['steps']==38 for r in h)
            histories[arm]=h;records[arm,seed]=m
        for key in ('initial_full_model_sha256','initial_scorer_sha256','source_selection_sha256','plan_sha256','parameters'):
            assert records['P1',seed]['config'][key]==records['P2',seed]['config'][key],key
        for key in ('order_and_augmentation_sha256','model_rng_sha256'):
            assert [r[key] for r in histories['P1']]==[r[key] for r in histories['P2']],(seed,key)
        old=c.P0/'pavia'/('C_%d'%seed)
        m=json.loads((old/'frozen_manifest.json').read_text())
        assert c.sha(m['prediction']['path'])==m['prediction']['sha256']
        assert m['checkpoints'][-1]['epoch']==200
        assert c.sha(m['checkpoints'][-1]['path'])==m['checkpoints'][-1]['sha256']
        records['P0',seed]=m
        audit[str(seed)]={'P1_P2_initialization_source_samples_order_augmentation_model_rng':'exactly matched all100 epochs',
                          'P0_formal_epoch200_reuse':'hash verified',
                          'P1_P0_initial_full_model_match':m['config']['initial_full_model_sha256']==records['P1',seed]['config']['initial_full_model_sha256']}
    c.dump(c.OUT/'all_predictions_frozen.json',{'passed':True,'audit':audit})
    # Label values enter metrics only after all endpoint predictions have been frozen and verified.
    labels=np.load(c.OUT/'inputs/evaluation_labels_OFFLINE_ONLY.npy')
    helper=c.environment()[3]
    perclass=[];seedrows=[]
    for (arm,seed),m in records.items():
        with np.load(m['prediction']['path']) as p:metric=helper.metrics(labels,p['predictions'],p['logits'],7)
        metric['collapse']=max(metric['prediction_shares'])>=.95
        metric['zero_recall_classes']=[i+1 for i,x in enumerate(metric['per_class_percent']) if x==0]
        row={'arm':arm,'seed':seed,'epoch':200 if arm=='P0' else 100,
             'OA':metric['oa_percent'],'AA':metric['aa_percent'],'Kappa':metric['kappa'],
             'collapse':metric['collapse'],'zero_recall_classes':';'.join(map(str,metric['zero_recall_classes'])),
             'parameters':m['config']['parameters'],'train_seconds':m['train_seconds']}
        seedrows.append(row);runs['%s_%d'%(arm,seed)]={'metrics':metric,**row}
        for k,v in enumerate(metric['per_class_percent']):
            perclass.append({'arm':arm,'seed':seed,'class':k+1,'recall_percent':v,
                             'prediction_share':metric['prediction_shares'][k],
                             'target_count':sum(metric['confusion_matrix'][k])})
    aggregate={}
    for arm in ('P0','P1','P2'):
        rr=[r for r in seedrows if r['arm']==arm]
        aggregate[arm]={k:stats([r[k] for r in rr]) for k in ('OA','AA','Kappa','train_seconds')}
    paired={};deltas=[]
    for later,earlier in (('P1','P0'),('P2','P1')):
        name=later+'-'+earlier
        paired[name]={k:stats([runs['%s_%d'%(later,s)][k]-runs['%s_%d'%(earlier,s)][k] for s in c.SEEDS]) for k in ('OA','AA','Kappa')}
        for i,s in enumerate(c.SEEDS):
            deltas.append({'comparison':name,'seed':s,**{k:paired[name][k]['values'][i] for k in ('OA','AA','Kappa')}})
    result={'formal_endpoints':{'P0':200,'P1':100,'P2':100},'seeds':c.SEEDS,
            'aggregate':aggregate,'paired_deltas':paired,'runs':runs,'audit':audit,
            'no_target_best':True,'no_auto_followups':True,
            'interpretation':'P1-P0 bundled training/data protocol change; P2-P1 official ILDA input adaptation, not our innovation'}
    c.dump(c.OUT/'summary.json',result)
    csvfile('per_seed.csv',seedrows);csvfile('paired_deltas.csv',deltas);csvfile('per_class_recall.csv',perclass)
    lines=['Pavia frozen-C protocol transfer: fixed predeclared endpoints, no target-best.',
           'P0 reused repaired-normalized C; P1 official-style training/data knobs; P2 adds official ILDA.',
           'All P1/P2 initialization, source selection, data order, augmentation and model RNG paired.',
           'OA/AA are percentages; paired differences in percentage points. SD is sample SD.', '']
    for arm,rr in aggregate.items():
        lines.append(arm+': '+', '.join('%s %.4f +/- %.4f'%(k,rr[k]['mean'],rr[k]['sample_std']) for k in ('OA','AA','Kappa')))
    lines += ['', 'Paired deltas:']
    for name,rr in paired.items():
        for k,v in rr.items():lines.append('%s %s mean%+.4f +/- %.4f; seeds=%s'%(name,k,v['mean'],v['sample_std'],v['values']))
    lines += ['', 'No automatic changes to loss, hyperparameters, architecture or additional experiments.',
              'Official MLUDA optimizer resets momentum each epoch; 38 updates/epoch; source constant-zero padding.',
              'C single-view augmentation retained; no additional MLUDA loss/view branches.',
              'Target candidate mask provenance inherited from P0; target class values excluded from training.',
              'Formal evaluation uses unchanged P0 pixels/order, running BN stats and target,target pairing.']
    (c.OUT/'RESULTS.txt').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'aggregate':aggregate,'paired_deltas':paired},indent=2),flush=True)


if __name__=='__main__':main()

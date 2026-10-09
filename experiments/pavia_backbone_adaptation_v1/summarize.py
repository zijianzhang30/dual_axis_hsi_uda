"""Frozen-endpoint barrier, two-by-two effects and historical references separate."""
import csv
import json
import numpy as np
import common as c


def stats(values):
    a=np.asarray(values,dtype=float)
    return {'values':a.tolist(),'mean':float(a.mean()),'sample_std':float(a.std(ddof=1)),
            'positive_seeds':int((a>0).sum())}


def write_csv(name,rows):
    with (c.OUT/name).open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def main():
    assert not (c.OUT/'summary.json').exists(),'Refuse overwrite'
    records={};checks={};histories={}
    for seed in c.SEEDS:
        for arm in c.ARMS:
            folder=c.OUT/('%s_%d'%(arm,seed));m=json.loads((folder/'frozen_manifest.json').read_text())
            assert m['epoch']==100 and c.sha(m['prediction']['path'])==m['prediction']['sha256']
            assert m['checkpoints'][-1]['epoch']==100
            assert c.sha(m['checkpoints'][-1]['path'])==m['checkpoints'][-1]['sha256']
            assert c.sha(folder/'last_source_reference.npy')==m['source_reference_sha256']
            h=[json.loads(x) for x in (folder/'history.jsonl').read_text().splitlines()]
            assert len(h)==100 and all(x['steps']==38 for x in h)
            records[arm,seed]=m;histories[arm,seed]=h
        first=histories['M_CE',seed]
        for arm in c.ARMS[1:]:
            for key in ('plan_sha256','all_input_views_sha256'):
                assert [r[key] for r in first]==[r[key] for r in histories[arm,seed]],(arm,seed,key)
        for network in ('M','O'):
            a,b=network+'_CE',network+'_AD'
            for key in ('initial_base_model_sha256','initial_complete_model_sha256','parameters'):
                assert records[a,seed]['config'][key]==records[b,seed]['config'][key]
            for key in ('model_rng_sha256','BN_num_batches'):
                assert [r[key] for r in histories[a,seed]]==[r[key] for r in histories[b,seed]],(seed,network,key)
            assert records[a,seed]['source_reference_sha256']==records[b,seed]['source_reference_sha256']
        checks[str(seed)]={'all_four_data_views_and_order':'identical for100 epochs',
                          'within_each_network_init_model_rng_BN_and_source_reference':'identical',
                          'fixed_epoch100_checkpoint_and_prediction_hashes':'verified'}
    c.dump(c.OUT/'all_predictions_frozen.json',{'passed':True,'checks':checks})
    labels=np.load(c.P2/'inputs/evaluation_labels_OFFLINE_ONLY.npy')
    metric=c.base().environment()[3].metrics
    rows=[];perclass=[];runs={}
    for (arm,seed),m in records.items():
        with np.load(m['prediction']['path']) as p:mm=metric(labels,p['predictions'],p['logits'],7)
        mm['collapse']=max(mm['prediction_shares'])>=.95
        mm['zero_recall_classes']=[i+1 for i,x in enumerate(mm['per_class_percent']) if x==0]
        row={'arm':arm,'seed':seed,'OA':mm['oa_percent'],'AA':mm['aa_percent'],'Kappa':mm['kappa'],
             'collapse':mm['collapse'],'zero_recall_classes':';'.join(map(str,mm['zero_recall_classes'])),
             'parameters':m['config']['parameters'],'train_seconds':m['training_seconds']}
        rows.append(row);runs['%s_%d'%(arm,seed)]={**row,'metrics':mm}
        for k,v in enumerate(mm['per_class_percent']):
            perclass.append({'arm':arm,'seed':seed,'class':k+1,'recall_percent':v,'prediction_share':mm['prediction_shares'][k]})
    aggregate={a:{k:stats([runs['%s_%d'%(a,s)][k] for s in c.SEEDS]) for k in ('OA','AA','Kappa','train_seconds')} for a in c.ARMS}
    paired={};delta_rows=[]
    for a,b in (('M_AD','M_CE'),('O_AD','O_CE'),('M_CE','O_CE'),('M_AD','O_AD')):
        name=a+'-'+b
        paired[name]={k:stats([runs['%s_%d'%(a,s)][k]-runs['%s_%d'%(b,s)][k] for s in c.SEEDS]) for k in ('OA','AA','Kappa')}
        for i,seed in enumerate(c.SEEDS):delta_rows.append({'comparison':name,'seed':seed,**{k:paired[name][k]['values'][i] for k in ('OA','AA','Kappa')}})
    interaction={k:stats(np.asarray(paired['M_AD-M_CE'][k]['values'])-np.asarray(paired['O_AD-O_CE'][k]['values'])) for k in ('OA','AA','Kappa')}
    old_p2=json.loads((c.P2/'summary.json').read_text())
    native=json.loads((c.OUT.parent/'mluda_pavia_native_v1/results.json').read_text())
    historical={'Ours_P2_single_view':old_p2['aggregate']['P2'],
                'native_MLUDA_different_10_seeds_native_mask_denominator_and_views':native['summary'],
                'not_in_paired_table':True}
    result={'endpoint':100,'seeds':c.SEEDS,'aggregate':aggregate,'paired':paired,
            'objective_gain_interaction_M_minus_O':interaction,'runs':runs,'pairing_audit':checks,
            'historical_references_separate':historical,'no_target_best':True,'no_auto_followups':True,
            'limitations':['Native flip reverses batch/channel axes without permuting labels; unchanged in all four cells.',
                           'M and O retain different BN and source-pair/readout semantics; row effect is a whole network package.',
                           'AD is a full official objective package with training projection heads; no single-loss attribution.',
                           'Main views retain P2 geometry; common math flags, pixel cohort and eval tails differ from historical native MLUDA.',
                           'Target candidate-mask provenance inherited from historical nonzero GT mask; no target values in optimization.']}
    c.dump(c.OUT/'summary.json',result)
    write_csv('per_seed.csv',rows);write_csv('per_class_recall.csv',perclass);write_csv('paired_deltas.csv',delta_rows)
    lines=['Pavia same-protocol network x adaptation decomposition: fixed100,3 seeds, no target-best.',
           'M_CE/M_AD: original MLUDA network; O_CE/O_AD: frozen C plus matched training-only projection heads.',
           'CE controls execute all3 views, preserving within-row BN/RNG; gradients use CE only.',
           'Data,all3 input views and sampling exactly matched for100 epochs in all4 arms.', '',
           'Mean +/- sample SD:']
    for arm,a in aggregate.items():lines.append(arm+': '+', '.join('%s %.4f +/- %.4f'%(k,a[k]['mean'],a[k]['sample_std']) for k in ('OA','AA','Kappa')))
    lines+=['','Paired deltas (percentage points for OA/AA):']
    for name,rr in paired.items():
        for k,v in rr.items():lines.append('%s %s %+.4f +/- %.4f; seeds=%s;positive=%d/3'%(name,k,v['mean'],v['sample_std'],v['values'],v['positive_seeds']))
    lines+=['','Historical91% MLUDA and68.68% P2 are separate references; not substituted into these cells.',
            'No new architecture, loss design, tuning or followup experiments automatically launched.', '', 'Limitations:']+result['limitations']
    (c.OUT/'RESULTS.txt').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'aggregate':aggregate,'paired':paired},indent=2),flush=True)


if __name__=='__main__':main()

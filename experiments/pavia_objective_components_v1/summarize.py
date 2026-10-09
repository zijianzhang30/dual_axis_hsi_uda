"""Score only after all9 removal endpoints freeze; conditional effects not additive."""
import csv
import json
import numpy as np
import common as c

def stats(v):
    a=np.asarray(v,dtype=float)
    return {'values':a.tolist(),'mean':float(a.mean()),'sample_std':float(a.std(ddof=1)),
            'positive_seeds':int((a>0).sum())}

def csvout(name,rows):
    with (c.OUT/name).open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def main():
    assert not (c.OUT/'summary.json').exists(),'Refuse overwrite'
    c.verify_files();assert c.audit_reuse()==json.loads((c.OUT/'reused_controls.json').read_text())
    records={};histories={};checks={}
    for seed in c.SEEDS:
        for arm in c.ARMS:
            folder=c.run_folder(arm,seed);m=json.loads((folder/'frozen_manifest.json').read_text())
            assert m['epoch']==100 and m['no_target_scoring'] and m['checkpoints'][-1]['epoch']==100
            assert c.sha(m['prediction']['path'])==m['prediction']['sha256']
            assert c.sha(m['checkpoints'][-1]['path'])==m['checkpoints'][-1]['sha256']
            assert c.sha(folder/'last_source_reference.npy')==m['source_reference_sha256']
            hh=[json.loads(v) for v in (folder/'history.jsonl').read_text().splitlines()]
            assert len(hh)==100 and all(r['steps']==38 and not r['target_scoring'] for r in hh)
            records[arm,seed]=m;histories[arm,seed]=hh
        baseline=records['Full',seed];rh=histories['Full',seed]
        for arm in c.ARMS:
            mm=records[arm,seed]
            for key in ('initial_base_model_sha256','initial_complete_model_sha256','parameters'):
                assert mm['config'][key]==baseline['config'][key]
            assert mm['source_reference_sha256']==baseline['source_reference_sha256']
            for key in ('plan_sha256','all_input_views_sha256','model_rng_sha256','BN_num_batches'):
                assert [r[key] for r in histories[arm,seed]]==[r[key] for r in rh],(arm,seed,key)
        checks[str(seed)]={'all5_init_data_views_model_RNG_BN_source_reference':'identical',
                          'all100_epochs_paired':True,'all5_epoch100_predictions_and_checkpoint_hashes':'verified'}
    c.dump(c.OUT/'all_predictions_frozen.json',{'passed':True,'checks':checks})
    # First and only evaluation GT values read, after every formal endpoint is immutable.
    labels=np.load(c.P2/'inputs/evaluation_labels_OFFLINE_ONLY.npy')
    metric=c.base().environment()[3].metrics
    runs={};rows=[];recalls=[]
    for (arm,seed),m in records.items():
        with np.load(m['prediction']['path']) as p:mm=metric(labels,p['predictions'],p['logits'],7)
        zero=[i+1 for i,v in enumerate(mm['per_class_percent']) if v==0]
        row={'arm':arm,'seed':seed,'OA':mm['oa_percent'],'AA':mm['aa_percent'],'Kappa':mm['kappa'],
             'collapse':max(mm['prediction_shares'])>=.95,'zero_recall_classes':';'.join(map(str,zero)),
             'parameters':m['config']['parameters'],'train_seconds':m['training_seconds'],'reused':arm in ('CE','Full')}
        rows.append(row);runs['%s_%d'%(arm,seed)]={**row,'metrics':mm,'folder':str(c.run_folder(arm,seed))}
        for k,v in enumerate(mm['per_class_percent']):
            recalls.append({'arm':arm,'seed':seed,'class':k+1,'recall_percent':v,'prediction_share':mm['prediction_shares'][k]})
    aggregate={arm:{k:stats([runs['%s_%d'%(arm,s)][k] for s in c.SEEDS]) for k in ('OA','AA','Kappa','train_seconds')} for arm in c.ARMS}
    paired={};delta=[]
    pairs=[('Full','CE')]+[('Full',a) for a in c.NEW_ARMS]+[(a,'CE') for a in c.NEW_ARMS]
    for a,b in pairs:
        name=a+'-'+b;paired[name]={k:stats([runs['%s_%d'%(a,s)][k]-runs['%s_%d'%(b,s)][k] for s in c.SEEDS]) for k in ('OA','AA','Kappa')}
        for i,seed in enumerate(c.SEEDS):
            delta.append({'comparison':name,'seed':seed,**{k:paired[name][k]['values'][i] for k in ('OA','AA','Kappa')}})
    perclass_aggregate={a:np.mean([runs['%s_%d'%(a,s)]['metrics']['per_class_percent'] for s in c.SEEDS],axis=0).tolist() for a in c.ARMS}
    result={'endpoint':100,'seeds':c.SEEDS,'aggregate':aggregate,'paired':paired,'runs':runs,
            'per_class_recall_mean':perclass_aggregate,'pairing_audit':checks,
            'no_target_best':True,'no_target_tuning':True,'no_auto_followups':True,
            'interpretation':'Full minus removal = conditional necessity within existing bundle; effects do not sum to Full-CE',
            'limitations':['Source contrastive is source regularization, not target alignment.',
             'Native flip on B,C,H,W reverses batch/bands without permuting labels; deliberately preserved for attribution.',
             'Historical target candidate mask uses nonzero-GT provenance; no target class values used for optimization or selection.',
             'Matched heads/all3views mean CE differs from the old single-view P2; only matrix CE is used.',
             'Timing includes diagnostic evaluation of excluded raw terms and shared GPU contention; not loss FLOP benchmark.',
             'No single-loss-only groups; redundancy/interactions cannot be isolated by leave-one-out alone.']}
    c.dump(c.OUT/'summary.json',result)
    csvout('per_seed.csv',rows);csvout('per_class_recall.csv',recalls);csvout('paired_deltas.csv',delta)
    lines=['Pavia Ours objective decomposition; fixed100; seeds2100/2101/2102.',
           'CE/Full reuse exact previously frozen controls;9 removals trained;no target-best.',
           'Mean +/- sampleSD:']
    for a,r in aggregate.items():lines.append(a+': '+', '.join('%s %.4f +/- %.4f'%(k,r[k]['mean'],r[k]['sample_std']) for k in ('OA','AA','Kappa')))
    lines+=['','Conditional full-minus-removal effects (OA/AA percentage points):']
    for a in c.NEW_ARMS:
        name='Full-'+a
        for k in ('OA','AA','Kappa'):
            r=paired[name][k];lines.append('%s %s %+.4f +/- %.4f; seeds=%s; positive=%d/3'%(name,k,r['mean'],r['sample_std'],r['values'],r['positive_seeds']))
    lines+=['','Per-class mean recall:']+[a+': '+str(v) for a,v in perclass_aggregate.items()]
    lines+=['','Interpretation: '+result['interpretation'],'','Limitations:']+result['limitations']
    (c.OUT/'RESULTS.txt').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'aggregate':aggregate,'paired':paired},indent=2),flush=True)

if __name__=='__main__':main()

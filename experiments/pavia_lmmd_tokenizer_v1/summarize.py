"""Both stages frozen before scoring; same-loss C vs A and simpler-target comparison."""
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
    records={};checks={};histories={}
    pre=json.loads((c.OUT/'verification.json').read_text())
    for seed in c.SEEDS:
        for arm in c.ARMS:
            folder=c.run_folder(arm,seed);m=json.loads((folder/'frozen_manifest.json').read_text())
            assert m['epoch']==100 and m['no_target_scoring'] and m['checkpoints'][-1]['epoch']==100
            assert c.sha(m['prediction']['path'])==m['prediction']['sha256']
            assert c.sha(m['checkpoints'][-1]['path'])==m['checkpoints'][-1]['sha256']
            assert c.sha(folder/'last_source_reference.npy')==m['source_reference_sha256']
            h=[json.loads(v) for v in (folder/'history.jsonl').read_text().splitlines()]
            assert len(h)==100 and all(v['epoch']==i+1 and v['steps']==38 and not v['target_scoring'] for i,v in enumerate(h))
            records[arm,seed]=m;histories[arm,seed]=h
        ref=records['C_CE',seed];rh=histories['C_CE',seed]
        for arm in c.ARMS:
            m=records[arm,seed]
            for k in ('plan_sha256','all_input_views_sha256','model_rng_sha256','BN_num_batches'):
                assert [v[k] for v in histories[arm,seed]]==[v[k] for v in rh],(arm,seed,k)
            assert m['source_reference_sha256']==ref['source_reference_sha256']
            if arm!='A_LMMD':
                for k in ('initial_base_model_sha256','initial_complete_model_sha256','parameters'):
                    assert m['config'][k]==ref['config'][k]
            if arm in c.NEW_ARMS:
                check=next(v for v in pre['checks'] if v['arm']==arm and v['seed']==seed)
                assert m['config']['initial_shared_model_sha256']==check['shared_initial_sha256']
                assert m['config']['initial_complete_model_sha256']==check['complete_initial_sha256']
        assert records['C_LMMD',seed]['config']['initial_shared_model_sha256']==records['A_LMMD',seed]['config']['initial_shared_model_sha256']
        checks[str(seed)]={'all4_order_views_model_RNG_BN_source_reference':'identical100epochs',
            'C_initial_states':'identical across objectives',
            'A_vs_C_non_tokenizer_initial_states':'identical','fixed100_hashes':'verified'}
    c.dump(c.OUT/'all_predictions_frozen.json',{'passed':True,'checks':checks})
    # Formal target labels first loaded after both complete phases and all endpoint audits.
    labels=np.load(c.P2/'inputs/evaluation_labels_OFFLINE_ONLY.npy');metric=c.base().environment()[3].metrics
    rows=[];recalls=[];runs={};conv=[]
    for (arm,seed),m in records.items():
        with np.load(m['prediction']['path']) as p:mm=metric(labels,p['predictions'],p['logits'],7)
        zero=[i+1 for i,v in enumerate(mm['per_class_percent']) if v==0]
        row={'arm':arm,'seed':seed,'OA':mm['oa_percent'],'AA':mm['aa_percent'],'Kappa':mm['kappa'],
            'collapse':max(mm['prediction_shares'])>=.95,'zero_recall_classes':';'.join(map(str,zero)),
            'parameters':m['config']['parameters'],'train_seconds':m['training_seconds'],'reused':arm not in c.NEW_ARMS}
        rows.append(row);runs['%s_%d'%(arm,seed)]={**row,'metrics':mm,'folder':str(c.run_folder(arm,seed))}
        for k,v in enumerate(mm['per_class_percent']):
            recalls.append({'arm':arm,'seed':seed,'class':k+1,'recall_percent':v,'prediction_share':mm['prediction_shares'][k]})
        last=histories[arm,seed][-1]
        conv.append({'arm':arm,'seed':seed,'source_train_OA':last['source_train_OA'],
            'source_CE':last['source_CE'],'LMMD':last['LMMD']})
    aggregate={a:{k:stats([runs['%s_%d'%(a,s)][k] for s in c.SEEDS]) for k in ('OA','AA','Kappa','train_seconds')} for a in c.ARMS}
    pairs=(('C_LMMD','C_CE'),('C_LMMD_CT','C_LMMD'),('C_LMMD','A_LMMD'),('C_LMMD_CT','C_CE'))
    paired={};delta=[];classdelta=[]
    support=np.bincount(labels,minlength=7);freq=support/support.sum()
    for a,b in pairs:
        name=a+'-'+b;paired[name]={k:stats([runs['%s_%d'%(a,s)][k]-runs['%s_%d'%(b,s)][k] for s in c.SEEDS]) for k in ('OA','AA','Kappa')}
        for i,seed in enumerate(c.SEEDS):
            delta.append({'comparison':name,'seed':seed,**{k:paired[name][k]['values'][i] for k in ('OA','AA','Kappa')}})
            rr=np.array(runs['%s_%d'%(a,seed)]['metrics']['per_class_percent'])-np.array(runs['%s_%d'%(b,seed)]['metrics']['per_class_percent'])
            assert abs(float((freq*rr).sum())-paired[name]['OA']['values'][i])<1e-8
            for k,v in enumerate(rr):classdelta.append({'comparison':name,'seed':seed,'class':k+1,
                'target_fraction':float(freq[k]),'recall_delta_pp':float(v),'OA_contribution_pp':float(freq[k]*v)})
    classmeans={a:np.mean([runs['%s_%d'%(a,s)]['metrics']['per_class_percent'] for s in c.SEEDS],axis=0).tolist() for a in c.ARMS}
    result={'endpoint':100,'seeds':c.SEEDS,'aggregate':aggregate,'paired':paired,'runs':runs,
        'per_class_recall_mean':classmeans,'source_convergence':conv,'pairing_audit':checks,
        'LMMD_weight':'0.3*same logistic ramp, not newly weighted',
        'no_target_best':True,'no_target_tuning':True,'no_auto_followups':True,
        'limitations':['LMMD is an existing objective,not an innovation.',
            'A/C parameter difference3172 is only tokenizer; this is not a capacity-matched scorer mechanism comparison.',
            'All3views retained for BN/RNG matching; unused contrastive heads do not train or decay.',
            'Native flip axes and historical target candidate-mask provenance preserved.',
            'C_LMMD_CT vs C_LMMD tests target contrastive with no source contrastive; differs from its conditional effect inside Full.',
            'Shared GPU load and retained3views limit timing interpretation.']}
    c.dump(c.OUT/'summary.json',result)
    csvout('per_seed.csv',rows);csvout('per_class_recall.csv',recalls);csvout('paired_deltas.csv',delta)
    csvout('conditional_class_effects.csv',classdelta);csvout('source_convergence.csv',conv)
    lines=['Pavia LMMD-only and tokenizer control;fixed100;seeds2100/2101/2102.',
           'C3runs first,A3runs second;no target-based branching;all endpoints frozen before scoring.',
           'Mean +/- sampleSD:']
    for a,r in aggregate.items():lines.append(a+': '+', '.join('%s %.4f +/- %.4f'%(k,r[k]['mean'],r[k]['sample_std']) for k in ('OA','AA','Kappa')))
    lines+=['','Paired differences,OA/AA percentage points:']
    for name,rr in paired.items():
        for k,r in rr.items():lines.append('%s %s %+.4f +/- %.4f; per-seed=%s;positive=%d/3'%(name,k,r['mean'],r['sample_std'],r['values'],r['positive_seeds']))
    lines+=['','Per-class mean recall:']+[a+': '+str(v) for a,v in classmeans.items()]
    lines+=['','Limitations:']+result['limitations']
    (c.OUT/'RESULTS.txt').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'aggregate':aggregate,'paired':paired},indent=2),flush=True)

if __name__=='__main__':main()

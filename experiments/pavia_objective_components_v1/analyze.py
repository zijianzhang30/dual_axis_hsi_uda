"""Post-barrier conditional effects by class and source convergence; no followup jobs."""
import csv
import json
import math
import numpy as np
import common as c

def main():
    assert json.loads((c.OUT/'launcher_status.json').read_text())['state']=='completed_no_followups'
    assert json.loads((c.OUT/'all_predictions_frozen.json').read_text())['passed']
    s=json.loads((c.OUT/'summary.json').read_text());runs=s['runs']
    # Use the confusion support, already scored offline, rather than loading GT again.
    cm=np.array(runs['Full_2100']['metrics']['confusion_matrix'])
    support=cm.sum(1);p=support/support.sum();rows=[];convergence=[]
    for seed in c.SEEDS:
        for arm in c.ARMS:
            folder=c.run_folder(arm,seed)
            h=[json.loads(v) for v in (folder/'history.jsonl').read_text().splitlines()]
            tail=h[-10:];r=h[-1]
            convergence.append({'arm':arm,'seed':seed,'epoch100_source_train_OA':r['source_train_OA'],
                'epoch100_source_CE':r['source_CE'],
                'last10_source_train_OA_mean':float(np.mean([v['source_train_OA'] for v in tail])),
                'last10_source_CE_mean':float(np.mean([v['source_CE'] for v in tail])),
                'raw_LMMD':r['LMMD'],'raw_source_contrastive':r['Con_s'],'raw_target_contrastive':r['Con_t'],
                'optimized_terms':c.flags(arm)})
        full=runs['Full_%d'%seed]
        for arm in c.NEW_ARMS:
            other=runs['%s_%d'%(arm,seed)]
            diff=np.array(full['metrics']['per_class_percent'])-np.array(other['metrics']['per_class_percent'])
            contrib=p*diff
            assert abs(contrib.sum()-(full['OA']-other['OA']))<1e-8
            assert abs(diff.mean()-(full['AA']-other['AA']))<1e-8
            for k in range(7):
                rows.append({'comparison':'Full-'+arm,'seed':seed,'class':k+1,
                    'support':int(support[k]),'target_fraction':float(p[k]),
                    'Full_recall':full['metrics']['per_class_percent'][k],
                    'removed_recall':other['metrics']['per_class_percent'][k],
                    'recall_delta_pp':float(diff[k]),'OA_contribution_pp':float(contrib[k])})
    for name,data in (('conditional_class_effects.csv',rows),('source_convergence.csv',convergence)):
        with (c.OUT/name).open('x',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
    ranking=sorted(c.NEW_ARMS,key=lambda a:s['paired']['Full-'+a]['OA']['mean'],reverse=True)
    result={'conditional_OA_removal_drop_ranking':ranking,
        'target_class_support':support.tolist(),'target_class_fraction':p.tolist(),
        'source_convergence':convergence,'conditional_effects_sum_to_OA_verified':True,
        'collapse_runs':[k for k,v in runs.items() if v['collapse']],
        'zero_recall_runs':{k:v['zero_recall_classes'] for k,v in runs.items() if v['zero_recall_classes']},
        'analysis_only_no_training_or_tuning':True,'analysis_script_sha256':c.sha(__file__)}
    c.dump(c.OUT/'conditional_analysis.json',result)
    lines=['Ours Pavia objective leave-one-out: frozen100, three paired seeds.',
        'CE/Full reused from the strict matrix; removal runs share initialization,data,augmentation,RNG,BN traces.',
        'Source train accuracy is reported as a convergence check; it is not source validation.',
        '', 'Full minus removal (positive = removal hurts):']
    for arm in ranking:
        r=s['paired']['Full-'+arm]
        lines.append(arm+': '+', '.join('%s %+.4f +/- %.4f; per-seed=%s; positive=%d/3'%(k,r[k]['mean'],r[k]['sample_std'],r[k]['values'],r[k]['positive_seeds']) for k in ('OA','AA')))
    lines+=['','No automatic loss redesign, parameter sweep, single-loss-only groups or followup benchmark.',
        'Effects measure conditional necessity inside Full; they are not additive components of the14.69pp gain.',
        'Native flip batch/band-axis issue remains a limitation on mechanism interpretation.',
        'Raw removed loss values are diagnostic only; excluded heads have grad=None and remain unmodified by SGD.']
    (c.OUT/'FINAL_ANALYSIS.txt').write_text('\n'.join(lines)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='source_convergence'},indent=2))

if __name__=='__main__':main()

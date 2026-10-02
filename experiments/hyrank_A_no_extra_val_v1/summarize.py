"""Compare only the three prescribed new A runs to immutable old A results."""
import csv
import json
import numpy as np
from control import c,OUT,SEEDS,dump


def aggregate(rows):
    return {k:{'mean':float(np.mean([r[k] for r in rows])),
               'sample_std':float(np.std([r[k] for r in rows],ddof=1))}
            for k in ('oa_percent','aa_percent','kappa')}


def main():
    old=json.loads((c.OUT/'audit_A_v1/trajectory_summary.json').read_text())
    fixed=[];best=[];paired={'fixed200':[],'oracle':[]};checks=[];manifests=[]
    for seed in SEEDS:
        folder=OUT/f'seed_{seed}'
        manifest=json.loads((folder/'frozen_manifest.json').read_text())
        assert manifest['completed_epochs']==200 and manifest['total_steps']==16000
        evaluations=manifest['evaluations']
        assert [r['epoch'] for r in evaluations]==list(range(10,201,10))
        for r in evaluations:
            assert c.helper.file_hash(r['prediction_path'])==r['prediction_sha256']
            assert c.helper.file_hash(r['checkpoint_path'])==r['checkpoint_sha256']
        h=[json.loads(s) for s in (folder/'history.jsonl').read_text().splitlines()]
        oh=[json.loads(s) for s in (c.OUT/f'hyrank_A_{seed}'/'history.jsonl').read_text().splitlines()]
        assert len(h)==200 and all(r['steps']==80 and r['source_validation_calls']==0 for r in h)
        assert all(r['first_training_forward']['student_all_modules_train'] for r in h)
        assert h[0]['batch_stream_sha256']==oh[0]['batch_stream_sha256']
        assert h[0]['rng_stream_sha256']==oh[0]['rng_stream_sha256']
        checks.append({'seed':seed,'epoch1_old_new_batch_and_torch_RNG_equal':True,
                       'first_batch_stream_difference_epoch':next((i+1 for i in range(200)
                                      if h[i]['batch_stream_sha256']!=oh[i]['batch_stream_sha256']),None),
                       'new_epoch1_student_train_end':h[0]['student_train_at_epoch_end'],
                       'old_epoch1_student_train_end':False,
                       'all_next_epoch_first_forwards_train':True,
                       'no_state_mutation_between_epoch_end_and_next_first_forward':True})
        f={'seed':seed,'epoch':200,**evaluations[-1]['metrics']}
        b=max(evaluations,key=lambda r:r['metrics']['oa_percent'])
        b={'seed':seed,'epoch':b['epoch'],**b['metrics']}
        fixed.append(f);best.append(b);manifests.append(manifest)
        oldf=next(r for r in old['trajectory'] if r['seed']==seed and r['epoch']==200)
        oldb=next(r for r in old['oracle_per_seed'] if r['seed']==seed)
        for name,new,previous in (('fixed200',f,oldf),('oracle',b,oldb)):
            paired[name].append({'seed':seed,'new_epoch':new['epoch'],'old_epoch':previous['epoch'],
                                 **{k:new[k]-previous[k] for k in ('oa_percent','aa_percent','kappa')}})
    summary={'arm':'A_no_extra_val','interpretation':'workflow control, not literally pure RNG',
             'primary':'fixed200; does not replace historical comparison table',
             'oracle_definition':'max OA on epochs10:10:200; AA/Kappa from the same OA-best checkpoint; diagnostic only',
             'fixed200_per_seed':fixed,'oracle_per_seed':best,
             'fixed200':aggregate(fixed),'oracle':aggregate(best),'paired_new_minus_old_A':paired,
             'paired_aggregate':{name:aggregate(rows) for name,rows in paired.items()},
             'runtime_checks':checks,
             'train_seconds':[{'seed':seed,'seconds':m['train_seconds']} for seed,m in zip(SEEDS,manifests)],
             'configuration_limit':'Native every10 target validation replaces old every10 source validation, as explicitly requested; no added evaluator or RNG consumption simulation.'}
    dump(OUT/'summary.json',summary)
    with (OUT/'RESULTS.csv').open('x',newline='') as stream:
        writer=csv.writer(stream);writer.writerow(['endpoint','seed','new_epoch','old_epoch','new_OA','new_AA','new_Kappa','delta_OA_pp','delta_AA_pp','delta_Kappa'])
        for name,rows in (('fixed200',fixed),('oracle',best)):
            for new,delta in zip(rows,paired[name]):
                writer.writerow([name,new['seed'],new['epoch'],delta['old_epoch'],new['oa_percent'],new['aa_percent'],new['kappa'],delta['oa_percent'],delta['aa_percent'],delta['kappa']])
    print(json.dumps({k:summary[k] for k in ('fixed200','oracle','paired_aggregate')},indent=2),flush=True)


if __name__=='__main__':main()

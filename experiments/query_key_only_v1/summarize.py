"""Frozen paired on/off/key-only comparison and the user's practical gate."""
import importlib.util
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('pe_aggregate',ROOT/'experiments/query_pe_off_multiseed/summarize.py')
base = importlib.util.module_from_spec(spec); spec.loader.exec_module(base)

def main():
    rows = []; metric_sets = {k:[] for k in ('key_only','pe_on','pe_off','bida_reference')}
    for seed in (2100,2101,2102):
        folders = {'key_only':ROOT/f'results/query_key_only_v1/formal_{seed}',
                   'pe_on':ROOT/f'results/query_class_readout_v1/formal_{seed}',
                   'pe_off':ROOT/('results/query_pe_off_v1/formal_2101' if seed == 2101 else f'results/query_pe_off_multiseed/formal_{seed}'),
                   'bida_reference':ROOT/f'results/bn_stabilization_v1/fixed_mixture_{seed}'}
        results = {k:base.read_result(p,verify=k!='bida_reference') for k,p in folders.items()}
        cfg = json.loads((folders['key_only']/'config.json').read_text())
        assert cfg['spatial_pe']=='key_only'
        for control in ('pe_on','pe_off'):
            other = json.loads((folders[control]/'config.json').read_text())
            for k in ('initial_bida_sha256','initial_full_transplant_sha256','initial_pipeline_sha256',
                      'trainable_parameters','depth','dropout','bn_momentum_per_step',
                      'pipeline_script_sha256','joint_script_sha256','a1_script_sha256'):
                assert cfg[k]==other[k],(seed,control,k)
        m = results['key_only']['target_at_final']; trajectory=results['key_only']['target_trajectory']
        row={'seed':seed,'fixed200':{k:r['target_at_final'] for k,r in results.items()},
             'oracle':results['key_only']['target_oracle'],
             'manifest_sha256':results['key_only']['frozen_manifest_sha256'],
             'zero_recall_classes':[i+1 for i,v in enumerate(m['per_class_percent']) if v==0],
             'grid_class6_collapses':sum(r['metrics']['prediction_shares'][5]>=.95 for r in trajectory),
             'grid_min_oa':min(r['metrics']['oa_percent'] for r in trajectory),
             'grid_zero_recall_epochs':[r['epoch'] for r in trajectory if any(v==0 for v in r['metrics']['per_class_percent'])],
             'key_only_minus':{name:{k:m[k]-results[name]['target_at_final'][k] for k in ('oa_percent','aa_percent','kappa')} for name in ('pe_on','pe_off')}}
        rows.append(row)
        for name,r in results.items():metric_sets[name].append(r['target_at_final'])
    summary={name:base.aggregate(v) for name,v in metric_sets.items()}
    summary.update({'seeds':[2100,2101,2102],'primary':'fixed_epoch_200','sd':'sample ddof=1',
                    'rows':rows,'verified_on_off_key_checkpoint_prediction_files':360})
    key=summary['key_only'];ref=summary['bida_reference']
    gate={'mean_oa_at_least_reference':key['oa_percent']['mean']>=ref['oa_percent']['mean'],
          'mean_aa_at_least_reference':key['aa_percent']['mean']>=ref['aa_percent']['mean'],
          'zero_of_three_class6_collapse':key['class6_collapsed_seeds']==0}
    summary['practical_gate']={**gate,'all_conditions_met':all(gate.values())}
    summary['oracle_oa_diagnostic']={'mean':float(np.mean([r['oracle']['metrics']['oa_percent'] for r in rows])),
                                    'sample_sd':float(np.std([r['oracle']['metrics']['oa_percent'] for r in rows],ddof=1))}
    out=ROOT/'results/query_key_only_v1/summary.json'
    if out.exists():raise RuntimeError(f'Refuse to overwrite {out}')
    out.write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k!='rows'},indent=2))

if __name__=='__main__':main()

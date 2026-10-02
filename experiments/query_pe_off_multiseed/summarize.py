"""Verify frozen artifacts and aggregate the preregistered three seeds."""
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]

def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def read_result(folder, verify=False):
    result = json.loads((folder / 'results.json').read_text())
    if verify:
        manifest_path = folder / 'frozen_manifest.json'
        assert digest(manifest_path) == result['frozen_manifest_sha256']
        manifest = json.loads(manifest_path.read_text())
        expected_epochs = list(range(10, 201, 10))
        for group in ('checkpoints', 'predictions'):
            assert [r['epoch'] for r in manifest[group]] == expected_epochs
            for artifact in manifest[group]:
                assert digest(artifact['path']) == artifact['sha256'], artifact['path']
        assert [r['epoch'] for r in result['target_trajectory']] == expected_epochs
        assert result['target_at_final'] == result['target_trajectory'][-1]['metrics']
    return result

def aggregate(metrics):
    out = {}
    for key in ('oa_percent', 'aa_percent', 'kappa'):
        v = np.asarray([m[key] for m in metrics])
        out[key] = {'mean': float(v.mean()), 'sample_sd': float(v.std(ddof=1)),
                    'minimum': float(v.min()), 'maximum': float(v.max())}
    for key in ('per_class_percent', 'prediction_shares'):
        v = np.asarray([m[key] for m in metrics])
        out[key] = {'mean': v.mean(0).tolist(), 'sample_sd': v.std(0, ddof=1).tolist()}
    out['class6_collapsed_seeds'] = sum(m['prediction_shares'][5] >= .95 for m in metrics)
    return out

def main():
    rows = []; on_metrics = []; off_metrics = []; reference_metrics = []
    for seed in (2100, 2101, 2102):
        off_folder = ROOT / ('results/query_pe_off_v1/formal_2101' if seed == 2101
                             else f'results/query_pe_off_multiseed/formal_{seed}')
        on_folder = ROOT / f'results/query_class_readout_v1/formal_{seed}'
        off = read_result(off_folder, True); on = read_result(on_folder, True)
        reference = read_result(ROOT / f'results/bn_stabilization_v1/fixed_mixture_{seed}')
        off_config = json.loads((off_folder / 'config.json').read_text())
        on_config = json.loads((on_folder / 'config.json').read_text())
        for key in ('initial_bida_sha256', 'initial_full_transplant_sha256',
                    'initial_pipeline_sha256', 'trainable_parameters', 'depth',
                    'dropout', 'bn_momentum_per_step', 'pipeline_script_sha256',
                    'joint_script_sha256', 'a1_script_sha256'):
            assert off_config[key] == on_config[key], (seed, key)
        a = off['target_at_final']; b = on['target_at_final']
        trajectory = off['target_trajectory']
        rows.append({'seed': seed, 'off_folder': str(off_folder),
                     'off_frozen_manifest_sha256': off['frozen_manifest_sha256'],
                     'pe_off_fixed200': a, 'pe_on_fixed200': b,
                     'off_minus_on': {k: a[k]-b[k] for k in ('oa_percent','aa_percent','kappa')},
                     'pe_off_oracle': off['target_oracle'],
                     'fixed200_zero_recall_classes': [i+1 for i,r in enumerate(a['per_class_percent']) if r == 0],
                     'grid_class6_collapses': sum(r['metrics']['prediction_shares'][5] >= .95 for r in trajectory),
                     'grid_min_oa': min(r['metrics']['oa_percent'] for r in trajectory),
                     'grid_max_prediction_share': max(max(r['metrics']['prediction_shares']) for r in trajectory),
                     'grid_zero_recall_epochs': [r['epoch'] for r in trajectory if any(v == 0 for v in r['metrics']['per_class_percent'])]})
        off_metrics.append(a); on_metrics.append(b); reference_metrics.append(reference['target_at_final'])
    summary = {'seeds': [2100,2101,2102], 'primary': 'fixed_epoch_200', 'sd': 'sample ddof=1',
               'verified_pe_on_off_checkpoint_prediction_files': 240,
               'rows': rows, 'pe_off': aggregate(off_metrics), 'pe_on': aggregate(on_metrics),
               'bida_tokenizer_depth3_reference': aggregate(reference_metrics),
               'paired_off_minus_on': {k: {'mean': float(np.mean([r['off_minus_on'][k] for r in rows])),
                                         'sample_sd': float(np.std([r['off_minus_on'][k] for r in rows],ddof=1))}
                                      for k in ('oa_percent','aa_percent','kappa')},
               'oracle_oa_diagnostic': {'mean': float(np.mean([r['pe_off_oracle']['metrics']['oa_percent'] for r in rows])),
                                        'sample_sd': float(np.std([r['pe_off_oracle']['metrics']['oa_percent'] for r in rows],ddof=1))}}
    out = ROOT / 'results/query_pe_off_multiseed/summary.json'
    if out.exists(): raise RuntimeError(f'Refuse to overwrite {out}')
    out.write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary, indent=2))

if __name__ == '__main__': main()

"""Execute the untouched native entry point; collect artifacts only after return."""
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import random
import runpy
import sys
import time

import numpy as np
import torch

OUT = Path('/nas1/zhangzj26/dual_axis_hsi_uda/results/mluda_pavia_native_v1')


def dump(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def main():
    snapshot = OUT / 'native_snapshot'
    manifest = json.loads((OUT / 'manifest.json').read_text())
    for name, expected in manifest['source_sha256'].items():
        assert hashlib.sha256((snapshot / name).read_bytes()).hexdigest() == expected
    os.chdir(str(snapshot))
    sys.path.insert(0, str(snapshot))
    versions = {}
    for name in ('torch', 'numpy', 'scipy', 'scikit-learn', 'scikit-image',
                 'opencv-contrib-python', 'pandas', 'hdf5storage', 'matplotlib'):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    dump(OUT / 'environment.json', {'python': sys.version, 'platform': platform.platform(),
         'packages': versions, 'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
         'torch_cuda': torch.version.cuda, 'gpu': torch.cuda.get_device_name(0),
         'OMP_NUM_THREADS': os.environ.get('OMP_NUM_THREADS'),
         'MKL_NUM_THREADS': os.environ.get('MKL_NUM_THREADS')})
    torch.save({'python': random.getstate(), 'numpy': np.random.get_state(),
                'torch_cpu': torch.get_rng_state(), 'torch_cuda': torch.cuda.get_rng_state_all()},
               OUT / 'initial_rng.pt')
    start = time.monotonic()
    state = runpy.run_path(str(snapshot / 'MLUDA_up.py'), run_name='__main__')
    elapsed = time.monotonic() - start
    seeds = state['seeds']
    oa = np.asarray(state['acc']).reshape(-1)
    recall = np.asarray(state['A']) * 100
    aa = recall.mean(1)
    kappa = np.asarray(state['k']).reshape(-1)
    assert len(seeds) == len(oa) == 10
    rows = [{'seed': int(seed), 'epoch': 100, 'oa_percent_native_denominator': float(oa[i]),
             'aa_percent': float(aa[i]), 'kappa': float(kappa[i]),
             'per_class_recall_percent': recall[i].tolist()} for i, seed in enumerate(seeds)]
    result = {'runs': rows, 'total_wall_seconds': elapsed,
              'updates_per_epoch': 38, 'native_target_dataset_n': len(state['test_loader'].dataset),
              'native_scored_n_last_run': len(state['labels']),
              'parameter_count': sum(p.numel() for p in state['feature_encoder'].parameters()),
              'summary': {name: {'mean': float(values.mean()),
                          'population_std': float(values.std(ddof=0)),
                          'sample_std': float(values.std(ddof=1))}
                          for name, values in (('oa_percent', oa), ('aa_percent', aa), ('kappa', kappa))}}
    np.savez_compressed(OUT / 'native_metric_arrays.npz', seeds=seeds, oa=oa, aa=aa,
                        per_class_recall=recall, kappa=kappa)
    np.savez_compressed(OUT / 'last_seed_predictions.npz', predictions=state['predict'],
                        labels=state['labels'], seed=seeds[-1])
    torch.save(state['feature_encoder'].state_dict(), OUT / 'last_seed_epoch100.pth')
    dump(OUT / 'results.json', result)
    lines = ['# Native MLUDA Pavia reproduction', '',
             'Unchanged native entry point, all ten author seeds, final epoch 100.', '',
             '| Seed | Native OA (%) | AA (%) | Kappa |', '| --- | ---: | ---: | ---: |']
    lines += ['| {seed} | {oa_percent_native_denominator:.3f} | {aa_percent:.3f} | {kappa:.5f} |'.format(**row)
              for row in rows]
    lines += ['', '## Summary', '', json.dumps(result['summary'], indent=2), '',
              'Both population and sample standard deviations are reported. Native OA retains',
              'the released full-dataset denominator despite drop_last evaluation.',
              'Paper Pavia OA: 91.26 +/- 0.53%; AA: 92.70 +/- 0.80%.',
              'Not directly comparable to BiDA-style results with different protocols.',
              'See manifest.json, environment.json and native.log for provenance.']
    (OUT / 'RESULTS.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()

"""Create new NAS snapshot, recover native pre-ILDA RNG, freeze shared cubes."""
import os
import random
import shutil
import sys

import numpy as np
import torch

from common import ROOT, OLD, OUT, SEEDS, dump, sha, tensor_hash


def main():
    OUT.mkdir(exist_ok=False)
    for p in Path(__file__).parent.iterdir():
        if p.is_file():
            shutil.copy2(p, OUT / p.name)
    snapshot = OUT / 'native_snapshot'
    snapshot.mkdir()
    for p in (OLD / 'native_snapshot').iterdir():
        if p.is_file():
            shutil.copy2(p, snapshot / p.name)
    (snapshot / 'datasets').mkdir()
    (snapshot / 'datasets/Pavia').symlink_to((OLD / 'native_snapshot/datasets/Pavia').resolve(), target_is_directory=True)
    (ROOT / 'results/mluda_mechanism_pavia_v1').symlink_to(OUT, target_is_directory=True)
    shutil.copy2(OLD / 'initial_rng.pt', OUT / 'historical_initial_rng.pt')
    shutil.copy2(OLD / 'environment.json', OUT / 'baseline_environment.json')
    shutil.copy2(OLD / 'results.json', OUT / 'historical_results.json')
    os.chdir(snapshot)
    sys.path.insert(0, str(snapshot))
    import utils
    from UtilsCMS import ILDA
    import config_UP2PC as cfg
    assert tuple(cfg.seeds[:3]) == SEEDS
    s, ls = utils.load_data_pavia('./datasets/Pavia/paviaU.mat', './datasets/Pavia/paviaU_gt_7.mat')
    t, lt = utils.load_data_pavia('./datasets/Pavia/pavia.mat', './datasets/Pavia/pavia_gt_7.mat')
    rng = torch.load(OUT / 'historical_initial_rng.pt', map_location='cpu', weights_only=False)
    # Native imports/load_data do not draw randomness. PCA consumes NumPy RNG.
    random.setstate(rng['python'])
    np.random.set_state(rng['numpy'])
    torch.set_rng_state(rng['torch_cpu'])
    s, t = ILDA(s, t, cfg.pca_n, cfg.radius)
    cache = OUT / 'preprocessed'
    cache.mkdir()
    for name, array in [('source', s), ('target', t), ('source_labels', ls), ('target_labels', lt)]:
        np.save(cache / (name + '.npy'), array)
    dump(OUT / 'manifest.json', {
        'seeds': SEEDS, 'snapshot_sha256': {p.name: sha(p) for p in snapshot.glob('*.py')},
        'study_sha256': {p.name: sha(p) for p in OUT.glob('*.py')},
        'protocol_sha256': sha(OUT / 'PROTOCOL.md'),
        'data_sha256': {p.name: sha(p) for p in (snapshot / 'datasets/Pavia').glob('*.mat')},
        'preprocessing_sha256': {p.name: sha(p) for p in cache.glob('*.npy')},
        'cube_tensor_sha256': {'source': tensor_hash(s), 'target': tensor_hash(t)},
        'historical_rng_sha256': sha(OUT / 'historical_initial_rng.pt'),
        'historical_reuse': False, 'reason': 'No first-three full checkpoints/batch streams saved; rebuilt paired M0.',
    })
    print('Prepared', OUT, flush=True)


if __name__ == '__main__':
    from pathlib import Path
    main()

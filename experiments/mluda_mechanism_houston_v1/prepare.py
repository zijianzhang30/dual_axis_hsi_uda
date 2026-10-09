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
    (snapshot / 'datasets/Houston').symlink_to(Path('/nas1/zhangzj26/TGRS_MLUDA-2024/datasets/Houston'), target_is_directory=True)
    (ROOT / 'results/mluda_mechanism_houston_v1').symlink_to(OUT, target_is_directory=True)
    shutil.copy2(OLD / 'environment.json', OUT / 'baseline_environment.json')
    os.chdir(snapshot)
    sys.path.insert(0, str(snapshot))
    import utils
    from UtilsCMS import ILDA
    import config_Houston as cfg
    assert tuple(cfg.seeds[:3]) == SEEDS
    s, ls = utils.load_data_houston('./datasets/Houston/Houston13.mat', './datasets/Houston/Houston13_7gt.mat')
    t, lt = utils.load_data_houston('./datasets/Houston/Houston18.mat', './datasets/Houston/Houston18_7gt.mat')
    # Native ILDA precedes model seeds; freeze a documented independent preprocessing RNG.
    random.seed(20261006); np.random.seed(20261006); torch.manual_seed(20261006)
    torch.save({'python':random.getstate(),'numpy':np.random.get_state(),'torch_cpu':torch.get_rng_state()},OUT/'preprocessing_initial_rng.pt')
    s, t = ILDA(s, t, cfg.pca_n, cfg.radius)
    cache = OUT / 'preprocessed'
    cache.mkdir()
    for name, array in [('source', s), ('target', t), ('source_labels', ls), ('target_labels', lt)]:
        np.save(cache / (name + '.npy'), array)
    dump(OUT / 'manifest.json', {
        'seeds': SEEDS, 'snapshot_sha256': {p.name: sha(p) for p in snapshot.glob('*.py')},
        'study_sha256': {p.name: sha(p) for p in OUT.glob('*.py')},
        'protocol_sha256': sha(OUT / 'PROTOCOL.md'),
        'data_sha256': {p.name: sha(p) for p in (snapshot / 'datasets/Houston').glob('*.mat')},
        'preprocessing_sha256': {p.name: sha(p) for p in cache.glob('*.npy')},
        'cube_tensor_sha256': {'source': tensor_hash(s), 'target': tensor_hash(t)},
        'preprocessing_rng_sha256': sha(OUT / 'preprocessing_initial_rng.pt'), 'preprocessing_seed':20261006,
        'historical_reuse': False, 'reason': 'No strict matching Houston checkpoint/preprocessing/stream bundle identified; rebuild paired M0.',
    })
    print('Prepared', OUT, flush=True)


if __name__ == '__main__':
    from pathlib import Path
    main()

"""Reuse the historical Pavia procedure with strictly band/class adaptations."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import types
import numpy as np
import torch
from scipy.io import loadmat

ROOT = Path(__file__).resolve().parents[2]
OUT = Path('/nas1/zhangzj26/dual_axis_hsi_uda/results/mluda_bida_extension_v1')
DATA = Path('/nas1/zhangzj26/TGRS_MLUDA-2024/datasets')


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


pavia = module('extension_historical_pavia', ROOT/'experiments/pavia_joint_v1/train.py')
helper = pavia.helper
joint = module('extension_joint', ROOT/'experiments/bn_stabilization_v1/normalization.py')


def dump(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def relation(bands):
    path = ROOT/'experiments/relation_tokenizer_v1/relation.py'
    text = path.read_text()
    for old, new in [('bands=48', 'bands=%d'%bands), ('(1, 48)', '(1, %d)'%bands),
                     ('B×1×48×H×W', 'B×1×%d×H×W'%bands)]:
        assert text.count(old) == 1
        text = text.replace(old, new)
    value = types.ModuleType('extension_relation_%d'%bands)
    exec(compile(text, str(path), 'exec'), value.__dict__)
    return value, text


def dimensions(task):
    return (102, 7) if task == 'pavia' else (198, 3)


def normband(image):
    image = image.astype(np.float32)
    bands = image.shape[-1]
    flat = image.reshape(-1, bands)
    index = np.where(flat.sum(axis=-1) != 0)
    values = flat[index]
    values = values / values.max()
    norm = np.sqrt((values ** 2).sum(1))[:, None].repeat(bands, axis=1)
    norm[norm == 0] = 1
    flat[index] = values / norm
    assert np.isfinite(image).all()
    return image


def scenes(task):
    if task == 'pavia':
        directory = DATA/'Pavia'
        return pavia.load_scene(directory, 'paviaU'), pavia.load_scene(directory, 'pavia')
    raw = loadmat(DATA/'Shanghai-Hangzhou/DataCube.mat', simplify_cells=True)
    assert list(raw['common_class']) == [1, 2, 3]
    assert list(raw['class_name']) == ['Water', 'Ground/Building', 'Plant']
    result = []
    for i, shape in ((1, (1600,230,198)), (2, (590,230,198))):
        image, labels = raw['DataCube%d'%i], raw['gt%d'%i]
        assert image.shape == shape and labels.shape == shape[:2]
        assert set(np.unique(labels)) == {1,2,3}
        result.append((normband(image), labels.astype(int)-1))
    return tuple(result)


class Unlabeled(torch.utils.data.Dataset):
    def __init__(self, original): self.original = original
    def __len__(self): return len(self.original)
    def __getitem__(self, i):
        images, labels = self.original[i]
        return images, torch.zeros_like(labels)
    def __getattr__(self, key): return getattr(self.original, key)


def loaders(task, seed, workers=4):
    (source, sg), (target, tg) = scenes(task)
    train_gt, val_gt = pavia.sample_gt(sg, .95, seed, mode='random')
    target_gt, _ = pavia.sample_gt(tg, 1, seed, mode='random')
    pi = lambda x: np.pad(x, ((6,6),(6,6),(0,0)), mode='reflect')
    pg = lambda x: np.pad(x, ((6,6),(6,6)), mode='reflect')
    source, target = pi(source), pi(target)
    sets = [pavia.HSIDataset(source, pg(train_gt), 13, data_aug=True),
            pavia.HSIDataset(source, pg(val_gt), 13, data_aug=False),
            pavia.HSIDataset(target, pg(target_gt), 13, data_aug=True),
            pavia.HSIDataset(target, pg(target_gt), 13, data_aug=False)]
    generator = torch.Generator().manual_seed(seed)
    result = [torch.utils.data.DataLoader(Unlabeled(ds) if i==2 else ds,
              batch_size=128, shuffle=i in (0,2), generator=generator,
              num_workers=workers, drop_last=False) for i, ds in enumerate(sets)]
    return result


def build(task, arm, seed):
    bands, classes = dimensions(task)
    model = pavia.BiDAnet(n_bands=bands, num_classes=classes, num_tokens=4, dim=64, depth=3)
    initial = helper.state_hash(model)
    unused = pavia.BiDAnet(n_bands=bands, num_classes=classes, num_tokens=4, dim=64, depth=3)
    del unused
    joint.install(model, 'fixed_mixture')
    if arm == 'C': relation(bands)[0].install(model, 'C', seed)
    return model, initial


def inputs_audit(task, values):
    directory = DATA/('Pavia' if task == 'pavia' else 'Shanghai-Hangzhou')
    return {'indices_sha256': {str(i): hashlib.sha256(x.dataset.indices.tobytes()).hexdigest()
                              for i,x in enumerate(values)},
            'sample_counts': [len(x.dataset) for x in values],
            'data_sha256': {p.name: helper.file_hash(p) for p in sorted(directory.glob('*.mat'))},
            'target_candidates': 'existing GT mask; label values replaced by zero for training',
            'source_split': '95/5 stratified, native random_state=23',
            'classes': ['Water','Ground/Building','Plant'] if task=='shanghai' else 'existing Pavia seven-class IDs',
            'norm': 'historical BiDA normband, spectral L2', 'patch':13}


def historical(task, seed, initial, values):
    if task != 'pavia': return None
    path = ROOT/'results/pavia_joint_v1'/('joint_%d'%seed)
    cfg = json.loads((path/'config.json').read_text())
    expected = {'script_sha256': ROOT/'experiments/pavia_joint_v1/train.py',
                'protocol_sha256': ROOT/'experiments/pavia_joint_v1/PROTOCOL.md',
                'normalization_sha256': ROOT/'experiments/bn_stabilization_v1/normalization.py',
                'model_code_sha256': pavia.BIDA/'models/BiDA.py',
                'dataset_code_sha256': pavia.BIDA/'utils/dataset.py'}
    for key, file in expected.items(): assert cfg[key]==helper.file_hash(file), key
    assert cfg['seed']==seed and cfg['epochs']==200 and cfg['arm']=='joint'
    assert cfg['initial_model_sha256']==initial
    assert cfg['data_sha256']==inputs_audit(task, values)['data_sha256']
    assert list(cfg['sample_counts'].values())==[len(x.dataset) for x in values]
    return [json.loads(line) for line in (path/'history.jsonl').read_text().splitlines()]

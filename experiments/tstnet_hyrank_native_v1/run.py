"""Run pinned TSTnet HyRANK, preserving native loops and RNG-consuming calls."""
import ast
import difflib
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import site
import subprocess
import sys
import time

# Reuse installed CUDA torch without changing the shared experiment environment.
site.addsitedir('/home/zhangzj26/TGRS_MLUDA-2024/.venv/lib/python3.8/site-packages')

ROOT = Path('/nas1/zhangzj26/dual_axis_hsi_uda/results/tstnet_hyrank_native_v1')
ORIGINAL = ROOT / 'official_snapshot'
RUN = ROOT / 'run_snapshot'
DATA = Path('/nas1/zhangzj26/dual_axis_hsi_uda/datasets/HyRANK_author_v1/HyRANK')


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def prepare():
    if (ROOT / 'config.json').exists():
        raise RuntimeError('Refusing to overwrite frozen config')
    if not RUN.exists():
        shutil.copytree(ORIGINAL, RUN, ignore=shutil.ignore_patterns('.git'))
    p = RUN / 'utils_HSI.py'
    before = (ORIGINAL / 'utils_HSI.py').read_text()
    after = before.replace('return h5py.File(dataset)',
                           "return io.loadmat(dataset) if not h5py.is_hdf5(dataset) else {k: np.asarray(v).T for k, v in h5py.File(dataset).items() if not k.startswith('#')}")
    after = after.replace('dtype=np.bool)', 'dtype=bool)')
    assert before != after
    p.write_text(after)
    (ROOT / 'compatibility.patch').write_text(''.join(difflib.unified_diff(before.splitlines(True), after.splitlines(True), fromfile='official/utils_HSI.py', tofile='run/utils_HSI.py')))
    config = dict(task='Dioni -> Loukia', model='official full TSTnet',
                  commit=(ORIGINAL / '.git/refs/heads/main').read_text().strip(),
                  patch_size=12, batch_size=100, source_ratio=0.05, re_ratio=10,
                  lr=0.001, lambda_1=1.0, lambda_2=0.1, epochs=500,
                  trials=10, seed=1233, reseed_between_trials=False,
                  evaluation_epochs=list(range(1, 501, 10)), selection='native target-best, reproduction only',
                  target_training='native labeled candidate mask; labels used for reporting, not loss',
                  evaluation_augmentation='native flip and radiation enabled, preserved',
                  optimizer='native SGD recreated each epoch, momentum .9, decay 1e-4',
                  paper=dict(oa=63.31, oa_std=3.05, kappa=0.5537, kappa_std=0.0382),
                  environment={x: importlib.metadata.version(x) for x in ['torch', 'torch-geometric', 'numpy', 'scipy', 'tensorboardX']},
                  environment_limitation='Modern torch 2.4.1 / PyG 2.6.1, not author torch 1.6 / PyG 1.6.3',
                  patches=['MAT v5/v7.3 compatibility with original array axes', 'np.bool -> bool'],
                  hashes={str(p.relative_to(ORIGINAL)): digest(p) for p in ORIGINAL.rglob('*') if p.is_file() and '.git' not in p.parts})
    import scipy.io
    import numpy as np
    for name in ['Dioni', 'Loukia']:
        cube = scipy.io.loadmat(str(DATA / (name + '.mat')))['ori_data']
        gt_path = DATA / (name + '_gt_out68.mat')
        gt = scipy.io.loadmat(str(gt_path))['map']
        assert cube.shape[:2] == gt.shape and cube.shape[-1] == 176
        assert np.array_equal(gt, scipy.io.loadmat(str(ORIGINAL / 'datasets/HyRANK' / gt_path.name))['map'])
        config[name] = dict(shape=list(cube.shape), classes=np.bincount(gt.ravel().astype(int), minlength=13).tolist(), gt_sha256=digest(gt_path), cube_sha256=digest(DATA / (name + '.mat')))
    (ROOT / 'config.json').write_text(json.dumps(config, indent=2))


def execute(preflight=False):
    os.chdir(RUN)
    sys.path.insert(0, str(RUN))
    sys.argv = ['train.py', '--data_path', str(DATA) + '/', '--source_name', 'Dioni', '--target_name', 'Loukia',
                '--lr', '0.001', '--lambda_1', '1', '--lambda_2', '0.1', '--re_ratio', '10',
                '--save_path', str(ROOT / 'checkpoints'), '--log_path', str(ROOT / 'logs')]
    for folder in ['checkpoints', 'logs', 'predictions']:
        (ROOT / folder).mkdir(exist_ok=True)
    import numpy as np
    import torch
    torch.set_num_threads(2)
    tree = ast.parse((RUN / 'train.py').read_text())
    namespace = {'__name__': '__main__', '__file__': str(RUN / 'train.py')}
    boundary = next(i for i, node in enumerate(tree.body) if isinstance(node, ast.If) and isinstance(node.test, ast.Compare) and ast.unparse(node.test).startswith('__name__')) if hasattr(ast, 'unparse') else next(i for i, node in enumerate(tree.body) if isinstance(node, ast.If))
    exec(compile(ast.Module(body=tree.body[:boundary], type_ignores=[]), str(RUN / 'train.py'), 'exec'), namespace)
    if preflight:
        from datasets import get_dataset
        metadata = {}
        for name in ['Dioni', 'Loukia']:
            image, gt, *_ = get_dataset(name, str(DATA) + '/')
            norms = np.linalg.norm(image, axis=-1)
            assert image.shape[-1] == 176 and np.isfinite(image).all()
            metadata[name] = {'shape': list(image.shape), 'unit_norm_max_error': float(abs(norms[norms > 0] - 1).max())}
        model = namespace['TSTnet'].Feature_Extractor(176, 12, 12).cuda().train()
        source = torch.randn(100, 176, 12, 12, device='cuda')
        target = torch.randn_like(source)
        labels = torch.arange(100, device='cuda') % 12
        out = model(source, target)
        loss = torch.nn.functional.cross_entropy(out[0], labels) + torch.nn.functional.cross_entropy(out[5], labels) + out[1] + .1 * (out[2] + out[3]) - (out[0].softmax(1) * out[5].log_softmax(1)).sum() / 100
        assert torch.isfinite(loss)
        loss.backward()
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        metadata.update(loss=float(loss), parameters=sum(p.numel() for p in model.parameters()), peak_memory_bytes=torch.cuda.max_memory_allocated())
        (ROOT / 'preflight.json').write_text(json.dumps(metadata, indent=2))
        print(json.dumps(metadata), flush=True)
        return
    original_train, original_test = namespace['train'], namespace['test']
    started = time.monotonic()
    def train(epoch, model, count):
        result = original_train(epoch, model, count)
        namespace['_evaluation_epoch'] = epoch + 1
        with (ROOT / 'progress.jsonl').open('a') as f:
            f.write(json.dumps(dict(trial=int(namespace['flag']), epoch=epoch + 1, elapsed_seconds=time.monotonic() - started)) + '\n')
        if epoch + 1 == count:
            torch.save(model.state_dict(), ROOT / 'checkpoints' / ('trial_%02d_final.pth' % namespace['flag']))
        return result
    def test(model):
        result = original_test(model)
        pred, labels = np.concatenate(result[2]), np.concatenate(result[3])
        cm = np.bincount(12 * labels.astype(int) + pred.astype(int), minlength=144).reshape(12, 12)
        recall = np.diag(cm) / cm.sum(1)
        shares = np.bincount(pred.astype(int), minlength=12) / len(pred)
        observed = np.trace(cm) / cm.sum()
        expected = (cm.sum(0) * cm.sum(1)).sum() / float(cm.sum() ** 2)
        record = dict(trial=int(namespace['flag']), epoch=int(namespace['_evaluation_epoch']), oa=float(observed * 100), aa=float(recall.mean() * 100), kappa=float((observed - expected) / (1 - expected)), recall=(recall * 100).tolist(), prediction_shares=shares.tolist(), confusion_matrix=cm.tolist(), single_class_collapse=bool(shares.max() >= .95), zero_recall_classes=(np.flatnonzero(recall == 0) + 1).tolist())
        assert abs(observed - result[1]) < 1e-10
        np.savez_compressed(ROOT / 'predictions' / ('trial_%02d_epoch_%03d.npz' % (record['trial'], record['epoch'])), pred=pred, labels=labels)
        with (ROOT / 'trajectory.jsonl').open('a') as f:
            f.write(json.dumps(record) + '\n')
        return result
    namespace.update(train=train, test=test)
    exec(compile(ast.Module(body=tree.body[boundary:], type_ignores=[]), str(RUN / 'train.py'), 'exec'), namespace)
    records = [json.loads(x) for x in (ROOT / 'trajectory.jsonl').read_text().splitlines()]
    best = [max([x for x in records if x['trial'] == trial], key=lambda x: x['oa']) for trial in range(10)]
    summary = dict(status='completed', trials=best, elapsed_seconds=time.monotonic() - started,
                   means={key: float(np.mean([r[key] for r in best])) for key in ['oa', 'aa', 'kappa']},
                   sample_std={key: float(np.std([r[key] for r in best], ddof=1)) for key in ['oa', 'aa', 'kappa']},
                   population_std={key: float(np.std([r[key] for r in best])) for key in ['oa', 'aa', 'kappa']})
    (ROOT / 'summary.json').write_text(json.dumps(summary, indent=2))


if __name__ == '__main__':
    if '--prepare' in sys.argv:
        prepare()
    else:
        execute('--preflight' in sys.argv)

"""Locked Pavia transfer test using the existing BiDA self and joint-BN code."""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.io import loadmat

ROOT = Path(__file__).resolve().parents[2]
BIDA = Path('/home/zhangzj26/IEEE_TCSVT_BiDA')
sys.path.insert(0, str(BIDA))
sys.path.insert(0, str(ROOT / 'experiments/bn_stabilization_v1'))
from models.BiDA import BiDAnet
from utils.dataset import HSIDataset, sample_gt
from utils.utils_HSI import seed_worker
from normalization import install
import importlib.util
spec = importlib.util.spec_from_file_location('helpers', ROOT / 'experiments/bida_self_dropout/train.py')
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)


def load_scene(directory, name):
    image = loadmat(directory / f'{name}.mat')[name].astype(np.float32)
    labels = loadmat(directory / f'{name}_gt_7.mat')[f'{name}_gt_7'].astype(int) - 1
    if image.shape[-1] != 102 or set(np.unique(labels)) != set(range(-1, 7)):
        raise ValueError('Unexpected Pavia shape or class mapping')
    flat = image.reshape(-1, 102)
    index = np.where(flat.sum(axis=-1) != 0)
    values = flat[index]
    values = values / values.max()
    norm = np.sqrt((values ** 2).sum(1))[:, None].repeat(102, axis=1)
    norm[norm == 0] = 1
    flat[index] = values / norm
    if not np.isfinite(image).all():
        raise ValueError('Nonfinite data')
    return image, labels


def make_loaders(args):
    source, source_gt = load_scene(args.dataset_dir, 'paviaU')
    target, target_gt = load_scene(args.dataset_dir, 'pavia')
    train_gt, val_gt = sample_gt(source_gt, .95, args.seed, mode='random')
    target_gt, _ = sample_gt(target_gt, 1, args.seed, mode='random')
    pad_image = lambda x: np.pad(x, ((6, 6), (6, 6), (0, 0)), mode='reflect')
    pad_gt = lambda x: np.pad(x, ((6, 6), (6, 6)), mode='reflect')
    source, target = pad_image(source), pad_image(target)
    sets = [HSIDataset(source, pad_gt(train_gt), 13, data_aug=True),
            HSIDataset(source, pad_gt(val_gt), 13, data_aug=False),
            HSIDataset(target, pad_gt(target_gt), 13, data_aug=True),
            HSIDataset(target, pad_gt(target_gt), 13, data_aug=False)]
    generator = torch.Generator().manual_seed(args.seed)
    return [torch.utils.data.DataLoader(dataset, batch_size=128,
            shuffle=i in (0, 2), generator=generator, num_workers=args.num_workers,
            drop_last=False) for i, dataset in enumerate(sets)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arm', choices=('original', 'joint'), required=True)
    parser.add_argument('--seed', type=int, choices=(2100, 2101, 2102), required=True)
    parser.add_argument('--device', type=int, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--dataset-dir', type=Path, default=Path('/home/zhangzj26/TGRS_MLUDA-2024/datasets/Pavia'))
    args = parser.parse_args()
    if args.out.exists() and any(args.out.iterdir()):
        parser.error('Output must be empty')
    torch.set_num_threads(2)
    device = torch.device(f'cuda:{args.device}')
    torch.cuda.set_device(device)
    seed_worker(args.seed)
    start = time.monotonic()
    train, val, target_train, target_eval = make_loaders(args)
    model = BiDAnet(n_bands=102, num_classes=7, num_tokens=4, dim=64, depth=3)
    initial_hash = helper.state_hash(model)
    unused = BiDAnet(n_bands=102, num_classes=7, num_tokens=4, dim=64, depth=3)
    del unused
    if args.arm == 'joint':
        install(model, 'fixed_mixture')
    model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=.01)
    args.out.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = args.out / 'checkpoints'
    checkpoints_dir.mkdir()
    config = {'arm': args.arm, 'seed': args.seed, 'device': str(device),
              'benchmark': 'PaviaU -> PaviaC (existing 7-class mapping)',
              'epochs': 200, 'primary': 'fixed_epoch_200', 'initial_model_sha256': initial_hash,
              'trainable_parameters': sum(p.numel() for p in model.parameters()),
              'sample_counts': dict(zip(('source_train', 'source_val', 'target_train', 'target_eval'),
                                       (len(x.dataset) for x in (train, val, target_train, target_eval)))),
              'protocol_sha256': helper.file_hash(Path(__file__).with_name('PROTOCOL.md')),
              'script_sha256': helper.file_hash(Path(__file__)),
              'normalization_sha256': helper.file_hash(ROOT / 'experiments/bn_stabilization_v1/normalization.py'),
              'model_code_sha256': helper.file_hash(BIDA / 'models/BiDA.py'),
              'dataset_code_sha256': helper.file_hash(BIDA / 'utils/dataset.py'),
              'data_sha256': {p.name: helper.file_hash(p) for p in sorted(args.dataset_dir.glob('*.mat'))}}
    (args.out / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
    prep_seconds = time.monotonic() - start
    print(json.dumps(config), flush=True)
    checkpoints = []
    torch.cuda.synchronize(device)
    start = time.monotonic()
    with (args.out / 'history.jsonl').open('w') as history:
        for epoch in range(1, 201):
            model.train()
            loss_sum, count, steps = 0., 0, 0
            batches, rng = hashlib.sha256(), hashlib.sha256()
            for (images, labels), (target_images, _) in zip(train, target_train):
                if len(images) != len(target_images):
                    continue
                for tensor in (images, labels, target_images):
                    batches.update(tensor.numpy().tobytes())
                rng.update(torch.get_rng_state().numpy().tobytes())
                rng.update(torch.cuda.get_rng_state(device).cpu().numpy().tobytes())
                images, labels, target_images = images.to(device), labels.to(device), target_images.to(device)
                optimizer.zero_grad(set_to_none=True)
                loss = F.cross_entropy(model(images, target_images)[0], labels)
                loss.backward()
                optimizer.step()
                loss_sum += loss.item() * len(labels)
                count += len(labels)
                steps += 1
            if count == 0:
                raise RuntimeError('No equal-sized paired batches')
            record = {'epoch': epoch, 'steps': steps, 'source_ce': loss_sum / count,
                      'batch_sha256': batches.hexdigest(), 'rng_sha256': rng.hexdigest()}
            if epoch == 1 or epoch % 10 == 0:
                record['source_val'] = helper.source_evaluate(model, val, device)
                print(json.dumps(record), flush=True)
            if epoch % 10 == 0:
                path = checkpoints_dir / f'epoch_{epoch:03d}.pth'
                torch.save({'model': model.state_dict(), 'epoch': epoch, 'config': config}, path)
                checkpoints.append({'epoch': epoch, 'path': str(path), 'sha256': helper.file_hash(path)})
            history.write(json.dumps(record) + '\n')
            history.flush()
    torch.cuda.synchronize(device)
    train_seconds = time.monotonic() - start
    start = time.monotonic()
    logits, predictions = helper.predict(model, target_eval, device)
    prediction_path = args.out / 'predictions_epoch_200.npz'
    np.savez_compressed(prediction_path, logits=logits, predictions=predictions)
    manifest = {'config': config, 'checkpoints': checkpoints,
                'prediction': {'path': str(prediction_path), 'sha256': helper.file_hash(prediction_path)},
                'prep_seconds': prep_seconds, 'train_seconds': train_seconds,
                'inference_seconds': time.monotonic() - start,
                'peak_cuda_memory_bytes': torch.cuda.max_memory_allocated(device)}
    manifest_path = args.out / 'frozen_manifest.json'
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
    manifest_hash = helper.file_hash(manifest_path)
    print(json.dumps({'frozen_manifest_sha256': manifest_hash}), flush=True)
    # Ground-truth values are accessed for scoring only after prediction freezing.
    labels = np.concatenate([labels.numpy() for _, labels in target_eval])
    metrics = helper.metrics(labels, predictions, logits, 7)
    metrics['collapse'] = max(metrics['prediction_shares']) >= .95
    metrics['class6_prediction_ratio'] = metrics['prediction_shares'][5]
    result = {'target_at_final': metrics, 'source_val_at_final': record['source_val'],
              'frozen_manifest_sha256': manifest_hash, **manifest}
    (args.out / 'results.json').write_text(json.dumps(result, indent=2) + '\n')
    print('RESULT ' + json.dumps(metrics), flush=True)


if __name__ == '__main__':
    main()

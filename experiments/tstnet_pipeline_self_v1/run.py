"""Our B/C models under TSTnet data/optimizer/loader/evaluation protocol."""
import ast
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import site
import sys
import time
from types import SimpleNamespace

site.addsitedir('/home/zhangzj26/TGRS_MLUDA-2024/.venv/lib/python3.8/site-packages')
PROJECT = Path('/home/zhangzj26/dual_axis_hsi_uda')
NATIVE = Path('/nas1/zhangzj26/dual_axis_hsi_uda/results/tstnet_hyrank_native_v1/run_snapshot')
ROOT = Path('/nas1/zhangzj26/dual_axis_hsi_uda/results/tstnet_pipeline_self_v1')
DATA = Path('/nas1/zhangzj26/dual_axis_hsi_uda/datasets/HyRANK_author_v1/HyRANK')


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    obj = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(obj)
    return obj


def state_hash(model):
    h = hashlib.sha256()
    for value in model.state_dict().values():
        h.update(value.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arm', choices=['B', 'C'], required=True)
    parser.add_argument('--preflight', action='store_true')
    own = parser.parse_args()
    outdir = ROOT / own.arm
    outdir.mkdir(parents=True, exist_ok=True)
    os.chdir(NATIVE)
    sys.path.insert(0, str(NATIVE))
    sys.path.insert(0, '/home/zhangzj26/IEEE_TCSVT_BiDA')
    sys.argv = ['train.py', '--data_path', str(DATA) + '/', '--source_name', 'Dioni', '--target_name', 'Loukia',
                '--lr', '0.001', '--re_ratio', '10', '--num_trials', '1', '--seed', '1233',
                '--save_path', str(outdir / 'native_checkpoints'), '--log_path', str(outdir / 'logs')]
    import numpy as np
    import torch
    from torch.nn import functional as F
    from models.get_model import get_model
    torch.set_num_threads(2)
    joint = module('pipeline_joint', PROJECT / 'experiments/bn_stabilization_v1/normalization.py')
    original = (PROJECT / 'experiments/relation_tokenizer_v1/relation.py').read_text()
    derived = original
    for a, b in [('bands=48', 'bands=176'), ('(1, 48)', '(1, 176)'), ('B×1×48×H×W', 'B×1×176×H×W')]:
        assert derived.count(a) == 1
        derived = derived.replace(a, b)
    relation = SimpleNamespace()
    rel_namespace = {}
    exec(compile(derived, 'relation_176.py', 'exec'), rel_namespace)
    relation.install = rel_namespace['install']
    tree = ast.parse((NATIVE / 'train.py').read_text())
    boundary = next(i for i, node in enumerate(tree.body) if isinstance(node, ast.If))
    ns = {'__name__': '__main__', '__file__': str(NATIVE / 'train.py')}
    exec(compile(ast.Module(body=tree.body[:boundary], type_ignores=[]), str(NATIVE / 'train.py'), 'exec'), ns)
    args = ns['args']
    opts = SimpleNamespace(dim=64, depth=3, num_tokens=4)

    def build(bands, num_classes, patch_size):
        assert bands == 176 and num_classes == 12 and patch_size == 12
        model = get_model('BiDA', 'Dioni', patch_size, opts)
        shared_initial = state_hash(model)
        unused_ema = get_model('BiDA', 'Dioni', patch_size, opts, ema=True)
        del unused_ema
        joint.install(model, 'fixed_mixture')
        if own.arm == 'C':
            relation.install(model, 'C', 1233)
        ns['_model_audit'] = {'shared_initial_sha256': shared_initial, 'parameters': sum(x.numel() for x in model.parameters()), 'bn_momentum': .19, 'initial_sha256': state_hash(model)}
        return model

    if own.preflight:
        ns['seed_worker'](1233)
        model = build(176, 12, 12).cuda().train()
        source = torch.randn(2, 1, 176, 12, 12, device='cuda')
        target = torch.randn_like(source).requires_grad_(True)
        loss = F.cross_entropy(model(source, target)[0], torch.tensor([0, 1], device='cuda'))
        assert torch.isfinite(loss)
        loss.backward()
        assert target.grad is not None and torch.isfinite(target.grad).all() and target.grad.abs().sum() > 0
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        audit = dict(ns['_model_audit'], loss=float(loss), target_gradient_l1=float(target.grad.abs().sum()), input_grid=[12, 12], token_count=4, token_dim=64)
        if own.arm == 'C':
            weights = model.relation_scores(source)
            assert weights.shape == (2, 4, 144)
            assert torch.allclose(weights.sum(-1), torch.ones_like(weights.sum(-1)), atol=1e-6)
            audit['relation_weight_shape'] = list(weights.shape)
            audit['anchor_index'] = [6, 6]
        model.eval()
        with torch.no_grad():
            assert model(source, source)[1].shape == (2, 12)
        (outdir / 'preflight.json').write_text(json.dumps(audit, indent=2))
        print(json.dumps(audit), flush=True)
        return

    if (outdir / 'history.jsonl').exists():
        raise RuntimeError('Refusing to overwrite an existing training run')
    for name in ['native_checkpoints', 'logs', 'predictions']:
        (outdir / name).mkdir(exist_ok=True)
    started = time.monotonic()

    def train(epoch, model, count):
        if epoch == 0:
            config = dict(arm=own.arm, seed=1233, epochs=500, patch=12, batch=100, source_fraction=.05,
                          source_repeat=10, source_split='native stratified random_state=23',
                          lr=.001, momentum=.9, weight_decay=1e-4, lr_schedule='native .001/(1+10*epoch/500)^.75',
                          optimizer_recreated_each_epoch=True, objective='source CE only',
                          native_grid=list(range(1, 501, 10)), primary='fixed epoch500',
                          oracle='native-grid diagnostic only, does not select formal model',
                          target_centers='native GT-stratified subset; label values never enter loss',
                          test_augmentation='native random flip/radiation enabled',
                          source_count=ns['len_src_dataset'], target_train_count=ns['len_tar_train_dataset'], target_eval_count=ns['len_tar_dataset'],
                          source_steps_per_epoch=ns['len_src_loader'] - 1, **ns['_model_audit'])
            config['index_hashes'] = {key: hashlib.sha256(ns[key].indices.tobytes()).hexdigest() for key in ['train_dataset', 'train_tar_dataset', 'test_dataset']}
            config['labels_hashes'] = {key: hashlib.sha256(np.asarray(ns[key].labels).tobytes()).hexdigest() for key in ['train_dataset', 'test_dataset']}
            (outdir / 'config.json').write_text(json.dumps(config, indent=2))
            print(json.dumps(config), flush=True)
        learning_rate = args.lr / (1 + 10 * epoch / count) ** .75
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=args.momentum, weight_decay=args.l2_decay)
        source_iter = ns['data_prefetcher'](ns['train_loader'])
        target_iter = ns['data_prefetcher'](ns['train_tar_loader'])
        steps, loss_sum, correct, seen = 0, 0., 0, 0
        label_hash = hashlib.sha256()
        for i in range(1, ns['len_src_loader']):
            model.train()
            if 0 < (ns['len_tar_train_dataset'] - i * args.batch_size) < args.batch_size or i % ns['len_tar_train_loader'] == 0:
                target_iter = ns['data_prefetcher'](ns['train_tar_loader'])
            source, labels = source_iter.next()
            target, _ = target_iter.next()
            assert source.shape[0] == target.shape[0] == 100
            labels = (labels - 1).long()
            label_hash.update(labels.detach().cpu().numpy().tobytes())
            optimizer.zero_grad()
            logits = model(source.unsqueeze(1), target.unsqueeze(1))[0]
            loss = F.nll_loss(F.log_softmax(logits, dim=1), labels)
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite loss; no automatic correction')
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()
            correct += int((logits.argmax(1) == labels).sum())
            seen += len(labels)
            steps += 1
        assert steps == 100
        record = dict(epoch=epoch + 1, steps=steps, source_ce=loss_sum / steps, source_train_oa=correct / ns['len_src_dataset'] * 100,
                      actual_consumed_source_labels_sha256=label_hash.hexdigest(), consumed_source_samples=seen,
                      rng_cpu_sha256=hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest(),
                      elapsed_seconds=time.monotonic() - started)
        ns['_epoch'] = epoch + 1
        with (outdir / 'history.jsonl').open('a') as f:
            f.write(json.dumps(record) + '\n')
        print(json.dumps(record), flush=True)
        if epoch + 1 == count:
            torch.save(model.state_dict(), outdir / 'final_epoch500.pth')
        return model, correct / ns['len_src_dataset'], 0., 0.

    def test(model):
        model.eval()
        preds, labels = [], []
        with torch.no_grad():
            for image, label in ns['test_loader']:
                image = image.cuda().unsqueeze(1)
                pred = model(image, image)[1].argmax(1).cpu().numpy()
                preds.append(pred)
                labels.append((label - 1).numpy())
        pred, truth = np.concatenate(preds), np.concatenate(labels)
        cm = np.bincount(12 * truth.astype(int) + pred.astype(int), minlength=144).reshape(12, 12)
        recall = np.diag(cm) / cm.sum(1)
        shares = np.bincount(pred, minlength=12) / len(pred)
        observed = np.trace(cm) / cm.sum()
        expected = (cm.sum(0) * cm.sum(1)).sum() / float(cm.sum() ** 2)
        record = dict(arm=own.arm, epoch=ns['_epoch'], oa=float(100 * observed), aa=float(100 * recall.mean()), kappa=float((observed - expected) / (1 - expected)),
                      per_class_recall=(100 * recall).tolist(), confusion_matrix=cm.tolist(), prediction_shares=shares.tolist(),
                      single_class_collapse=bool(shares.max() >= .95), zero_recall_classes=(np.flatnonzero(recall == 0) + 1).tolist())
        np.savez_compressed(outdir / 'predictions' / ('epoch_%03d.npz' % ns['_epoch']), predictions=pred, labels=truth)
        with (outdir / 'trajectory.jsonl').open('a') as f:
            f.write(json.dumps(record) + '\n')
        print('TARGET ' + json.dumps(record), flush=True)
        return torch.tensor(int(np.trace(cm))), observed, preds, labels

    ns.update(train=train, test=test, TSTnet=SimpleNamespace(Feature_Extractor=build), results={})
    exec(compile(ast.Module(body=tree.body[boundary:], type_ignores=[]), str(NATIVE / 'train.py'), 'exec'), ns)
    # Extra fixed-500 test occurs only AFTER all training, never changes its RNG path.
    test(ns['model_TST'])
    rows = [json.loads(x) for x in (outdir / 'trajectory.jsonl').read_text().splitlines()]
    summary = dict(status='completed', fixed500=rows[-1], native_grid_oracle=max(rows[:-1], key=lambda r: r['oa']), training_seconds=time.monotonic() - started)
    (outdir / 'results.json').write_text(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()

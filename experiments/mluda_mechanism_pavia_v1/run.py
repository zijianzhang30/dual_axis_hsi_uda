"""Run the native training AST with instrumentation and explicit mechanism factory."""
import argparse
import ast
import hashlib
import json
import os
import sys
import time

import numpy as np
import torch

from common import OUT, SEEDS, ARMS, dump, sha, tensor_hash, state_hash, rng_hash
from mechanisms import install, bn_counters


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arm', required=True, choices=ARMS)
    parser.add_argument('--seed', required=True, type=int, choices=SEEDS)
    args = parser.parse_args()
    folder = OUT / ('%s_%d' % (args.arm, args.seed))
    folder.mkdir(exist_ok=False)
    snapshot = OUT / 'native_snapshot'
    manifest = json.loads((OUT / 'manifest.json').read_text())
    for name, expected in manifest['snapshot_sha256'].items():
        assert sha(snapshot / name) == expected
    for name, expected in manifest['preprocessing_sha256'].items():
        assert sha(OUT / 'preprocessed' / name) == expected
    os.chdir(snapshot)
    sys.path.insert(0, str(snapshot))
    import net2
    import utils
    stream = hashlib.sha256()
    epoch_stream = hashlib.sha256()
    steps = 0
    model_ref = []
    calls = {}
    sample_hashes = {}

    native_sample = utils.get_sample_data
    native_all = utils.get_all_data

    def sample(*a, **kw):
        result = native_sample(*a, **kw)
        sample_hashes.update(source_data=tensor_hash(result[0]), source_labels=tensor_hash(result[1]))
        return result

    def all_data(*a, **kw):
        result = native_all(*a, **kw)
        sample_hashes.update(target_data=tensor_hash(result[1]), target_labels=tensor_hash(result[2]),
                             target_permutation=tensor_hash(result[4]), target_rows=tensor_hash(result[5]),
                             target_columns=tensor_hash(result[6]))
        np.savez_compressed(folder / 'target_order.npz', permutation=result[4], rows=result[5], columns=result[6])
        return result

    utils.get_sample_data = sample
    utils.get_all_data = all_data

    def factory(*a, **kw):
        model = net2.DSANSS(*a, **kw)
        before = {k: v.clone() for k, v in model.state_dict().items()}
        native_rng = rng_hash()
        install(model, args.arm in ('M1', 'M3'), args.arm in ('M2', 'M3'), args.seed)
        assert rng_hash() == native_rng
        for key, value in before.items():
            assert torch.equal(value, model.state_dict()[key]), key
        for name, layer in model.feature_layers.named_modules():
            if isinstance(layer, torch.nn.BatchNorm3d):
                calls[name] = 0
                def hook(mod, inp, output, name=name):
                    calls[name] += 1
                layer.register_forward_hook(hook)
        model_ref.append(model)
        dump(folder / 'initialization.json', {'native_state_sha256': state_hash(before),
             'native_rng_sha256': native_rng, 'full_state_sha256': state_hash(model.state_dict()),
             'samples': sample_hashes, 'bn_initial': bn_counters(model),
             'parameters': sum(p.numel() for p in model.parameters())})
        if hasattr(model.feature_layers, '_derived_source'):
            (folder / 'derived_feature_forward.py').write_text(model.feature_layers._derived_source)
        return model

    def audit_batch(*items):
        nonlocal steps
        h = hashlib.sha256()
        for value in items:
            h.update(tensor_hash(value).encode())
        h.update(rng_hash().encode())
        stream.update(h.digest())
        epoch_stream.update(h.digest())
        steps += 1

    def audit_epoch(epoch, model, optimizer, losses):
        nonlocal epoch_stream
        joint = args.arm in ('M1', 'M3')
        expected = epoch * 38 * 3 * (1 if joint else 2)
        counts = bn_counters(model)
        assert all(counts['bn%d' % i] == expected for i in range(1, 8)), counts
        assert counts['bn9'] == counts['bn10'] == 0
        pool = getattr(model.feature_layers, 'relation_pool', None)
        if pool is not None:
            ids = {id(p) for p in optimizer.param_groups[0]['params']}
            assert all(id(p) in ids for p in pool.parameters())
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in pool.parameters())
        record = {'epoch': epoch, 'updates': steps, 'batch_augmentation_rng_sha256': epoch_stream.hexdigest(),
                  'end_rng_sha256': rng_hash(), 'bn_counters': counts,
                  'bn_calls': dict(calls), 'losses': [float(v.detach()) for v in losses]}
        with (folder / 'history.jsonl').open('a') as f:
            f.write(json.dumps(record) + '\n')
        epoch_stream = hashlib.sha256()

    def freeze(model, source_data, loader, elapsed):
        assert steps == 3800
        model.eval()
        counts_before = bn_counters(model)
        preds = []
        labels = []
        source_hash = tensor_hash(source_data)
        evaluation_start = time.monotonic()
        with torch.no_grad():
            for data, label in loader:
                result = model(source_data.cuda(), data.cuda())
                preds.append(result[8].argmax(1).cpu().numpy())
                labels.append(label.numpy())
        assert counts_before == bn_counters(model)
        np.save(folder / 'predictions.npy', np.concatenate(preds))
        # Labels stored without scoring; separate summarize runs after all 12 freeze.
        np.save(folder / 'labels.npy', np.concatenate(labels))
        torch.save(model.state_dict(), folder / 'epoch100.pth')
        dump(folder / 'frozen.json', {'arm': args.arm, 'seed': args.seed, 'epoch': 100,
            'updates': steps, 'native_target_dataset_n': len(loader.dataset),
            'scored_n': sum(len(x) for x in preds), 'test_source_pair_sha256': source_hash,
            'samples': sample_hashes, 'batch_augmentation_rng_sha256': stream.hexdigest(),
            'parameters': sum(p.numel() for p in model.parameters()),
            'training_seconds': elapsed, 'evaluation_seconds': time.monotonic() - evaluation_start,
            'bn_train_counters': counts_before, 'bn_all_calls': calls,
            'predictions_sha256': sha(folder / 'predictions.npy'), 'labels_sha256': sha(folder / 'labels.npy'),
            'checkpoint_sha256': sha(folder / 'epoch100.pth')})

    source = (snapshot / 'MLUDA_up.py').read_text()
    source = source[:source.index('\nAA = np.mean(A, 1)')]
    source = source.replace('from UtilsCMS import *', 'from UtilsCMS import *\nnDataSet=1\nseeds=[_seed]\nDSANSS=_factory')
    source = source.replace('data_s,data_t = ILDA(data_s,data_t,pca_n,radius)', "data_s,data_t = np.load(_cache/'source.npy'), np.load(_cache/'target.npy')")
    source = source.replace('            (source_features, source1, _, source_outputs, source_out,', '            _audit_batch(source_data, source_label, target_data, source_data0,source_data1,target_data0,target_data1)\n            (source_features, source1, _, source_outputs, source_out,', 1)
    index = source.index('        if epoch % epochs == 0:')
    source = source[:index] + '        if epoch % epochs == 0:\n            _freeze(feature_encoder,source_data,test_loader,time.time()-train_start)\n        _audit_epoch(epoch, feature_encoder, optimizer, (cls_loss,lmmd_loss,contrastive_loss_s,contrastive_loss_t,loss))\n'
    (folder / 'derived_native_entry.py').write_text(source)
    dump(folder / 'config.json', {'arm': args.arm, 'seed': args.seed, 'epochs': 100,
         'batch': 32, 'patch': 11, 'gpu': os.environ.get('CUDA_VISIBLE_DEVICES'),
         'environment': {'torch': torch.__version__, 'numpy': np.__version__, 'python': sys.version},
         'native_entry_sha256': sha(snapshot / 'MLUDA_up.py'), 'derived_entry_sha256': sha(folder / 'derived_native_entry.py'),
         'joint_momentum': .19 if args.arm in ('M1','M3') else .1,
         'target_evaluation_only': True})
    env = {'__name__': '__main__', '_seed': args.seed, '_factory': factory,
           '_cache': OUT / 'preprocessed', '_audit_batch': audit_batch,
           '_audit_epoch': audit_epoch, '_freeze': freeze}
    exec(compile(source, str(folder / 'derived_native_entry.py'), 'exec'), env)
    assert (folder / 'frozen.json').exists()


if __name__ == '__main__':
    main()

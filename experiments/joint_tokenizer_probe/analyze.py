"""Read-only checkpoint probes for joint-BN tokenizer instability."""
import argparse
import importlib.util
import json
from pathlib import Path
from types import MethodType, SimpleNamespace
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('hybrid_v2_probe_helpers', ROOT / 'experiments/hybrid_v2/train.py')
hybrid = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hybrid)

class ChannelStats:
    def __init__(self):
        self.count, self.total, self.squared = 0, None, None
    def add(self, value):
        dims = (0,) + tuple(range(2, value.ndim))
        value = value.detach().double()
        total = value.sum(dims).cpu()
        squared = value.square().sum(dims).cpu()
        self.total = total if self.total is None else self.total + total
        self.squared = squared if self.squared is None else self.squared + squared
        self.count += value.numel() // value.shape[1]
    def report(self):
        mean = self.total / self.count
        variance = (self.squared / self.count - mean.square()).clamp_min(0)
        return {'channel_mean': mean.tolist(), 'channel_variance': variance.tolist(),
                'mean_channel_abs_mean': float(mean.abs().mean()),
                'mean_channel_variance': float(variance.mean()), 'max_channel_variance': float(variance.max())}

def vectors_report(values):
    values = torch.cat(values).double()
    norms = values.norm(dim=-1)
    unit = values / norms.clamp_min(1e-12).unsqueeze(-1)
    resultant = unit.mean(0)
    n = len(values)
    return {'mean_norm': float(norms.mean()), 'norm_sd': float(norms.std()),
            'norm_p95': float(torch.quantile(norms, .95)),
            'direction_concentration': float(resultant.norm()),
            'mean_pairwise_cosine': float((n * resultant.square().sum() - 1) / (n-1)),
            'mean_vector': values.mean(0).tolist()}, values

@torch.no_grad()
def probe(model, source, target, mode, device):
    model.eval() if mode == 'eval' else model.train()
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.eval()
    channels = {name: {domain: ChannelStats() for domain in ('source', 'target')}
                for name in ('bn3d', 'bn2d')}
    vectors = {domain: {key: [] for key in ('tokens', 'pre_norm_cls', 'z', 'logits')}
               for domain in ('source', 'target')}
    current = {'domain': 'source', 'tokens': 0, 'norm': 0, 'head': 0}
    def bn_hook(name):
        def hook(module, inputs, output):
            if mode == 'joint_train':
                left, right = output.split(128)
                channels[name]['source'].add(left)
                channels[name]['target'].add(right)
            else:
                # Eval forward tokenizes the same input twice; count once.
                if current['tokens'] == 0:
                    channels[name][current['domain']].add(output)
        return hook
    original_semantic = model._forward_semantic_tokens
    def semantic(self, features):
        output = original_semantic(features)
        domain = ('source' if current['tokens'] == 0 else 'target') if mode == 'joint_train' else current['domain']
        if mode == 'joint_train' or current['tokens'] == 0:
            vectors[domain]['tokens'].append(output.flatten(0, 1).cpu())
        current['tokens'] += 1
        return output
    def norm_hook(module, inputs):
        index = current['norm']
        if mode == 'eval' or index < 2:
            domain = current['domain'] if mode == 'eval' else ('source' if index == 0 else 'target')
            vectors[domain]['pre_norm_cls'].append(inputs[0][:, 0].cpu())
        current['norm'] += 1
    def head_hook(module, inputs, output):
        index = current['head']
        if mode == 'eval' or index < 2:
            domain = current['domain'] if mode == 'eval' else ('source' if index == 0 else 'target')
            vectors[domain]['z'].append(inputs[0].cpu())
            vectors[domain]['logits'].append(output.cpu())
        current['head'] += 1
    handles = [model.conv3d_features[1].register_forward_hook(bn_hook('bn3d')),
               model.conv2d_features[1].register_forward_hook(bn_hook('bn2d')),
               model.norm.register_forward_pre_hook(norm_hook), model.nn1.register_forward_hook(head_hook)]
    model._forward_semantic_tokens = MethodType(semantic, model)
    try:
        for start in range(0, len(source), 128):
            xs, xt = source[start:start+128].to(device), target[start:start+128].to(device)
            if mode == 'joint_train':
                current.update(tokens=0, norm=0, head=0)
                model(xs, xt)
            else:
                for domain, images in (('source', xs), ('target', xt)):
                    current.update(domain=domain, tokens=0, norm=0, head=0)
                    model(images, images)
    finally:
        model._forward_semantic_tokens = original_semantic
        for handle in handles:
            handle.remove()
    result = {}
    weight = model.nn1.weight.detach().cpu().double()
    for domain in ('source', 'target'):
        row = {name: channels[name][domain].report() for name in channels}
        matrices = {}
        for key in ('tokens', 'pre_norm_cls', 'z'):
            row[key], matrices[key] = vectors_report(vectors[domain][key])
        z = matrices['z']
        cosine = torch.nn.functional.normalize(z, dim=-1) @ torch.nn.functional.normalize(weight, dim=-1).T
        logits = torch.cat(vectors[domain]['logits']).double()
        margin = logits[:, 5] - torch.cat((logits[:, :5], logits[:, 6:]), dim=1).max(1).values
        row['classifier_cosine_mean'] = cosine.mean(0).tolist()
        row['logit_mean'] = logits.mean(0).tolist()
        row['class6_margin_mean'] = float(margin.mean())
        row['class6_margin_p95'] = float(torch.quantile(margin, .95))
        row['prediction_shares'] = (torch.bincount(logits.argmax(1), minlength=7) / len(logits)).tolist()
        row['token_to_bn2d_rms_ratio'] = row['tokens']['mean_norm'] / (64 * (np.mean(row['bn2d']['channel_variance']) + np.mean(np.square(row['bn2d']['channel_mean'])))) ** .5
        result[domain] = row
    for name in ('bn3d', 'bn2d'):
        ms = np.array(result['source'][name]['channel_mean'])
        mt = np.array(result['target'][name]['channel_mean'])
        result[name + '_domain_mean_distance'] = float(np.linalg.norm(ms - mt))
    result['source_target_z_mean_cosine'] = float(torch.nn.functional.cosine_similarity(
        torch.tensor(result['source']['z']['mean_vector'])[None], torch.tensor(result['target']['z']['mean_vector'])[None]))
    return result

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', required=True)
    parser.add_argument('--seeds', type=int, nargs='+', choices=(2100, 2101, 2102), required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    out = ROOT / 'results/joint_tokenizer_probe'
    out.mkdir(parents=True, exist_ok=True)
    opts = SimpleNamespace(seed=2100, num_workers=0, dataset_dir='/home/zhangzj26/TGRS_MLUDA-2024/datasets/Houston/', num_tokens=4, dim=64, depth=3)
    _, _, _, targets, sources, _ = hybrid.make_loaders(opts)
    rng = np.random.default_rng(93017)
    si = rng.choice(len(sources.dataset), 2048, replace=False)
    ti = rng.choice(len(targets.dataset), 2048, replace=False)
    source = torch.stack([sources.dataset[int(i)][0] for i in si])
    target = torch.stack([targets.dataset[int(i)][0] for i in ti])
    sample_hash = hybrid.helper.state_hash(torch.nn.Identity())
    import hashlib
    sample_hash = hashlib.sha256(source.numpy().tobytes() + target.numpy().tobytes()).hexdigest()
    device = torch.device('cuda:' + args.device)
    for seed in args.seeds:
        destination = out / f'seed_{seed}.json'
        if destination.exists():
            raise RuntimeError(f'Existing probe requires inspection: {destination}')
        rows = []
        for arm in ('bida', 'query'):
            folder = ROOT / ('results/bn_stabilization_v1' if arm == 'bida' else 'results/hybrid_v2') / (f'fixed_mixture_{seed}' if arm == 'bida' else f'formal_{seed}')
            manifest_path = folder / 'frozen_manifest.json'
            manifest = json.loads(manifest_path.read_text())
            result = json.loads((folder / 'results.json').read_text())
            assert hybrid.helper.file_hash(manifest_path) == result['frozen_manifest_sha256']
            model = hybrid.get_model('BiDA', 'Houston13', 13, opts)
            if arm == 'query':
                hybrid.transplant(model)
            hybrid.joint.install(model, 'fixed_mixture')
            model.to(device)
            for item in manifest['checkpoints']:
                assert hybrid.helper.file_hash(item['path']) == item['sha256']
                saved = torch.load(item['path'], map_location=device, weights_only=True)
                model.load_state_dict(saved['model'], strict=True)
                before = hybrid.helper.state_hash(model)
                modes = {}
                for mode in ('eval', 'joint_train'):
                    model.load_state_dict(saved['model'], strict=True)
                    modes[mode] = probe(model, source, target, mode, device)
                    model.load_state_dict(saved['model'], strict=True)
                    assert hybrid.helper.state_hash(model) == before
                row = {'arm': arm, 'epoch': item['epoch'], 'checkpoint_sha256': item['sha256'], 'modes': modes,
                       'classifier_weight_norm': model.nn1.weight.detach().norm(dim=1).cpu().tolist(),
                       'classifier_bias': model.nn1.bias.detach().cpu().tolist(),
                       'bn2d_affine_weight_norm': float(model.conv2d_features[1].weight.norm()),
                       'full_target_metrics': next(x['metrics'] for x in result['target_trajectory'] if x['epoch'] == item['epoch'])}
                rows.append(row)
                print(json.dumps({'seed': seed, 'arm': arm, 'epoch': item['epoch'], 'target_margin': modes['eval']['target']['class6_margin_mean'], 'target_token_norm': modes['eval']['target']['tokens']['mean_norm']}), flush=True)
            del model
        payload = {'seed': seed, 'sample_sha256': sample_hash, 'source_indices': si.tolist(), 'target_indices': ti.tolist(),
                   'protocol_sha256': hybrid.helper.file_hash(Path(__file__).with_name('PROTOCOL.md')),
                   'script_sha256': hybrid.helper.file_hash(Path(__file__)), 'rows': rows}
        destination.write_text(json.dumps(payload, indent=2) + '\n')

if __name__ == '__main__':
    main()

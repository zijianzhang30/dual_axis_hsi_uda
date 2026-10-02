"""Retrospective per-sublayer probes of frozen depth checkpoints."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import torch
import train
from layers import LayerCapture, report

ROOT = Path(__file__).resolve().parents[2]

@torch.no_grad()
def measure(model, images, device):
    model.eval()
    capture = LayerCapture(model)
    collected = {}
    try:
        for start in range(0, len(images), 128):
            capture.values = {}
            x = images[start:start+128].to(device)
            model(x, x)
            for key, value in capture.values.items():
                if key != 'final_logits':
                    collected.setdefault(key, []).append(value.cpu())
    finally:
        capture.close()
    return {key: report(values, model, final=key == 'final_cls') for key, values in collected.items()}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', required=True)
    parser.add_argument('--seeds', type=int, nargs='+', choices=(2100, 2101, 2102), required=True)
    parser.add_argument('--arms', nargs='+', choices=('query3', 'query1', 'bida3'), required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    device = torch.device('cuda:' + args.device)
    opts = SimpleNamespace(seed=2100, num_workers=0, dataset_dir='/home/zhangzj26/TGRS_MLUDA-2024/datasets/Houston/', num_tokens=4, dim=64, depth=3)
    _, _, _, target_loader, source_loader, _ = train.make_loaders(opts)
    sample = json.loads((ROOT / 'results/joint_tokenizer_probe/seed_2101.json').read_text())
    images = {'source': torch.stack([source_loader.dataset[i][0] for i in sample['source_indices']]),
              'target': torch.stack([target_loader.dataset[i][0] for i in sample['target_indices']])}
    import hashlib
    assert hashlib.sha256(images['source'].numpy().tobytes() + images['target'].numpy().tobytes()).hexdigest() == sample['sample_sha256']
    out = ROOT / 'results/hybrid_depth_v1/layer_probes'
    out.mkdir(parents=True, exist_ok=True)
    for seed in args.seeds:
        for arm in args.arms:
            destination = out / f'{arm}_{seed}.json'
            if destination.exists():
                raise RuntimeError(f'Existing layer probe requires inspection: {destination}')
            if arm == 'bida3':
                folder = ROOT / f'results/bn_stabilization_v1/fixed_mixture_{seed}'
            else:
                folder = ROOT / ('results/hybrid_v2' if arm == 'query3' else 'results/hybrid_depth_v1') / f'formal_{seed}'
            manifest_path = folder / 'frozen_manifest.json'
            manifest = json.loads(manifest_path.read_text())
            result = json.loads((folder / 'results.json').read_text())
            assert train.helper.file_hash(manifest_path) == result['frozen_manifest_sha256']
            model = train.get_model('BiDA', 'Houston13', 13, opts)
            if arm != 'bida3':
                train.transplant(model)
            if arm == 'query1':
                model.blocks = torch.nn.ModuleList([model.blocks[0]])
            train.joint.install(model, 'fixed_mixture')
            model.to(device)
            rows = []
            for item in manifest['checkpoints']:
                assert train.helper.file_hash(item['path']) == item['sha256']
                saved = torch.load(item['path'], map_location=device, weights_only=True)
                model.load_state_dict(saved['model'], strict=True)
                before = train.helper.state_hash(model)
                domains = {domain: measure(model, values, device) for domain, values in images.items()}
                assert train.helper.state_hash(model) == before
                rows.append({'epoch': item['epoch'], 'checkpoint_sha256': item['sha256'], 'domains': domains})
                print(json.dumps({'arm': arm, 'seed': seed, 'epoch': item['epoch'], 'final_concentration': domains['target']['final_cls']['direction_concentration']}), flush=True)
            destination.write_text(json.dumps({'arm': arm, 'seed': seed, 'sample_sha256': sample['sample_sha256'],
                                               'protocol_sha256': train.helper.file_hash(Path(__file__).with_name('PROTOCOL.md')),
                                               'probe_script_sha256': train.helper.file_hash(Path(__file__)),
                                               'layers_script_sha256': train.helper.file_hash(Path(__file__).with_name('layers.py')),
                                               'rows': rows}, indent=2) + '\n')
            del model

if __name__ == '__main__':
    main()

"""Label-free extraction. Does not import any scorer or open GT/label files."""
import importlib.util
import sys
import time
import numpy as np
import torch
from common import BASE, DATA, HERE, OUT, SEEDS, dump, sha, ahash

sys.dont_write_bytecode = True
sys.path.insert(0, str(BASE))
spec = importlib.util.spec_from_file_location('original_C_common', BASE / 'common.py')
frozen = importlib.util.module_from_spec(spec)
spec.loader.exec_module(frozen)

class Patches(torch.utils.data.Dataset):
    def __init__(self, domain):
        self.cube = np.load(DATA / f'{domain}_cube.npy', mmap_mode='r')
        self.coords = np.load(DATA / f'{domain}_coords.npy')

    def __len__(self):
        return len(self.coords)

    def __getitem__(self, index):
        x, y = self.coords[index]
        patch = np.array(self.cube[x-6:x+7, y-6:y+7], dtype=np.float32)
        return torch.from_numpy(patch.transpose(2, 0, 1)).unsqueeze(0)

@torch.inference_mode()
def extract(model, seed, domain, folder):
    dataset = Patches(domain)
    np.save(folder / f'{domain}_coords.npy', dataset.coords)
    loader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=False, num_workers=0)
    arrays, latest, norms = {}, {}, {}
    hook = model.conv2d_features.register_forward_hook(lambda m, args, out: latest.update(V=out))
    original = model._forward_semantic_tokens
    def tokens(v):
        t = original(v)
        latest['T'] = t
        return t
    model._forward_semantic_tokens = tokens
    official = np.load(BASE / 'houston' / f'C_{seed}/predictions_200.npz') if domain == 'target' else None
    oldtokens = np.load(BASE / 'houston' / f'C_{seed}/tokens_200.npy', mmap_mode='r') if domain == 'target' else None
    maxlogit = maxtoken = 0.0
    pos = 0
    try:
        for patch in loader:
            assert not model.training and all(not m.training for m in model.modules())
            logits, z = model(patch.cuda(), patch.cuda(), return_feat_prob=True)[1::2]
            batch = {}
            for stage, value in [('V', latest['V']), ('T', latest['T']), ('Z', z)]:
                flat = value.flatten(1)
                norm = flat.norm(dim=1)
                assert torch.isfinite(flat).all() and (norm > 1e-12).all()
                norms.setdefault(stage, []).append(norm.cpu().numpy())
                batch[stage] = torch.nn.functional.normalize(flat, dim=1, eps=1e-12).cpu().numpy()
            batch['logits'] = logits.cpu().numpy()
            batch['predictions'] = batch['logits'].argmax(1).astype(np.int64)
            n = len(patch)
            if official is not None:
                assert np.array_equal(batch['predictions'], official['predictions'][pos:pos+n])
                maxlogit = max(maxlogit, float(abs(batch['logits']-official['logits'][pos:pos+n]).max()))
                maxtoken = max(maxtoken, float(abs(latest['T'].cpu().numpy()-oldtokens[pos:pos+n]).max()))
            for key, a in batch.items():
                if key not in arrays:
                    arrays[key] = np.lib.format.open_memmap(folder / f'{domain}_{key}.npy', mode='w+', dtype=a.dtype, shape=(len(dataset), *a.shape[1:]))
                arrays[key][pos:pos+n] = a
            pos += n
    finally:
        hook.remove()
        model._forward_semantic_tokens = original
    assert pos == len(dataset) and max(maxlogit, maxtoken) < 2e-5, (pos, maxlogit, maxtoken)
    for a in arrays.values():
        a.flush()
    return {'samples': pos, 'coordinate_array_sha256': ahash(dataset.coords), 'native_logits_max_error': maxlogit,
            'native_tokens_max_error': maxtoken, 'native_predictions_exact': domain == 'target',
            'raw_feature_norms': {k: {'min': float(np.concatenate(v).min()), 'mean': float(np.concatenate(v).mean())} for k, v in norms.items()}}

def main():
    torch.set_num_threads(2)
    OUT.mkdir(exist_ok=False)
    # These metadata hashes were fixed by historical preparation, without opening labels.
    metadata = __import__('json').loads((DATA.parent / 'fixed_inputs.json').read_text())
    inputs = {str(DATA / f'{d}_{k}.npy'): sha(DATA / f'{d}_{k}.npy') for d in ['source', 'target'] for k in ['cube', 'coords']}
    codes = {str(p): sha(p) for p in HERE.glob('*') if p.is_file()}
    provenance = {'protocol_sha256': sha(HERE / 'PROTOCOL.txt'), 'code_sha256': codes,
                  'input_sha256': inputs, 'historical_input_metadata': metadata, 'seeds': {}}
    start = time.monotonic()
    for seed in SEEDS:
        folder = OUT / f'seed_{seed}'
        folder.mkdir()
        cp = BASE / 'houston' / f'C_{seed}/checkpoints/epoch_200.pth'
        cph = sha(cp)
        assert cph == metadata['checkpoints'][str(seed)]['sha256']
        config = __import__('json').loads((cp.parent.parent / 'config.json').read_text())
        assert config['arm'] == 'C' and config['epochs'] == 200 and config['BN'] == 'unchanged Full Joint/momentum.19'
        frozen.environment()[4](seed)
        model, _ = frozen.build(seed, 'C', 'houston')
        model.load_state_dict(torch.load(cp, map_location='cpu', weights_only=True)['model'], strict=True)
        model.cuda().eval().requires_grad_(False)
        before = frozen.state_hash(model)
        audit = {}
        for domain in ['source', 'target']:
            audit[domain] = extract(model, seed, domain, folder)
            assert audit[domain]['coordinate_array_sha256'] == metadata['domains'][domain]['coords_sha256']
            print(seed, domain, 'extracted; eval/frozen BN', flush=True)
        assert frozen.state_hash(model) == before and sha(cp) == cph
        provenance['seeds'][str(seed)] = {'checkpoint': str(cp), 'checkpoint_sha256': cph, 'state_sha256': before,
                                        'state_unchanged': True, 'config': config, 'audit': audit}
        del model
        torch.cuda.empty_cache()
    for path, h in inputs.items():
        assert sha(path) == h
    dump(OUT / 'extraction_provenance.json', provenance)
    files = {str(p.relative_to(OUT)): sha(p) for p in OUT.rglob('*.npy')}
    files['extraction_provenance.json'] = sha(OUT / 'extraction_provenance.json')
    dump(OUT / 'ALL_FEATURES_PREDICTIONS_FROZEN.json', {'files_sha256': files, 'all_seeds_completed': True,
         'target_GT_opened_by_extractor': False, 'seconds': time.monotonic()-start, 'protocol_sha256': provenance['protocol_sha256']})
    print('ALL THREE SEEDS FROZEN; no target labels opened', flush=True)

if __name__ == '__main__':
    main()

"""Freeze native raw CLS/head before any target labels are opened."""
import json
import shutil
import numpy as np
import torch
from common import HERE,OUT,CACHE,RAW,DATA,BASE,SEEDS,sha,ahash,unit,dump

def main():
    OUT.mkdir(exist_ok=False)
    gate=json.loads((CACHE/'ALL_NEIGHBORS_FROZEN.json').read_text())
    old=json.loads((CACHE/'provenance.json').read_text())
    assert gate['all_seeds_completed'] and not gate['target_GT_opened_by_this_diagnostic']
    assert sha(CACHE/'provenance.json')==gate['files_sha256']['provenance.json']
    assert sha(CACHE/'source_train_mask.npy')==gate['files_sha256']['source_train_mask.npy']
    shutil.copyfile(CACHE/'source_train_mask.npy',OUT/'source_train_mask.npy')
    mask=np.load(OUT/'source_train_mask.npy');assert mask.sum()==2393
    sy=np.load(DATA/'source_labels.npy')
    assert ahash(sy)==old['historical_metadata']['domains']['source']['labels_sha256']
    np.save(OUT/'source_labels.npy',sy)
    prov={'protocol_sha256':sha(HERE/'PROTOCOL.txt'),'code_sha256':{str(p):sha(p) for p in HERE.glob('*') if p.is_file()},
          'cached_manifest_sha256':sha(CACHE/'ALL_NEIGHBORS_FROZEN.json'),
          'historical_metadata':old['historical_metadata'],'seeds':{}}
    for seed in SEEDS:
        folder=OUT/f'seed_{seed}';folder.mkdir()
        rec=old['seeds'][str(seed)];cp=rec['checkpoint'];assert sha(cp)==rec['checkpoint_sha256']
        config=json.loads((BASE/'houston'/f'C_{seed}/config.json').read_text())
        assert config['arm']=='C' and config['epochs']==200 and config['BN']=='unchanged Full Joint/momentum.19'
        state=torch.load(cp,map_location='cpu',weights_only=True)['model']
        weight=state['nn1.weight'].numpy();bias=state['nn1.bias'].numpy()
        np.savez_compressed(folder/'frozen_head.npz',weight=weight,bias=bias)
        audit={}
        for domain in ['source','target']:
            for key in ['Z','logits','predictions','coords']:
                rel=f'seed_{seed}/{domain}_{key}.npy';assert sha(CACHE/rel)==gate['files_sha256'][rel]
                shutil.copyfile(CACHE/rel,folder/f'{domain}_{"unitZ" if key=="Z" else key}.npy')
            src=RAW/f'seed_{seed}/{domain}_Z.npy';rh=sha(src)
            z=np.load(src);cached=np.load(folder/f'{domain}_unitZ.npy')
            assert z.shape==cached.shape and z.shape[1]==64 and np.isfinite(z).all()
            direction_error=float(abs(unit(z.astype(np.float64))-cached).max());assert direction_error<2e-6
            logits=z.astype(np.float64)@weight.astype(np.float64).T+bias.astype(np.float64)
            native=np.load(folder/f'{domain}_logits.npy');pred=np.load(folder/f'{domain}_predictions.npy')
            error=float(abs(logits-native).max());assert error<2e-5
            assert np.array_equal(logits.argmax(1),pred) and np.array_equal(native.argmax(1),pred)
            shutil.copyfile(src,folder/f'{domain}_rawZ.npy');assert sha(src)==rh==sha(folder/f'{domain}_rawZ.npy')
            np.save(folder/f'{domain}_sample_indices.npy',np.arange(len(z)))
            coords=np.load(folder/f'{domain}_coords.npy')
            assert ahash(coords)==old['historical_metadata']['domains'][domain]['coords_sha256']
            if domain=='source':
                assert ahash(coords[mask])==config['indices_sha256']['source']
                assert ahash(coords[~mask])==config['indices_sha256']['val']
            else:assert ahash(coords)==config['indices_sha256']['target_eval']
            audit[domain]={'raw_cache_file':str(src),'raw_cache_sha256':rh,'unit_direction_max_error':direction_error,
                           'native_head_logit_max_error':error,'native_predictions_exact':True}
        prov['seeds'][str(seed)]={'checkpoint':cp,'checkpoint_sha256':rec['checkpoint_sha256'],'audit':audit}
        print(seed,'raw CLS and native classifier authenticated/frozen',flush=True)
    dump(OUT/'provenance.json',prov)
    files={str(p.relative_to(OUT)):sha(p) for p in OUT.rglob('*') if p.is_file()}
    dump(OUT/'ALL_INPUTS_FROZEN.json',{'files_sha256':files,'all_seeds_completed':True,
        'target_GT_opened_by_input_freezer':False,'raw_CLS_classifier_replay_exact_predictions':True})
    print('All native inputs frozen; target GT unopened',flush=True)

if __name__=='__main__':main()

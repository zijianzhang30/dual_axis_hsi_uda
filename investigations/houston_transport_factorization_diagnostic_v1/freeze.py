"""Predicted factors and D00 only; no target GT or oracle branch access."""
import json
import shutil
import numpy as np
from common import HERE,OUT,CACHE,SEEDS,sha,ahash,dump

def main():
    OUT.mkdir(exist_ok=False)
    gate=json.loads((CACHE/'ALL_UNSUPERVISED_FROZEN.json').read_text())
    old=json.loads((CACHE/'provenance.json').read_text())
    assert gate['all_seeds_completed'] and not gate['target_GT_opened_by_constructor']
    assert not gate['oracle_G2_artifacts_opened_by_constructor']
    assert sha(CACHE/'provenance.json')==gate['files_sha256']['provenance.json']
    for rel in ['source_labels.npy','source_train_mask.npy']:
        assert sha(CACHE/rel)==gate['files_sha256'][rel]
        shutil.copyfile(CACHE/rel,OUT/rel)
    train=np.load(OUT/'source_train_mask.npy');sy=np.load(OUT/'source_labels.npy')
    assert train.sum()==2393
    provenance={'protocol_sha256':sha(HERE/'PROTOCOL.txt'),
        'code_sha256':{str(p):sha(p) for p in HERE.glob('*') if p.is_file()},
        'cached_unsupervised_manifest_sha256':sha(CACHE/'ALL_UNSUPERVISED_FROZEN.json'),
        'historical_metadata':old['historical_metadata'],'factorization':'hard F1 anchored','seeds':{}}
    for seed in SEEDS:
        folder=OUT/f'seed_{seed}';folder.mkdir();rec=old['seeds'][str(seed)]
        assert sha(rec['checkpoint'])==rec['checkpoint_sha256']
        names=['frozen_head.npz','fixed_unlabeled_statistics.npz','F1_hard_routing.npy']
        names += [f'{d}_{key}.npy' for d in ['source','target'] for key in ['rawZ','coords','sample_indices','logits','predictions']]
        for name in names:
            rel=f'seed_{seed}/{name}';assert sha(CACHE/rel)==gate['files_sha256'][rel]
            shutil.copyfile(CACHE/rel,folder/name);assert sha(folder/name)==gate['files_sha256'][rel]
        s=np.load(folder/'source_rawZ.npy').astype(float)[train];z=np.load(folder/'target_rawZ.npy').astype(float)
        routing=np.load(folder/'F1_hard_routing.npy');native=np.load(folder/'target_predictions.npy')
        assert np.array_equal(routing,native)
        source_means=np.stack([s[sy[train]==c].mean(0) for c in range(7)])
        counts=np.bincount(routing,minlength=7)
        pseudo=np.stack([z[routing==c].mean(0) if counts[c] else source_means[c] for c in range(7)])
        stats=np.load(folder/'fixed_unlabeled_statistics.npz')
        assert np.array_equal(source_means,stats['source_means']) and np.array_equal(pseudo,stats['hard_target_means'])
        np.savez_compressed(folder/'predicted_factors.npz',source_means=source_means,pseudo_target_means=pseudo,
                            pseudo_assignment_counts=counts,predicted_routing=routing)
        d00=z+(source_means-pseudo)[routing]
        refrel=f'seed_{seed}/F1_rawZ.npy';assert sha(CACHE/refrel)==gate['files_sha256'][refrel]
        assert np.array_equal(d00,np.load(CACHE/refrel))
        head=np.load(folder/'frozen_head.npz');w=head['weight'].astype(float);b=head['bias'].astype(float)
        raw=z@w.T+b;assert np.array_equal(raw.argmax(1),native)
        error=float(abs(raw-np.load(folder/'target_logits.npy')).max());assert error<2e-5
        logits=d00@w.T+b
        readrel=f'seed_{seed}/F1_readouts.npz';assert sha(CACHE/readrel)==gate['files_sha256'][readrel]
        reference=np.load(CACHE/readrel)
        assert np.array_equal(logits,reference['classifier_logits'])
        assert np.array_equal(logits.argmax(1),reference['classifier_predictions'])
        np.save(folder/'D00_rawZ.npy',d00)
        np.savez_compressed(folder/'D00_readouts.npz',logits=logits,predictions=logits.argmax(1))
        coords=np.load(folder/'target_coords.npy')
        assert ahash(coords)==old['historical_metadata']['domains']['target']['coords_sha256']
        provenance['seeds'][str(seed)]={'checkpoint':rec['checkpoint'],'checkpoint_sha256':rec['checkpoint_sha256'],
            'predicted_factors_exact_replay':True,'D00_F1_features_logits_predictions_exact_replay':True,
            'native_logit_max_error':error,'native_predictions_exact':True}
        print(seed,'predicted factors and D00 frozen; exact F1 replay',flush=True)
    dump(OUT/'provenance.json',provenance)
    files={str(p.relative_to(OUT)):sha(p) for p in OUT.rglob('*') if p.is_file()}
    dump(OUT/'ALL_PREDICTED_FACTORS_FROZEN.json',{'files_sha256':files,'all_seeds_completed':True,
        'target_GT_opened':False,'oracle_G2_opened':False,'predicted_statistics_routing_and_D00_all_frozen':True})
    print('All predicted factors/D00 frozen before target GT',flush=True)

if __name__=='__main__':main()

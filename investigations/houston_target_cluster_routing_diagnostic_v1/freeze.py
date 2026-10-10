"""Authenticate inputs, cluster/match, freeze transport: no target GT access."""
import json
import shutil
import numpy as np
import scipy
import sklearn
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from common import HERE,OUT,CACHE,FACT,SEEDS,STAGES,CONFIG,sha,ahash,dump,unit

def copy(src,dst,root,gate):
    rel=str(src.relative_to(root));assert sha(src)==gate['files_sha256'][rel]
    shutil.copyfile(src,dst);assert sha(dst)==gate['files_sha256'][rel]

def main():
    OUT.mkdir(exist_ok=False)
    cg=json.loads((CACHE/'ALL_ROUTING_TRANSPORT_FROZEN.json').read_text())
    fg=json.loads((FACT/'ALL_PREDICTED_FACTORS_FROZEN.json').read_text())
    assert not cg['target_GT_opened_by_constructor'] and not fg['target_GT_opened']
    old=json.loads((CACHE/'provenance.json').read_text())
    prov={'fixed_config':CONFIG,'sklearn_version':sklearn.__version__,'scipy_version':scipy.__version__,
      'parent_router_gate_sha256':sha(CACHE/'ALL_ROUTING_TRANSPORT_FROZEN.json'),
      'parent_factor_gate_sha256':sha(FACT/'ALL_PREDICTED_FACTORS_FROZEN.json'),
      'code_sha256':{str(p):sha(p) for p in HERE.glob('*') if p.is_file()},
      'historical_metadata':old['historical_metadata'],'seeds':old['seeds']}
    for name in ['source_labels.npy','source_train_mask.npy']:copy(CACHE/name,OUT/name,CACHE,cg)
    sy=np.load(OUT/'source_labels.npy');train=np.load(OUT/'source_train_mask.npy');assert train.sum()==2393
    assert ahash(sy)==old['historical_metadata']['domains']['source']['labels_sha256']
    for seed in SEEDS:
        f=OUT/f'seed_{seed}';f.mkdir();rec=old['seeds'][str(seed)];assert sha(rec['checkpoint'])==rec['checkpoint_sha256']
        for domain in ['source','target']:
            for stage in STAGES:copy(CACHE/f'seed_{seed}/{domain}_{stage}.npy',f/f'{domain}_{stage}.npy',CACHE,cg)
        for name in ['target_rawZ.npy','target_logits.npy','target_predictions.npy','target_coords.npy','target_sample_indices.npy','source_coords.npy','frozen_head.npz','predicted_factors.npz']:
            copy(FACT/f'seed_{seed}/{name}',f/name,FACT,fg)
        assert ahash(np.load(f/'target_coords.npy'))==old['historical_metadata']['domains']['target']['coords_sha256']
    dump(OUT/'provenance.json',prov)
    dump(OUT/'ALL_INPUTS_FROZEN.json',{'target_GT_opened':False,'all_seeds_completed':True,
      'files_sha256':{str(p.relative_to(OUT)):sha(p) for p in OUT.rglob('*') if p.is_file()}})
    print('All fixed features/labels(native source only)/head/statistics input hashes frozen; target GT unopened.',flush=True)
    for seed in SEEDS:
        f=OUT/f'seed_{seed}';z=np.load(f/'target_rawZ.npy').astype(float);logits=np.load(f/'target_logits.npy').astype(float);r0=np.load(f/'target_predictions.npy')
        stats=np.load(f/'predicted_factors.npz');delta=stats['source_means']-stats['pseudo_target_means']
        head=np.load(f/'frozen_head.npz');w=head['weight'].astype(float);b=head['bias'].astype(float)
        assert np.array_equal((z@w.T+b).argmax(1),r0) and np.array_equal(logits.argmax(1),r0)
        assert np.array_equal(stats['predicted_routing'],r0)
        d00=z+delta[r0];dlog=d00@w.T+b
        for name in ['D00_rawZ.npy','D00_readouts.npz']:
            rel=f'seed_{seed}/{name}';assert sha(FACT/rel)==fg['files_sha256'][rel]
        assert np.array_equal(d00,np.load(FACT/f'seed_{seed}/D00_rawZ.npy'))
        assert np.array_equal(dlog,np.load(FACT/f'seed_{seed}/D00_readouts.npz')['logits'])
        np.savez_compressed(f/'R0_readouts.npz',routing=r0,logits=logits,predictions=r0)
        np.save(f/'D00_rawZ.npy',d00);np.savez_compressed(f/'D00_readouts.npz',routing=r0,logits=dlog,predictions=dlog.argmax(1))
        for stage in STAGES:
            source=unit(np.load(f/f'source_{stage}.npy').astype(float)[train]);target=unit(np.load(f/f'target_{stage}.npy').astype(float))
            source_centers=unit(np.stack([source[sy[train]==c].mean(0) for c in range(7)]))
            km=KMeans(n_clusters=7,init='k-means++',n_init=10,max_iter=300,tol=1e-4,algorithm='lloyd',random_state=0)
            clusters=km.fit_predict(target);centers=km.cluster_centers_;cost=1-unit(centers)@source_centers.T
            ri,ci=linear_sum_assignment(cost);mapping=np.full(7,-1,int);mapping[ri]=ci
            assert np.array_equal(np.sort(mapping),np.arange(7)) and len(np.unique(clusters))==7
            routing=mapping[clusters];transformed=z+delta[routing];scores=transformed@w.T+b
            np.savez_compressed(f/f'{stage}_clustering_matching.npz',clusters=clusters,centers=centers,normalized_centers=unit(centers),
                source_centers=source_centers,cosine_cost=cost,cluster_to_source_class=mapping,routing=routing,
                inertia=km.inertia_,n_iter=km.n_iter_,cluster_sizes=np.bincount(clusters,minlength=7))
            np.save(f/f'{stage}_cluster_rawZ.npy',transformed)
            np.savez_compressed(f/f'{stage}_cluster_readouts.npz',routing=routing,logits=scores,predictions=scores.argmax(1))
            assert np.array_equal(routing,mapping[km.predict(target)])
            print(seed,stage,'KMeans/Hungarian/transport frozen; n_iter',km.n_iter_,'cluster sizes',np.bincount(clusters).tolist(),flush=True)
    for rel,h in json.loads((OUT/'ALL_INPUTS_FROZEN.json').read_text())['files_sha256'].items():assert sha(OUT/rel)==h
    dump(OUT/'ALL_CLUSTER_ROUTING_TRANSPORT_FROZEN.json',{'all_seeds_completed':True,'target_GT_opened':False,
      'no_network_training_backprop_tuning_followup':True,'D00_pseudo_statistics_unchanged':True,
      'files_sha256':{str(p.relative_to(OUT)):sha(p) for p in OUT.rglob('*') if p.is_file()}})
    print('All T/Z cluster matching/routing/transport across3seeds hash-frozen; target GT unopened.',flush=True)

if __name__=='__main__':main()

"""Construct frozen counterfactuals using source labels only. Never opens target GT."""
import importlib.util
import json
import shutil
import time
import numpy as np
import torch
from common import HERE,OUT,CACHE,DATA,BASE,SEEDS,STAGES,MODES,KS,sha,ahash,unit,dump

# Reuse only the independently verified exact tie-safe selection function.
previous=HERE.parent/'houston_local_support_diagnostic_v1/freeze_neighbors.py'
spec=importlib.util.spec_from_file_location('previous_exact_neighbors',previous)
helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)

@torch.inference_mode()
def search(query,reference,qcoords=None,rcoords=None):
    q=torch.tensor(query,device='cuda',dtype=torch.float64)
    r=torch.tensor(reference,device='cuda',dtype=torch.float64)
    qi=torch.tensor(qcoords,device='cuda') if qcoords is not None else None
    ri=torch.tensor(rcoords,device='cuda') if rcoords is not None else None
    ids,cosines,excluded=[],[],[];ties=0
    for start in range(0,len(q),512):
        score=q[start:start+512]@r.T
        if qi is not None:
            a=qi[start:start+len(score)]
            mask=(abs(a[:,None,0]-ri[None,:,0])<=12)&(abs(a[:,None,1]-ri[None,:,1])<=12)
            n=mask.sum(1);assert ((len(r)-n)>=16).all()
            score.masked_fill_(mask,-torch.inf);excluded.append(n.cpu().numpy())
        ix,vs,nt=helper.select_neighbors(score)
        ids.append(ix);cosines.append(vs);ties+=nt
        if start%(512*25)==0:print('  exact search',start,'/',len(q),'nonoverlap' if qi is not None else 'source',flush=True)
    return np.concatenate(ids),np.concatenate(cosines),np.concatenate(excluded) if excluded else None,ties

def source_outputs(q,source,source_ids,labels,centroids):
    ids,cosines,_,ties=search(q,source)
    ids=source_ids[ids]
    scores=q@centroids.T
    arrays={'source_indices':ids,'source_cosines':cosines,'centroid_scores':scores,
            'centroid_predictions':scores.argmax(1).astype(np.int64)}
    for k in KS:
        counts=np.stack([(labels[ids[:,:k]]==c).sum(1) for c in range(7)],axis=1)
        arrays[f'source_counts_k{k}']=counts
        arrays[f'source_knn_predictions_k{k}']=counts.argmax(1).astype(np.int64)
    arrays['source_cutoff_tie_rows']=np.array(ties)
    return arrays

def coordinate_check(coords,config,metadata):
    assert coords.shape==(52901,2) and np.issubdtype(coords.dtype,np.integer)
    assert len(np.unique(coords,axis=0))==len(coords)
    assert ahash(coords)==config['indices_sha256']['target_eval']==metadata['domains']['target']['coords_sha256']
    height,width=metadata['domains']['target']['shape']
    assert (coords>=6).all() and (coords[:,0]<height+6).all() and (coords[:,1]<width+6).all()

def fixtures():
    helper.check_knn()
    rng=np.random.default_rng(2100);q=unit(rng.normal(size=(40,9)))
    coords=np.column_stack((np.arange(40)//2*26,np.arange(40)%2*2))
    ids,cs,_,_=search(q,q,coords,coords)
    score=q@q.T
    for i in range(40):
        invalid=np.max(abs(coords-coords[i]),axis=1)<=12
        score[i,invalid]=-np.inf
        expected=np.lexsort((np.arange(40),-score[i]))[:15]
        assert np.array_equal(ids[i],expected)
        assert (np.max(abs(coords[ids[i]]-coords[i]),axis=1)>12).all()
    # Validate exactly fixed one-step consensus, without graph/parameter mutation.
    consensus=unit(q+q[ids[:,:5]].mean(1))
    assert np.allclose(np.linalg.norm(consensus,axis=1),1)

def main():
    start=time.monotonic();torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    OUT.mkdir(exist_ok=False)
    oldgate=json.loads((CACHE/'ALL_NEIGHBORS_FROZEN.json').read_text())
    oldprov=json.loads((CACHE/'provenance.json').read_text())
    assert oldgate['all_seeds_completed'] and not oldgate['target_GT_opened_by_this_diagnostic']
    assert sha(CACHE/'provenance.json')==oldgate['files_sha256']['provenance.json']
    selected=['source_train_mask.npy']
    for seed in SEEDS:
        selected.extend(f'seed_{seed}/{d}_{key}.npy' for d in ['source','target'] for key in ['T','Z','coords','logits','predictions'])
        selected.extend(f'seed_{seed}/{stage}_neighbors.npz' for stage in STAGES)
        (OUT/f'seed_{seed}').mkdir()
    for rel in selected:
        assert sha(CACHE/rel)==oldgate['files_sha256'][rel]
        shutil.copyfile(CACHE/rel,OUT/rel)
        assert sha(OUT/rel)==oldgate['files_sha256'][rel]
    mask=np.load(OUT/'source_train_mask.npy');source_ids=np.flatnonzero(mask);assert len(source_ids)==2393
    # Source labels are legitimate references for frozen source-trained readouts.
    sy=np.load(DATA/'source_labels.npy')
    assert ahash(sy)==oldprov['historical_metadata']['domains']['source']['labels_sha256']
    np.save(OUT/'source_labels.npy',sy)
    provenance={'cached_preGT_manifest_sha256':sha(CACHE/'ALL_NEIGHBORS_FROZEN.json'),
                'protocol_sha256':sha(HERE/'PROTOCOL.txt'),'code_sha256':{str(p):sha(p) for p in HERE.glob('*') if p.is_file()},
                'neighbor_helper_sha256':sha(previous),'historical_metadata':oldprov['historical_metadata'],
                'source_labels_sha256':sha(DATA/'source_labels.npy'),'seeds':{},
                'counterfactual_classifier':'not executed: unit consensus Z does not preserve native raw-CLS input normalization'}
    fixtures()
    for seed in SEEDS:
        folder=OUT/f'seed_{seed}';rec=oldprov['seeds'][str(seed)]
        assert sha(rec['checkpoint'])==rec['checkpoint_sha256']
        config=json.loads((BASE/'houston'/f'C_{seed}/config.json').read_text())
        assert config['arm']=='C' and config['epochs']==200
        scoords=np.load(folder/'source_coords.npy');coords=np.load(folder/'target_coords.npy')
        assert ahash(scoords[mask])==config['indices_sha256']['source']
        assert ahash(scoords[~mask])==config['indices_sha256']['val']
        coordinate_check(coords,config,oldprov['historical_metadata'])
        np.save(folder/'source_sample_indices.npy',np.arange(len(scoords)))
        np.save(folder/'target_sample_indices.npy',np.arange(len(coords)))
        audit={'coordinate_authentication':True,'patch':13,'exclusion_Chebyshev_leq':12,'common_padding_translation':6,'stages':{}}
        for stage in STAGES:
            source=unit(np.load(folder/f'source_{stage}.npy').astype(np.float64))[mask]
            target=unit(np.load(folder/f'target_{stage}.npy').astype(np.float64))
            assert np.isfinite(source).all() and np.isfinite(target).all()
            centroids=unit(np.stack([source[sy[mask]==c].mean(0) for c in range(7)]))
            np.save(folder/f'{stage}_source_centroids.npy',centroids)
            np.save(folder/f'{stage}_before_features.npy',target)
            baseline=source_outputs(target,source,source_ids,sy,centroids)
            old=np.load(folder/f'{stage}_neighbors.npz')
            assert np.array_equal(baseline['source_indices'],old['source_indices'])
            np.savez_compressed(folder/f'{stage}_before_readouts.npz',**baseline)
            nonids,noncos,excluded,ties=search(target,target,coords,coords)
            neighborhoods={'standard_knn':(old['target_indices'],old['target_cosines']),
                           'nonoverlap_knn':(nonids,noncos)}
            audit['stages'][stage]={'nonoverlap_cutoff_tie_rows':ties,'excluded_candidates_mean':float(excluded.mean()),
                                   'excluded_candidates_min':int(excluded.min()),'excluded_candidates_max':int(excluded.max()),'modes':{}}
            for mode,(ids,cosines) in neighborhoods.items():
                assert not (ids==np.arange(len(coords))[:,None]).any()
                distances=np.max(abs(coords[ids]-coords[:,None,:]),axis=-1)
                if mode=='nonoverlap_knn':assert (distances>12).all()
                np.savez_compressed(folder/f'{stage}_{mode}_target_neighbors.npz',indices=ids,cosines=cosines,
                                    center_Chebyshev_distance=distances,excluded_candidate_counts=excluded if mode=='nonoverlap_knn' else np.ones(len(coords),int))
                audit['stages'][stage]['modes'][mode]={}
                for kt in KS:
                    # Chunk gather to keep temporary memory bounded for T256.
                    consensus=np.empty_like(target)
                    for start in range(0,len(target),1024):
                        ix=ids[start:start+1024,:kt]
                        a=target[start:start+1024]+target[ix].mean(1)
                        assert (np.linalg.norm(a,axis=1)>1e-12).all()
                        consensus[start:start+len(ix)]=unit(a)
                    np.save(folder/f'{stage}_{mode}_kt{kt}_features.npy',consensus)
                    outputs=source_outputs(consensus,source,source_ids,sy,centroids)
                    np.savez_compressed(folder/f'{stage}_{mode}_kt{kt}_readouts.npz',**outputs)
                    audit['stages'][stage]['modes'][mode][str(kt)]={'overlapping_neighbor_fraction':float((distances[:,:kt]<=12).mean()),
                        'minimum_neighbor_center_Chebyshev':int(distances[:,:kt].min()),'source_cutoff_tie_rows':int(outputs['source_cutoff_tie_rows'])}
                    print(seed,stage,mode,'kt',kt,'counterfactual features/predictions frozen',flush=True)
        dump(folder/'construction_audit.json',audit)
        provenance['seeds'][str(seed)]={'checkpoint':rec['checkpoint'],'checkpoint_sha256':rec['checkpoint_sha256'],
                                      'strict_cache_reuse':True,'coordinate_authentication':True}
    dump(OUT/'provenance.json',provenance)
    frozen={str(p.relative_to(OUT)):sha(p) for p in OUT.rglob('*') if p.is_file()}
    dump(OUT/'ALL_COUNTERFACTUALS_FROZEN.json',{'files_sha256':frozen,'all_seeds_completed':True,
         'target_GT_opened_by_constructor':False,'features_neighbors_predictions_all_frozen':True,
         'source_GT_used_only_for_readouts':True,'counterfactual_classifier_executed':False,
         'fixture_checks_passed':True,'seconds':time.monotonic()-start})
    print('ALL COUNTERFACTUALS FROZEN; target GT unopened',flush=True)

if __name__=='__main__':main()

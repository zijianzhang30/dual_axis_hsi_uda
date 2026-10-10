"""Construct all source-anchored routers and fixed majority transport without target GT."""
import importlib.util
import json
import os
import shutil
import numpy as np
import torch
from common import HERE,OUT,CACHE,GEOM,FACT,BASE,SEEDS,STAGES,KS,sha,ahash,unit,dump

hp=HERE.parent/'houston_local_support_diagnostic_v1/freeze_neighbors.py'
spec=importlib.util.spec_from_file_location('exact_neighbor_helper',hp)
helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)

def majority(votes,r0):
    count=np.stack([(votes==c).sum(1) for c in range(7)],axis=1)
    tied=(count==count.max(1,keepdims=True)).sum(1)>1
    pred=count.argmax(1);pred[tied]=r0[tied]
    return pred,tied,count

def fixtures():
    votes=np.array([[1,1,2,3],[1,1,2,2],[1,2,3,4],[6,6,6,4]])
    pred,tie,_=majority(votes,np.array([0,5,0,0]))
    assert pred.tolist()==[1,5,0,6] and tie.tolist()==[False,True,True,False]

@torch.inference_mode()
def neighbors(query,source,source_ids):
    q=torch.tensor(query,device='cuda',dtype=torch.float64)
    s=torch.tensor(source,device='cuda',dtype=torch.float64)
    ids,cos=[],[];ties=0
    for i in range(0,len(q),512):
        ix,vs,nt=helper.select_neighbors(q[i:i+512]@s.T)
        ids.append(source_ids[ix]);cos.append(vs);ties+=nt
        if i%(512*40)==0:print('  source search',i,'/',len(q),'dimension',q.shape[1],flush=True)
    return np.concatenate(ids),np.concatenate(cos),ties

def copy_checked(src,dst,h,link=False):
    assert sha(src)==h
    if link:os.link(src.resolve(),dst)
    else:shutil.copyfile(src,dst)
    assert sha(dst)==h

def main():
    fixtures();torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    helper.check_knn();OUT.mkdir(exist_ok=False)
    cg=json.loads((CACHE/'ALL_NEIGHBORS_FROZEN.json').read_text())
    gg=json.loads((GEOM/'ALL_FEATURES_PREDICTIONS_FROZEN.json').read_text())
    fg=json.loads((FACT/'ALL_PREDICTED_FACTORS_FROZEN.json').read_text())
    old=json.loads((FACT/'provenance.json').read_text())
    assert not fg['target_GT_opened'] and not cg['target_GT_opened_by_this_diagnostic']
    assert not gg['target_GT_opened_by_extractor']
    for name in ['source_labels.npy','source_train_mask.npy']:
        copy_checked(FACT/name,OUT/name,fg['files_sha256'][name])
    sy=np.load(OUT/'source_labels.npy');train=np.load(OUT/'source_train_mask.npy');source_ids=np.flatnonzero(train)
    assert len(source_ids)==2393
    prov={'protocol_sha256':sha(HERE/'PROTOCOL.txt'),'code_sha256':{str(p):sha(p) for p in HERE.glob('*') if p.is_file()},
          'historical_metadata':old['historical_metadata'],'V_storage':'authenticated read-only hard links',
          'helper_sha256':sha(hp),'seeds':{}}
    for seed in SEEDS:
        folder=OUT/f'seed_{seed}';folder.mkdir();rec=old['seeds'][str(seed)]
        assert sha(rec['checkpoint'])==rec['checkpoint_sha256']
        for name in ['frozen_head.npz','predicted_factors.npz']+[f'{d}_{key}.npy' for d in ['source','target'] for key in ['rawZ','coords','sample_indices','logits','predictions']]:
            rel=f'seed_{seed}/{name}';copy_checked(FACT/rel,folder/name,fg['files_sha256'][rel])
        for domain in ['source','target']:
            for stage in STAGES:
                rel=f'seed_{seed}/{domain}_{stage}.npy'
                src=GEOM/rel if stage=='V' else CACHE/rel
                hashes=gg['files_sha256'] if stage=='V' else cg['files_sha256']
                copy_checked(src,folder/f'{domain}_{stage}.npy',hashes[rel],link=stage=='V')
            coords=np.load(folder/f'{domain}_coords.npy')
            assert ahash(coords)==old['historical_metadata']['domains'][domain]['coords_sha256']
            assert np.array_equal(coords,np.load(CACHE/f'seed_{seed}/{domain}_coords.npy'))
            assert np.array_equal(coords,np.load(GEOM/f'seed_{seed}/{domain}_coords.npy'))
        config=json.loads((BASE/'houston'/f'C_{seed}/config.json').read_text())
        assert ahash(np.load(folder/'source_coords.npy')[train])==config['indices_sha256']['source']
        z=np.load(folder/'target_rawZ.npy').astype(float);r0=np.load(folder/'target_predictions.npy')
        head=np.load(folder/'frozen_head.npz');w=head['weight'].astype(float);b=head['bias'].astype(float)
        assert np.array_equal((z@w.T+b).argmax(1),r0)
        assert abs(z@w.T+b-np.load(folder/'target_logits.npy')).max()<2e-5
        assert abs(unit(z)-np.load(folder/'target_Z.npy')).max()<2e-6
        np.save(folder/'R0_predictions.npy',r0)
        audit={}
        for stage,centroid_router,knn_router in [('Z','R1','R2'),('T','R3','R4'),('V','R5','R6')]:
            source=unit(np.load(folder/f'source_{stage}.npy').astype(float)[train])
            target=unit(np.load(folder/f'target_{stage}.npy').astype(float))
            cent=unit(np.stack([source[sy[train]==c].mean(0) for c in range(7)]))
            # Chunk centroid products for the full V descriptor.
            score=np.concatenate([target[i:i+1024]@cent.T for i in range(0,len(target),1024)])
            np.savez_compressed(folder/f'{centroid_router}_centroid_scores.npz',source_centroids=cent,target_scores=score)
            np.save(folder/f'{centroid_router}_predictions.npy',score.argmax(1))
            ids,cos,ties=neighbors(target,source,source_ids)
            counts={}
            for k in KS:
                c=np.stack([(sy[ids[:,:k]]==j).sum(1) for j in range(7)],axis=1)
                counts[f'counts_k{k}']=c
                np.save(folder/f'{knn_router}_k{k}_predictions.npy',c.argmax(1))
            np.savez_compressed(folder/f'{knn_router}_neighbors.npz',source_indices=ids,source_cosines=cos,**counts)
            audit[stage]={'dimension':source.shape[1],'source_cutoff_tie_rows':ties}
            del source,target
            torch.cuda.empty_cache()
            print(seed,stage,'routers frozen',flush=True)
        stats=np.load(folder/'predicted_factors.npz');delta=stats['source_means']-stats['pseudo_target_means']
        assert np.array_equal(stats['predicted_routing'],r0)
        baseline=z+delta[r0];logits=baseline@w.T+b
        rel=f'seed_{seed}/D00_rawZ.npy';assert sha(FACT/rel)==fg['files_sha256'][rel]
        assert np.array_equal(baseline,np.load(FACT/rel))
        reference=np.load(FACT/f'seed_{seed}/D00_readouts.npz')
        assert np.array_equal(logits,reference['logits'])
        np.savez_compressed(folder/'D00_readouts.npz',logits=logits,predictions=logits.argmax(1))
        for k in KS:
            votes=np.stack([np.load(folder/'R1_predictions.npy'),np.load(folder/f'R2_k{k}_predictions.npy'),
                            np.load(folder/'R3_predictions.npy'),np.load(folder/f'R4_k{k}_predictions.npy')],axis=1)
            route,tied,count=majority(votes,r0)
            np.savez_compressed(folder/f'Majority_k{k}_routing.npz',predictions=route,tied=tied,votes=votes,class_vote_counts=count)
            transformed=z+delta[route]
            np.save(folder/f'Majority_k{k}_rawZ.npy',transformed)
            logits=transformed@w.T+b
            np.savez_compressed(folder/f'Majority_k{k}_transport_readouts.npz',logits=logits,predictions=logits.argmax(1))
        prov['seeds'][str(seed)]={'checkpoint':rec['checkpoint'],'checkpoint_sha256':rec['checkpoint_sha256'],
            'stage_audit':audit,'D00_stats_features_logits_exact_replay':True}
    dump(OUT/'provenance.json',prov)
    files={str(p.relative_to(OUT)):sha(p) for p in OUT.rglob('*') if p.is_file()}
    dump(OUT/'ALL_ROUTING_TRANSPORT_FROZEN.json',{'files_sha256':files,'all_seeds_completed':True,
        'target_GT_opened_by_constructor':False,'all_routers_majorities_transport_features_predictions_frozen':True,
        'fixed_D00_pseudo_statistics_unchanged':True,'no_training_tuning_followup':True})
    print('All routers and fixed-statistics majority transport frozen; target GT unopened',flush=True)

if __name__=='__main__':main()

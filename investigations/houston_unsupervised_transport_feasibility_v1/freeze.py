"""Construct F0-F3 without reading target GT or prior oracle-derived artifacts."""
import importlib.util
import json
import shutil
import time
import numpy as np
import torch
from common import HERE,OUT,CACHE,SEEDS,KS,sha,ahash,unit,softmax,dump

hp=HERE.parent/'houston_local_support_diagnostic_v1/freeze_neighbors.py'
spec=importlib.util.spec_from_file_location('exact_neighbor_helper',hp)
helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)

@torch.inference_mode()
def knn(q,source,source_ids):
    q=torch.tensor(unit(q),device='cuda',dtype=torch.float64)
    s=torch.tensor(unit(source),device='cuda',dtype=torch.float64)
    ids,cos=[],[];ties=0
    for i in range(0,len(q),512):
        ix,vs,n=helper.select_neighbors(q[i:i+512]@s.T)
        ids.append(source_ids[ix]);cos.append(vs);ties+=n
    return np.concatenate(ids),np.concatenate(cos),ties

def target_means(z,prob,pred,source_means):
    hard_mass=np.bincount(pred,minlength=7);soft_mass=prob.sum(0)
    hard=np.stack([z[pred==c].mean(0) if hard_mass[c] else source_means[c] for c in range(7)])
    weighted=prob.T@z
    soft=np.stack([weighted[c]/soft_mass[c] if soft_mass[c]>0 else source_means[c] for c in range(7)])
    return hard,soft,hard_mass,soft_mass

def fixtures():
    rng=np.random.default_rng(2100);z=rng.normal(size=(30,9));delta=rng.normal(size=(7,9))
    w=rng.normal(size=(7,9));b=rng.normal(size=7)
    h=z[:,None,:]+delta[None,:,:];all_logits=h@w.T+b
    scores=np.diagonal(all_logits,axis1=1,axis2=2)
    bias=np.einsum('cd,cd->c',delta,w)
    assert np.allclose(scores,z@w.T+b+bias,atol=1e-12)
    route=softmax(scores)
    assert np.allclose((route[:,:,None]*h).sum(1),z+route@delta)
    p=softmax(z@w.T+b);hard,soft,hm,sm=target_means(z,p,p.argmax(1),delta)
    for c in range(7):assert np.allclose(soft[c],(p[:,c,None]*z).sum(0)/p[:,c].sum())
    empty_pred=np.zeros(30,int);hard,_,hm,_=target_means(z,p,empty_pred,delta)
    assert np.array_equal(hard[1:],delta[1:]) and np.all(hm[1:]==0)

def main():
    start=time.monotonic();torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    fixtures();helper.check_knn();OUT.mkdir(exist_ok=False)
    gate=json.loads((CACHE/'ALL_INPUTS_FROZEN.json').read_text())
    old=json.loads((CACHE/'provenance.json').read_text())
    assert gate['all_seeds_completed'] and not gate['target_GT_opened_by_input_freezer']
    assert sha(CACHE/'provenance.json')==gate['files_sha256']['provenance.json']
    for rel in ['source_labels.npy','source_train_mask.npy']:
        assert sha(CACHE/rel)==gate['files_sha256'][rel]
        shutil.copyfile(CACHE/rel,OUT/rel)
    sy=np.load(OUT/'source_labels.npy');train=np.load(OUT/'source_train_mask.npy');source_ids=np.flatnonzero(train)
    assert len(source_ids)==2393
    prov={'protocol_sha256':sha(HERE/'PROTOCOL.txt'),'code_sha256':{str(p):sha(p) for p in HERE.glob('*') if p.is_file()},
          'native_input_manifest_sha256':sha(CACHE/'ALL_INPUTS_FROZEN.json'),
          'helper_code_sha256':sha(hp),'historical_metadata':old['historical_metadata'],'seeds':{}}
    for seed in SEEDS:
        folder=OUT/f'seed_{seed}';folder.mkdir();rec=old['seeds'][str(seed)]
        assert sha(rec['checkpoint'])==rec['checkpoint_sha256']
        names=['frozen_head.npz']+[f'{d}_{key}.npy' for d in ['source','target'] for key in ['rawZ','logits','predictions','coords','sample_indices']]
        for name in names:
            rel=f'seed_{seed}/{name}';assert sha(CACHE/rel)==gate['files_sha256'][rel]
            shutil.copyfile(CACHE/rel,folder/name);assert sha(folder/name)==gate['files_sha256'][rel]
        source=np.load(folder/'source_rawZ.npy').astype(np.float64)[train]
        z=np.load(folder/'target_rawZ.npy').astype(np.float64)
        head=np.load(folder/'frozen_head.npz');w=head['weight'].astype(np.float64);b=head['bias'].astype(np.float64)
        native=np.load(folder/'target_logits.npy').astype(np.float64);pred=np.load(folder/'target_predictions.npy')
        computed=z@w.T+b;error=float(abs(computed-native).max());assert error<2e-5 and np.array_equal(computed.argmax(1),pred)
        source_means=np.stack([source[sy[train]==c].mean(0) for c in range(7)])
        prob=softmax(native);hard,soft,hm,sm=target_means(z,prob,pred,source_means)
        dh=source_means-hard;ds=source_means-soft
        bias=np.einsum('cd,cd->c',ds,w)
        hypothesis_scores=computed+bias
        route=softmax(hypothesis_scores)
        np.savez_compressed(folder/'fixed_unlabeled_statistics.npz',source_means=source_means,
            hard_target_means=hard,soft_target_means=soft,hard_assignment_counts=hm,soft_assignment_mass=sm,
            hard_translation=dh,soft_translation=ds,F3_class_logit_bias=bias)
        np.save(folder/'original_posterior.npy',prob)
        np.save(folder/'F1_hard_routing.npy',pred)
        np.save(folder/'F2_soft_routing.npy',prob)
        np.save(folder/'F3_soft_routing.npy',route)
        np.save(folder/'F3_hypothesis_scores.npy',hypothesis_scores)
        hypotheses=np.lib.format.open_memmap(folder/'F3_hypothesis_features.npy',mode='w+',dtype=np.float64,shape=(len(z),7,64))
        hypothesis_logits=np.lib.format.open_memmap(folder/'F3_hypothesis_logits.npy',mode='w+',dtype=np.float64,shape=(len(z),7,7))
        max_algebra_error=0.0
        for i in range(0,len(z),1024):
            h=z[i:i+1024,None,:]+ds[None,:,:]
            logits=h@w.T+b
            hypotheses[i:i+len(h)]=h;hypothesis_logits[i:i+len(h)]=logits
            err=float(abs(np.diagonal(logits,axis1=1,axis2=2)-hypothesis_scores[i:i+len(h)]).max())
            max_algebra_error=max(max_algebra_error,err)
        hypotheses.flush();hypothesis_logits.flush();assert max_algebra_error<1e-10
        del hypotheses,hypothesis_logits
        features={'F0':z,'F1':z+dh[pred],'F2':z+prob@ds,'F3':z+route@ds}
        for group,x in features.items():
            assert np.isfinite(x).all() and (np.linalg.norm(x,axis=1)>1e-12).all()
            np.save(folder/f'{group}_rawZ.npy',x)
            logits=x@w.T+b;predictions=logits.argmax(1)
            if group=='F0':assert np.array_equal(predictions,pred)
            ids,cos,ties=knn(x,source,source_ids)
            readout={'classifier_logits':logits,'classifier_predictions':predictions,'source_indices':ids,'source_cosines':cos}
            for k in KS:
                counts=np.stack([(sy[ids[:,:k]]==c).sum(1) for c in range(7)],axis=1)
                readout[f'source_counts_k{k}']=counts;readout[f'source_knn_predictions_k{k}']=counts.argmax(1)
            np.savez_compressed(folder/f'{group}_readouts.npz',**readout)
            print(seed,group,'features/routing/readouts frozen',flush=True)
        prov['seeds'][str(seed)]={'checkpoint':rec['checkpoint'],'checkpoint_sha256':rec['checkpoint_sha256'],
            'native_classifier_logit_max_error':error,'native_classifier_predictions_exact':True,
            'hard_zero_assignment_classes_1based':(np.flatnonzero(hm==0)+1).tolist(),
            'soft_zero_assignment_classes_1based':(np.flatnonzero(sm==0)+1).tolist(),
            'F3_diagonal_logit_algebra_max_error':max_algebra_error}
    dump(OUT/'provenance.json',prov)
    files={str(p.relative_to(OUT)):sha(p) for p in OUT.rglob('*') if p.is_file()}
    dump(OUT/'ALL_UNSUPERVISED_FROZEN.json',{'files_sha256':files,'all_seeds_completed':True,
        'target_GT_opened_by_constructor':False,'oracle_G2_artifacts_opened_by_constructor':False,
        'target_statistics_routing_features_predictions_all_frozen':True,'no_training_backward_tuning_followup':True,
        'fixtures_passed':True,'seconds':time.monotonic()-start})
    print('ALL F0-F3 FROZEN; target GT and oracle G2 unopened',flush=True)

if __name__=='__main__':main()

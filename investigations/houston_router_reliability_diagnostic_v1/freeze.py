"""Freeze native confidence and agreement without target labels."""
import json
import shutil
import numpy as np
from common import HERE,OUT,BASE,SEEDS,KS,NAMES,STAGE,FAMILIES,sha,ahash,dump,signals

def copy(src,dst,gate):
    rel=str(src.relative_to(BASE));assert sha(src)==gate['files_sha256'][rel]
    shutil.copyfile(src,dst);assert sha(dst)==gate['files_sha256'][rel]

def main():
    OUT.mkdir(exist_ok=False)
    gate=json.loads((BASE/'ALL_ROUTING_TRANSPORT_FROZEN.json').read_text())
    assert gate['all_seeds_completed'] and not gate['target_GT_opened_by_constructor']
    old=json.loads((BASE/'provenance.json').read_text())
    prov={'parent_gate_sha256':sha(BASE/'ALL_ROUTING_TRANSPORT_FROZEN.json'),
          'historical_metadata':old['historical_metadata'],'seeds':old['seeds'],
          'code_sha256':{str(p):sha(p) for p in HERE.glob('*') if p.is_file()}}
    copy(BASE/'source_labels.npy',OUT/'source_labels.npy',gate)
    sy=np.load(OUT/'source_labels.npy')
    for seed in SEEDS:
        f=OUT/f'seed_{seed}';f.mkdir();src=BASE/f'seed_{seed}'
        rec=old['seeds'][str(seed)];assert sha(rec['checkpoint'])==rec['checkpoint_sha256']
        names=['R0_predictions.npy','target_logits.npy','target_coords.npy','target_sample_indices.npy']
        names += [f'{r}_predictions.npy' for r in ['R1','R3','R5']]
        names += [f'{r}_centroid_scores.npz' for r in ['R1','R3','R5']]
        names += [f'{r}_neighbors.npz' for r in ['R2','R4','R6']]
        names += [f'{r}_k{k}_predictions.npy' for r in ['R2','R4','R6'] for k in KS]
        for name in names:copy(src/name,f/name,gate)
        logits=np.load(f/'target_logits.npy').astype(float);r0=np.load(f/'R0_predictions.npy')
        assert np.array_equal(r0,logits.argmax(1));assert len(r0)==52901
        assert ahash(np.load(f/'target_coords.npy'))==old['historical_metadata']['domains']['target']['coords_sha256']
        top=np.sort(logits,axis=1)[:,-2:];r0_margin=top[:,1]-top[:,0]
        shift=logits-logits.max(1,keepdims=True)
        logp=shift-np.log(np.exp(shift).sum(1,keepdims=True));entropy=-(np.exp(logp)*logp).sum(1)
        for k in KS:
            a={'R0__predictions':r0,'R0__logit_margin':r0_margin,'R0__entropy':entropy}
            for r in ['R1','R3','R5']:
                z=np.load(f/f'{r}_centroid_scores.npz')['target_scores'];top=np.sort(z,axis=1)[:,-2:]
                pred=np.load(f/f'{r}_predictions.npy');assert np.array_equal(pred,z.argmax(1))
                a[r+'__predictions']=pred;a[r+'__cosine_margin']=top[:,1]-top[:,0]
            for r in ['R2','R4','R6']:
                z=np.load(f/f'{r}_neighbors.npz');counts=z[f'counts_k{k}']
                check=np.stack([(sy[z['source_indices'][:,:k]]==c).sum(1) for c in range(7)],axis=1)
                assert np.array_equal(counts,check)
                pred=np.load(f/f'{r}_k{k}_predictions.npy');assert np.array_equal(pred,counts.argmax(1))
                top=np.sort(counts,axis=1)[:,-2:]
                a[r+'__predictions']=pred;a[r+'__vote_margin']=(top[:,1]-top[:,0])/k
                a[r+'__neighbor_agreement']=top[:,1]/k
            pred=np.stack([a[r+'__predictions'] for r in NAMES],axis=1)
            agree=pred[:,:,None]==pred[:,None,:]
            for j,r in enumerate(NAMES):
                a[r+'__agreement_all']=(agree[:,j,:].sum(1)-1)/6
                other=[s for s in ['V','T','Z'] if s!=STAGE[r]]
                a[r+'__agreement_other_stages']=np.mean([agree[:,j,[i for i,q in enumerate(NAMES) if STAGE[q]==s]].mean(1) for s in other],axis=0)
            for family,label in zip(FAMILIES,['centroid','knn']):
                ids=[NAMES.index(r) for r in family]
                a[label+'__all_stages_agree']=(pred[:,ids]==pred[:,ids[:1]]).all(1).astype(float)
                for r in family:
                    j=NAMES.index(r);a[r+'__matched_stage_agreement']=agree[:,j,[i for i in ids if i!=j]].mean(1)
            a['pairwise_agreement']=agree
            assert np.array_equal(a['R0__logit_margin'],np.sort(logits,axis=1)[:,-1]-np.sort(logits,axis=1)[:,-2])
            np.savez_compressed(f/f'confidence_k{k}.npz',**a)
            with np.load(f/f'confidence_k{k}.npz') as frozen:
                quant={}
                for r in NAMES[1:]:
                    mask=a[r+'__predictions']!=r0;quant[r+'__disagreement']=mask
                    for name,(score,orientation) in signals(frozen,r).items():
                        assert np.isfinite(score).all()
                        edges=np.quantile(score[mask],np.arange(1,10)/10) if mask.any() else np.zeros(9)
                        quant[r+'__'+name+'__edges']=edges
                        quant[r+'__'+name+'__bins']=np.searchsorted(edges,score,side='right').astype(np.int8)
                np.savez_compressed(f/f'disagreement_deciles_k{k}.npz',**quant)
            print(seed,'k',k,'predictions/confidence/agreement/deciles frozen',flush=True)
    dump(OUT/'provenance.json',prov)
    dump(OUT/'ALL_ROUTER_CONFIDENCE_FROZEN.json',{'all_seeds_completed':True,'target_GT_opened':False,
         'no_training_tuning_followup':True,'files_sha256':{str(p.relative_to(OUT)):sha(p) for p in OUT.rglob('*') if p.is_file()}})
    print('All unlabeled router confidence frozen; target GT unopened.',flush=True)

if __name__=='__main__':main()

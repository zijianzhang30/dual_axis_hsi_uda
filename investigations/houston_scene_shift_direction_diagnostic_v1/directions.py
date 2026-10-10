"""Unlabeled domain statistics and basis construction; no labels opened."""
import json
import shutil
import numpy as np
from common import HERE,OUT,CACHE,SEEDS,CONFIG,sha,dump,unit,canonical

def copy(src,dst,gate):
    rel=str(src.relative_to(CACHE));assert sha(src)==gate['files_sha256'][rel]
    shutil.copyfile(src,dst);assert sha(dst)==gate['files_sha256'][rel]

def main():
    OUT.mkdir(exist_ok=False)
    old=json.loads((CACHE/'provenance.json').read_text())
    gate=json.loads((CACHE/'ALL_PREDICTED_FACTORS_FROZEN.json').read_text());assert not gate['target_GT_opened']
    prov={'fixed_config':CONFIG,'historical_metadata':old['historical_metadata'],'seeds':old['seeds'],
        'parent_preGT_gate_sha256':sha(CACHE/'ALL_PREDICTED_FACTORS_FROZEN.json'),
        'code_sha256':{str(p):sha(p) for p in HERE.glob('*') if p.is_file()}}
    for seed in SEEDS:
        f=OUT/f'seed_{seed}';f.mkdir();rec=old['seeds'][str(seed)]
        assert sha(rec['checkpoint'])==rec['checkpoint_sha256']
        for name in ['source_rawZ.npy','target_rawZ.npy','source_coords.npy','target_coords.npy','source_sample_indices.npy','target_sample_indices.npy','source_logits.npy','source_predictions.npy','frozen_head.npz']:
            copy(CACHE/f'seed_{seed}/{name}',f/name,gate)
    dump(OUT/'provenance.json',prov)
    dump(OUT/'ALL_UNLABELED_INPUTS_FROZEN.json',{'source_GT_opened':False,'target_GT_opened':False,
        'files_sha256':{str(p.relative_to(OUT)):sha(p) for p in OUT.rglob('*') if p.is_file()}})
    for seed in SEEDS:
        f=OUT/f'seed_{seed}';s=np.load(f/'source_rawZ.npy').astype(float);t=np.load(f/'target_rawZ.npy').astype(float)
        sm=s.mean(0);tm=t.mean(0);mean=tm-sm;assert np.linalg.norm(mean)>1e-12
        sc=s-sm;tc=t-tm;cs=sc.T@sc/(len(s)-1);ct=tc.T@tc/(len(t)-1);discrepancy=ct-cs
        u,sv,vh=np.linalg.svd(discrepancy,full_matrices=False);basis=canonical(u[:,:4]);assert sv[3]>1e-12
        assert np.max(abs(basis.T@basis-np.eye(4)))<1e-12
        np.savez_compressed(f/'unlabeled_domain_directions.npz',source_mean=sm,target_mean=tm,
            mean_shift=mean,mean_basis=unit(mean)[...,None],covariance_basis=basis,
            source_covariance=cs,target_covariance=ct,centered_covariance_discrepancy=discrepancy,
            singular_values=sv,signed_eigenvalues=np.linalg.eigvalsh(discrepancy),source_norms=np.linalg.norm(s,axis=1))
        head=np.load(f/'frozen_head.npz');scores=s@head['weight'].astype(float).T+head['bias'].astype(float)
        assert np.array_equal(scores.argmax(1),np.load(f/'source_predictions.npy'))
        assert np.max(abs(scores-np.load(f/'source_logits.npy')))<2e-5
        np.save(f/'source_replayed_logits.npy',scores)
        print(seed,'unlabeled mean and covariance-rank4 directions frozen; mean norm',np.linalg.norm(mean),flush=True)
    dump(OUT/'ALL_DOMAIN_DIRECTIONS_FROZEN.json',{'source_GT_opened_by_direction_constructor':False,'target_GT_opened':False,
        'all_seeds_completed':True,'files_sha256':{str(p.relative_to(OUT)):sha(p) for p in OUT.rglob('*') if p.is_file()}})
    print('All domain directions hash-frozen; source/target GT unopened.',flush=True)

if __name__=='__main__':main()

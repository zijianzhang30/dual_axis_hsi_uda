"""Project/freeze class-orthogonal bases and attacks; no target GT or training."""
import json
import shutil
import numpy as np
from common import HERE,BASE,OUT,SEEDS,EPS,PROBES,REPS,CONFIG,sha,ahash,dump,exact_attack,canonical,projected_basis,csvwrite

def main():
    OUT.mkdir(exist_ok=False)
    parent=json.loads((BASE/'ALL_SOURCE_PERTURBATIONS_FROZEN.json').read_text())
    originalprov=json.loads((BASE/'provenance.json').read_text())
    assert parent['all_seeds_probes_eps_controls_completed'] and not parent['target_GT_opened']
    prov={'fixed_config':CONFIG,'historical_metadata':originalprov['historical_metadata'],'seeds':originalprov['seeds'],
        'parent_freeze_sha256':sha(BASE/'ALL_SOURCE_PERTURBATIONS_FROZEN.json'),
        'parent_final_manifest_sha256':sha(BASE/'FINAL_DELIVERY.json'),
        'parent_math_sha256':sha(HERE.parent/'common.py'),
        'code_sha256':{str(p):sha(p) for p in HERE.glob('*') if p.is_file()}}
    def copy(rel,dst):
        assert sha(BASE/rel)==parent['files_sha256'][rel]
        shutil.copyfile(BASE/rel,dst);assert sha(dst)==parent['files_sha256'][rel]
    for name in ['source_labels.npy','source_train_mask.npy']:copy(name,OUT/name)
    y=np.load(OUT/'source_labels.npy')
    assert ahash(y)==prov['historical_metadata']['domains']['source']['labels_sha256']
    energy_rows=[];rank_rows=[];status={}
    for seed in SEEDS:
        f=OUT/f'seed_{seed}';f.mkdir();rec=prov['seeds'][str(seed)]
        assert sha(rec['checkpoint'])==rec['checkpoint_sha256']
        for name in ['source_rawZ.npy','source_replayed_logits.npy','source_coords.npy','source_sample_indices.npy','frozen_head.npz','unlabeled_domain_directions.npz']:
            copy(f'seed_{seed}/{name}',f/name)
        z=np.load(f/'source_rawZ.npy').astype(float);globalmean=z.mean(0)
        classmean=np.stack([z[y==c].mean(0) for c in range(7)])
        center=(classmean-globalmean).T
        u,s,vh=np.linalg.svd(center,full_matrices=False)
        tol=np.finfo(float).eps*max(center.shape)*s[0];rank=min(6,int((s>tol).sum()))
        cb=canonical(u[:,:rank]);assert rank>0 and rank<=6
        assert np.max(abs(cb.T@cb-np.eye(rank)))<1e-12
        counts=np.bincount(y,minlength=7);weighted=(classmean-globalmean)*np.sqrt(counts/len(y))[:,None]
        residual=z-classmean[y];head=np.load(f/'frozen_head.npz');w=head['weight'].astype(float);cw=w-w.mean(0)
        np.savez_compressed(f/'source_class_subspace.npz',class_means=classmean,global_mean=globalmean,
            class_basis=cb,class_rank=rank,class_singular_values=s,rank_tolerance=tol,class_counts=counts)
        ds=np.load(f/'unlabeled_domain_directions.npz');status[str(seed)]={}
        def energy(probe,method,rep,phase,b):
            energy_rows.append({'seed':seed,'probe':probe,'method':method,'rep':rep,'phase':phase,'rank':b.shape[1],
                'class_between_energy_fraction':float(np.square(weighted@b).sum()/np.square(weighted).sum()),
                'within_class_energy_fraction':float(np.square(residual@b).sum()/np.square(residual).sum()),
                'centered_classifier_weight_energy_fraction':float(np.square(cw@b).sum()/np.square(cw).sum()),
                'max_class_basis_inner_product':float(np.max(abs(cb.T@b)))})
        for probe in PROBES:
            p=f/probe;p.mkdir();b=ds['mean_basis'] if probe=='global_mean' else ds['covariance_basis']
            new,projected,sv,nrank=projected_basis(b,cb)
            valid=new is not None;status[str(seed)][probe]={'valid_same_rank':valid,'class_rank':rank,'original_rank':b.shape[1],'projected_numeric_rank':nrank}
            rank_rows.append({'seed':seed,'probe':probe,'class_rank':rank,'original_rank':b.shape[1],
                'projected_numeric_rank':nrank,'valid_same_rank':valid,
                'retained_domain_basis_energy_fraction':float(np.square(projected).sum()/b.shape[1]),
                'residual_smallest_singular_value':float(sv[-1])})
            np.savez_compressed(p/'domain_projection.npz',original_basis=b,projected_columns=projected,
                residual_singular_values=sv,projected_basis=np.empty((64,0)) if new is None else new)
            energy(probe,'domain',0,'before',b)
            if not valid:continue
            energy(probe,'domain',0,'class_orthogonal',new)
            for rep in range(REPS):
                rel=f'seed_{seed}/{probe}/fixed_random_rep{rep:02d}.npz';assert sha(BASE/rel)==parent['files_sha256'][rel]
                rb=np.load(BASE/rel)['random_basis'];rn,res,rsv,rrank=projected_basis(rb,cb)
                assert rn is not None and rn.shape==new.shape
                np.savez_compressed(p/f'random_projection_rep{rep:02d}.npz',original_basis=rb,projected_basis=rn,residual_singular_values=rsv)
                energy(probe,'random',rep,'before',rb);energy(probe,'random',rep,'class_orthogonal',rn)
        print(seed,'source class rank',rank,'domain residual rank status',status[str(seed)],flush=True)
    csvwrite('class_projection_rank_audit.csv',rank_rows);csvwrite('class_energy_per_repetition.csv',energy_rows)
    dump(OUT/'provenance.json',prov);dump(OUT/'projected_rank_status.json',status)
    dump(OUT/'ALL_CLASS_ORTHOGONAL_BASES_FROZEN.json',{'target_GT_opened':False,'source_GT_used_for_class_subspace':True,
        'files_sha256':{str(p.relative_to(OUT)):sha(p) for p in OUT.rglob('*') if p.is_file()}})
    audits=[]
    for seed in SEEDS:
        f=OUT/f'seed_{seed}';z=np.load(f/'source_rawZ.npy').astype(float);logits=np.load(f/'source_replayed_logits.npy')
        head=np.load(f/'frozen_head.npz');w=head['weight'].astype(float);cb=np.load(f/'source_class_subspace.npz')['class_basis']
        for probe in PROBES:
            if not status[str(seed)][probe]['valid_same_rank']:continue
            p=f/probe
            cases=[('CO_domain',0,np.load(p/'domain_projection.npz')['projected_basis'])]
            cases.extend(('CO_random_adversarial',rep,np.load(p/f'random_projection_rep{rep:02d}.npz')['projected_basis']) for rep in range(REPS))
            for method,rep,b in cases:
                for eps in EPS:
                    delta,scores,c=exact_attack(z,logits,w,y,b,eps)
                    ne=float(np.max(abs(np.linalg.norm(delta,axis=1)-eps*np.linalg.norm(z,axis=1))))
                    ce=float(np.max(abs(delta@cb)));se=float(np.max(abs(delta-(delta@b)@b.T)))
                    assert max(ne,ce,se)<1e-10
                    np.savez_compressed(p/f'{method}_rep{rep:02d}_eps{eps:.2f}.npz',delta=delta,logits=scores,predictions=scores.argmax(1),competitor=c)
                    audits.append({'seed':seed,'probe':probe,'method':method,'rep':rep,'epsilon':eps,
                        'max_L2_error':ne,'max_class_inner_product':ce,'max_attack_span_error':se})
            print(seed,probe,'all class-orthogonal domain/random attacks frozen',flush=True)
    csvwrite('perturbation_constraint_audit.csv',audits)
    dump(OUT/'ALL_CLASS_ORTHOGONAL_PERTURBATIONS_FROZEN.json',{'all_seeds_completed':True,
        'target_GT_opened':False,'source_GT_used_only_for_class_subspace_and_adversarial_attack':True,
        'no_training_backprop_tuning_followup':True,'same_rank_verified_or_degenerate_reported':True,
        'files_sha256':{str(p.relative_to(OUT)):sha(p) for p in OUT.rglob('*') if p.is_file()}})
    print('Class-orthogonal directions/deltas/logits/predictions hash-frozen; no target GT.',flush=True)

if __name__=='__main__':main()

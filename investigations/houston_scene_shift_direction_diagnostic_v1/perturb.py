"""Source-label-guided closed-form feature perturbations; no training or target GT."""
import json
import shutil
import numpy as np
from common import OUT,CACHE,SEEDS,EPS,PROBES,REPS,sha,ahash,dump,unit,canonical,exact_attack,csvwrite

def main():
    gate=json.loads((OUT/'ALL_DOMAIN_DIRECTIONS_FROZEN.json').read_text())
    assert not gate['source_GT_opened_by_direction_constructor'] and not gate['target_GT_opened']
    for rel,h in gate['files_sha256'].items():assert sha(OUT/rel)==h
    parent=json.loads((CACHE/'ALL_PREDICTED_FACTORS_FROZEN.json').read_text())
    for name in ['source_labels.npy','source_train_mask.npy']:
        assert sha(CACHE/name)==parent['files_sha256'][name]
        shutil.copyfile(CACHE/name,OUT/name);assert sha(OUT/name)==parent['files_sha256'][name]
    y=np.load(OUT/'source_labels.npy');train=np.load(OUT/'source_train_mask.npy')
    prov=json.loads((OUT/'provenance.json').read_text())
    assert ahash(y)==prov['historical_metadata']['domains']['source']['labels_sha256']
    assert train.sum()==2393 and (~train).sum()==127
    norm_audits=[];semantic=[]
    for seed in SEEDS:
        f=OUT/f'seed_{seed}';z=np.load(f/'source_rawZ.npy').astype(float);logits=np.load(f/'source_replayed_logits.npy')
        head=np.load(f/'frozen_head.npz');w=head['weight'].astype(float);norm=np.linalg.norm(z,axis=1)
        dirs=np.load(f/'unlabeled_domain_directions.npz')
        mean=z.mean(0);classmean=np.stack([z[y==c].mean(0) for c in range(7)])
        residual=z-classmean[y];within=residual.T@residual/len(z)
        offset=classmean[y]-mean;between=offset.T@offset/len(z)
        centeredw=w-w.mean(0);wenergy=float((centeredw**2).sum())
        for probe in PROBES:
            p=f/probe;p.mkdir();basis=dirs['mean_basis'] if probe=='global_mean' else dirs['covariance_basis'];r=basis.shape[1]
            def audit_sem(label,rep,b):
                semantic.append({'seed':seed,'probe':probe,'basis_type':label,'rep':rep,'rank':b.shape[1],
                    'source_between_class_energy_fraction':float(np.trace(b.T@between@b)/np.trace(between)),
                    'source_within_class_energy_fraction':float(np.trace(b.T@within@b)/np.trace(within)),
                    'centered_classifier_weight_energy_fraction':float(np.square(centeredw@b).sum()/wenergy),
                    'isotropic_expected_weight_fraction':b.shape[1]/64})
            audit_sem('domain',0,basis)
            np.save(p/'domain_basis.npy',basis)
            def save(method,rep,eps,delta,scores,competitor=None):
                rho=eps*norm;normerr=float(np.max(abs(np.linalg.norm(delta,axis=1)-rho)))
                assert normerr<1e-10
                ortherr=float(np.max(abs(delta@basis))) if method=='D_orthogonal' else None
                if ortherr is not None:assert ortherr<1e-10
                spanerr=float(np.max(abs(delta-(delta@basis)@basis.T))) if method in ['D_domain','D_domain_random','D_mean_forward'] else None
                if spanerr is not None:assert spanerr<1e-10
                np.savez_compressed(p/f'{method}_rep{rep:02d}_eps{eps:.2f}.npz',delta=delta,logits=scores,predictions=scores.argmax(1),
                    competitor=np.array([],int) if competitor is None else competitor)
                norm_audits.append({'seed':seed,'probe':probe,'method':method,'rep':rep,'epsilon':eps,
                    'max_L2_norm_error':normerr,'max_orthogonality_error':ortherr,'max_domain_span_error':spanerr,
                    'mean_perturbation_L2':float(rho.mean())})
            for eps in EPS:
                delta,scores,c=exact_attack(z,logits,w,y,basis,eps);save('D_domain',0,eps,delta,scores,c)
                if probe=='global_mean':
                    delta=eps*norm[:,None]*basis[:,0];save('D_mean_forward',0,eps,delta,logits+delta@w.T)
            for rep in range(REPS):
                # Same underlying per-origin draws across probes and model seeds.
                rng=np.random.default_rng(20261012+rep);random=unit(rng.normal(size=z.shape))
                orth=unit(random-(random@basis)@basis.T)
                inspan=unit((random@basis)@basis.T)
                rb=canonical(np.linalg.qr(np.random.default_rng(20271012+rep).normal(size=(64,4)))[0][:,:r])
                audit_sem('random_subspace',rep,rb)
                np.savez_compressed(p/f'fixed_random_rep{rep:02d}.npz',random_unit=random,orthogonal_unit=orth,domain_random_unit=inspan,random_basis=rb)
                for eps in EPS:
                    for method,direction in [('D_random',random),('D_orthogonal',orth),('D_domain_random',inspan)]:
                        delta=eps*norm[:,None]*direction;save(method,rep,eps,delta,logits+delta@w.T)
                    delta,scores,c=exact_attack(z,logits,w,y,rb,eps);save('D_random_subspace_adversarial',rep,eps,delta,scores,c)
            print(seed,probe,'all eps/16 fixed repetitions frozen',flush=True)
    csvwrite('perturbation_norm_audit.csv',norm_audits);csvwrite('source_semantic_direction_audit.csv',semantic)
    dump(OUT/'ALL_SOURCE_PERTURBATIONS_FROZEN.json',{'all_seeds_probes_eps_controls_completed':True,
        'source_GT_used_only_for_requested_adversarial_direction_and_postdirection_semantic_audit':True,
        'target_GT_opened':False,'no_training_backprop_tuning_followup':True,
        'files_sha256':{str(p.relative_to(OUT)):sha(p) for p in OUT.rglob('*') if p.is_file()}})
    print('All source perturbations/logits/predictions hash-frozen; no target GT.',flush=True)

if __name__=='__main__':main()

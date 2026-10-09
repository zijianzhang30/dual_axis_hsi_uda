"""Lock diagnostic samples/definitions before reading any diagnostic result."""
import json
import shutil
from pathlib import Path
import numpy as np
import diagnostics as d

def main():
    assert not d.OUT.exists(),'Refuse overwrite';d.OUT.mkdir()
    for p in Path(__file__).parent.iterdir():
        if p.is_file():shutil.copy2(p,d.OUT/p.name)
    d.c.verify_files()
    with np.load(d.c.P2/'inputs/source_selection_2100.npz') as z:
        source_coords=z['coords'];source_labels=z['labels']
    target_coords=np.load(d.c.P2/'inputs/target_training_coords.npy')
    target_labels=np.load(d.c.P2/'inputs/evaluation_labels_OFFLINE_ONLY.npy')
    # Selection is uniform by index; GT is inspected only after indices are fixed.
    indices=np.sort(np.random.RandomState(731909).choice(len(target_coords),d.TARGET_N,replace=False))
    np.savez_compressed(d.OUT/'fixed_samples.npz',source_coords=source_coords,source_labels=source_labels,
        target_indices=indices,target_coords=target_coords[indices],target_labels=target_labels[indices])
    cfg={'study':'Pavia C+CE+LMMD read-only class-aware alignment diagnosis',
         'seeds':d.SEEDS,'checkpoints':d.PHASES,'comparison':'paired C_CE epoch100, not a new training run',
         'source_fixed_pool':'all1260 seed2100 source training selections, fixed for all3 seeds; centroid statistics,not source-val',
         'target_fixed_pool':'8192 uniform target indices,RandomState731909; no error-based or GT-stratified sampling',
         'reliability':'eval running-statistics fixed8192; population39348 at100 from saved logits; actual JointBN/dropout38batches1216 plus checkpoint-eval on identical augmented patches at each phase',
         'probability_audit':'precision/recall,NLL,ECE10,Brier,softclass-membership purity,hard-presence class activation',
         'alignment_space':'exact native normalized classifier-input CLS64; raw L2 primary,cosine auxiliary',
         'geometry':'GT class centers,nearest-wrong margin,sample negative margin,soft-vs-GT center bias,scatter/effective rank',
         'multimodality':'fixed2means on CE100 target class features,n_init10,seeds731911+k; track same sample groups in LMMD features; descriptive only',
         'gradient_batches':d.GRADIENT_BATCHES,'gradient_protocol':'frozen epoch-end weights,actual corresponding epoch inputs and reconstructed historical dropout,not actual historical per-update gradients',
         'gradient_scope':'CE vs LMMD globally/by block; class4/7 CE-vs-alignment; weightednormratio; first-order dot ignores momentum/WD',
         'oracle':'GT-only native weight counterfactual,feature/parameter gradients for diagnosis; never optimized or passed to main CE/LMMD',
         'model_state':'no optimizer/no step/no backward accumulation; BN buffers restored after every query; complete state hash checked',
         'no_target_best':True,'no_retraining':True,'no_loss_modification':True,'no_auto_method_design':True,
         'target_counts_fixed_pool':np.bincount(target_labels[indices],minlength=7).tolist()}
    d.write(d.OUT/'frozen_plan.json',cfg)
    hashes=dict(json.loads((d.BASE/'provenance.json').read_text())['sha256'])
    for p in d.OUT.iterdir():
        if p.is_file():hashes[str(p)]=d.sha(p)
    for seed in d.SEEDS:
        for folder,epochs in [(d.BASE/('C_LMMD_%d'%seed),d.PHASES),(d.c.run_folder('C_CE',seed),(100,))]:
            m=json.loads((folder/'frozen_manifest.json').read_text())
            for ep in epochs:
                item=next(v for v in m['checkpoints'] if v['epoch']==ep)
                assert d.sha(item['path'])==item['sha256'];hashes[item['path']]=item['sha256']
            for name in ('history.jsonl','config.json','frozen_manifest.json'):hashes[str(folder/name)]=d.sha(folder/name)
            item=m['prediction'];assert d.sha(item['path'])==item['sha256'];hashes[item['path']]=item['sha256']
    d.write(d.OUT/'provenance.json',{'sha256':hashes,'all_original_artifacts_read_only':True})
    link=Path('/home/zhangzj26/dual_axis_hsi_uda/results')/d.OUT.name
    assert not link.exists();link.symlink_to(d.OUT,target_is_directory=True)
    print('Locked read-only plan,samples,and all checkpoint/input hashes: '+str(d.OUT),flush=True)

if __name__=='__main__':main()

"""Lock both stages before training; no scientific edits to previous studies."""
import json
import shutil
from pathlib import Path
import common as c

def main():
    assert not c.OUT.exists(),'Refuse overwrite';c.OUT.mkdir()
    for p in Path(__file__).parent.iterdir():
        if p.is_file():shutil.copy2(p,c.OUT/p.name)
    cfg=json.loads((c.BASE/'frozen_config.json').read_text())
    cfg.update(study='Pavia LMMD-only and original/relation tokenizer control',arms=c.ARMS,
        new_arms=c.NEW_ARMS,seeds=c.SEEDS,epochs=100,primary='fixed_epoch100',
        objective='source CE + 0.3*(2/(1+exp(-10*epoch/100))-1)*official LMMD',
        weights='unchanged from prior Full/noSource; not unweighted LMMD or a coefficient sweep',
        source_contrastive=False,target_contrastive=False,
        alignment_feature='normalized classifier-input CLS64, identical placement for A/C',
        A='original unmodified conv_a spatial softmax; Full Joint BN; original3self blocks',
        C='frozen nonlinear normalized center-relative scorer; otherwise identical to A',
        projection_heads='same two training heads retained; unused by CE+LMMD; no gradients or SGD decay',
        stages='C_LMMD3seeds first, then A_LMMD3seeds regardless of target metrics; score only after both stages',
        comparators='reuse exact C_CE and CE+LMMD+target contrastive from paired prior studies',
        initializations='A/C common model tensors and head initialization identical; C scorer independent RNG',
        no_target_best=True,no_target_tuning=True,no_auto_followups=True,
        interpretation='known LMMD is diagnostic, not a claimed second innovation')
    cfg['AD_objective']=cfg['objective']
    cfg['O_network']='Full Joint BN + selected A/C tokenizer + original3self blocks'
    cfg.pop('M_network',None)
    c.dump(c.OUT/'frozen_config.json',cfg);c.dump(c.OUT/'reused_controls.json',c.audit_reuse())
    hashes=dict(json.loads((c.BASE/'provenance.json').read_text())['sha256'])
    for p in c.OUT.iterdir():
        if p.is_file():hashes[str(p)]=c.sha(p)
    for arm in ('C_CE','C_LMMD_CT'):
        for seed in c.SEEDS:
            folder=c.run_folder(arm,seed)
            for name in ('config.json','history.jsonl','frozen_manifest.json'):hashes[str(folder/name)]=c.sha(folder/name)
    c.dump(c.OUT/'provenance.json',{'sha256':hashes,'immutable_matrix':str(c.BASE)})
    link=Path('/home/zhangzj26/dual_axis_hsi_uda/results')/c.OUT.name
    assert not link.exists();link.symlink_to(c.OUT,target_is_directory=True)
    print('Locked6newruns and6reused controls: '+str(c.OUT),flush=True)

if __name__=='__main__':main()

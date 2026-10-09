"""Lock 3 removals x 3 seeds before any training; never alter existing artifacts."""
import json
import shutil
from pathlib import Path
import common as c

def main():
    assert not c.OUT.exists(),'Refuse overwrite'
    c.OUT.mkdir()
    for p in Path(__file__).parent.iterdir():
        if p.is_file():shutil.copy2(p,c.OUT/p.name)
    config=json.loads((c.BASE/'frozen_config.json').read_text())
    config.update(study='Ours official objective leave-one-out',arms=c.ARMS,seeds=c.SEEDS,
           reused=['CE','Full'],new_runs=9,network='O',
           formulas={'CE':'CE','Full':'CE + .3*lambda*LMMD + Con_t + Con_s',
             'Full_no_LMMD':'CE + Con_t + Con_s',
             'Full_no_source_contrastive':'CE + .3*lambda*LMMD + Con_t',
             'Full_no_target_contrastive':'CE + .3*lambda*LMMD + Con_s'},
           excluded_terms='computed for logging but genuinely omitted from backward; no 0*loss substitutes',
           interpretation='Conditional leave-one-out effects, not additive shares of14.69pp gain',
           no_auto_followups=True)
    c.dump(c.OUT/'frozen_config.json',config)
    reused=c.audit_reuse();c.dump(c.OUT/'reused_controls.json',reused)
    provenance=json.loads((c.BASE/'provenance.json').read_text())['sha256']
    for p in c.OUT.iterdir():
        if p.is_file():provenance[str(p)]=c.sha(p)
    for arm in ('CE','Full'):
        for seed in c.SEEDS:
            folder=c.run_folder(arm,seed)
            for name in ('config.json','history.jsonl','frozen_manifest.json'):
                provenance[str(folder/name)]=c.sha(folder/name)
    c.dump(c.OUT/'provenance.json',{'sha256':provenance,'immutable_base':str(c.BASE)})
    link=Path('/home/zhangzj26/dual_axis_hsi_uda/results')/c.OUT.name
    assert not link.exists();link.symlink_to(c.OUT,target_is_directory=True)
    print('Locked9 newruns and6 reused controls: '+str(c.OUT),flush=True)

if __name__=='__main__':main()

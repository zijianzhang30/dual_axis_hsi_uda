"""Immutable four-cell study; reuse only inputs/plans, never mismatched scores."""
import json
import shutil
from pathlib import Path
import common as c


def main():
    assert not c.OUT.exists(),'Refuse overwrite'
    c.OUT.mkdir();snap=c.OUT/'snapshot';snap.mkdir()
    shutil.copytree(c.P2/'snapshot',snap,dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copytree(c.NATIVE,snap/'MLUDA',dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__','datasets'))
    (snap/'protocol').mkdir()
    text=(c.P2/'common.py').read_text()
    before="OUT = Path('/nas1/zhangzj26/dual_axis_hsi_uda/results/pavia_protocol_transfer_v1')"
    assert text.count(before)==1
    (snap/'protocol/common.py').write_text(text.replace(before,"OUT = Path('/nas1/zhangzj26/dual_axis_hsi_uda/results/pavia_backbone_adaptation_v1')"))
    # Only the study root changes; frozen C construction and input semantics stay identical.
    (c.OUT/'inputs').symlink_to(c.P2/'inputs',target_is_directory=True)
    (c.OUT/'plans').symlink_to(c.P2/'plans',target_is_directory=True)
    for path in Path(__file__).parent.iterdir():
        if path.suffix in ('.py','.txt'):shutil.copy2(path,c.OUT/path.name)
    old_provenance=json.loads((c.P2/'provenance.json').read_text())
    for rel,h in old_provenance['file_sha256'].items():assert c.sha(c.P2/rel)==h,rel
    fixed={'study':'Pavia network-package x official adaptation objective decomposition',
           'arms':c.ARMS,'seeds':c.SEEDS,'epochs':100,'primary':'fixed_epoch100',
           'data':'reuse frozen P2 z-score+ILDA cubes,180/class selections,11x11,bs32,38steps/epoch',
           'optimizer':'SGD .001/.9/5e-4 recreated each epoch; no scheduler',
           'main_augmentation':'P2 single-view spatial geometry, fixed common plans',
           'aux_augmentation':'actual official radiation(.9..1.1,beta.04) and native flip(B/C axes), unchanged',
           'all_cells_forward':'three source-target paired views every step; CE controls also execute auxiliary views',
           'M_network':'unmodified DSANSS/DCRN_02, native sequential CNN BN and original cross-domain attention',
           'O_network':'frozen C with Full Joint BN(.19), relation tokenizer,3 self Transformer blocks, same classifier',
           'O_training_heads':'two Linear64->128/F.normalize, isolated RNGseed+93000; same in CE/AD; inference ignores them',
           'AD_objective':'CE + .3*(2/(1+exp(-10*epoch/100))-1)*official LMMD + source SupCon + target pseudo-label SupCon',
           'contrastive_temperature':.1,'projection_dim':128,'head_nonlinearity':False,
           'alignment_feature':'classifier-input feature: M288, O normalized CLS64',
           'CE_objective':'source CE only; no LMMD/contrastive/pseudo-label optimization',
           'within_row_pairing':'same base/head initializations,all input views,model RNG,BN forward counts',
           'target_values':'not used for training or checkpoint selection; inherited P2 candidate-mask provenance',
           'evaluation':'same39348 P2 pixels/order,bs32,all tail pixels retained via repeat-last padding and trimming',
           'M_evaluation_source':'saved last source batch, same within M_CE/M_AD, as required by native cross-domain model',
           'O_evaluation_source':'target,target pairing, same within O_CE/O_AD',
           'native_flip_caveat':'np.fliplr/flipud on B,C,H,W reverses bands/batch; native source/pseudo labels not permuted; preserved and disclosed',
           'historical_scores':'P2 68.68 and native MLUDA91.23 mean are separate references, not paired2x2 cells',
           'historical_native_MLUDA_seeds':[1622,1322,1256,1712,1535,1670,1648,1191,1384,1436],
           'no_target_best':True,'no_target_tuning':True,'no_auto_followups':True,
           'interpretation':'whole network/normalization/readout packages vs complete official objective package; not pure CNN or individual loss attribution'}
    c.dump(c.OUT/'frozen_config.json',fixed)
    files=[p for p in c.OUT.rglob('*') if p.is_file()]
    # Frozen study plus every actually-used external protocol/input dependency.
    hashes={str(p):c.sha(p) for p in files}
    hashes.update({str(c.P2/rel):h for rel,h in old_provenance['file_sha256'].items()})
    c.dump(c.OUT/'provenance.json',{'sha256':hashes})
    print('Four-cell protocol frozen:',c.OUT,flush=True)


if __name__=='__main__':main()

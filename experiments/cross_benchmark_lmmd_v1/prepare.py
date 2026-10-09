"""Freeze protocol/code/reusable controls; never change old studies."""
import shutil
from pathlib import Path
import json
import common as c

assert not (c.OUT/'frozen_plan.json').exists(),'Never overwrite an existing study'
c.dump(c.OUT/'frozen_plan.json',c.PLAN)
paths=list(Path(__file__).parent.glob('*.py'))
paths+=list((c.RESULTS/'tokenizer_mechanism_v2/snapshot').rglob('*.py'))
paths+=[c.RESULTS/'tokenizer_mechanism_v2/common.py',c.RESULTS/'tokenizer_mechanism_v2/scorer.py',
        c.ROOT/'experiments/official_bench_v1/common.py',c.ROOT/'experiments/relation_tokenizer_v1/relation.py',
        c.ROOT/'experiments/bn_stabilization_v1/normalization.py',c.ROOT/'experiments/bida_self_dropout/train.py']
native=c.RESULTS/'pavia_backbone_adaptation_v1/snapshot/MLUDA'
paths+=[native/'mmd.py',native/'Weight.py']
bida=Path('/home/zhangzj26/IEEE_TCSVT_BiDA')
paths+=list((bida/'models').rglob('*.py'))+list((bida/'utils').rglob('*.py'))+list((bida/'loss').rglob('*.py'))
paths+=[bida/'train_pipeline.py',bida/'experiments/bida_memory_v1/run.py',bida/'experiments/bida_memory_v1/memory_attention.py']
data=Path('/nas1/zhangzj26/dual_axis_hsi_uda/datasets/HyRANK_author_v1/HyRANK')
paths+=list(data.glob('*.mat'))
data=Path('/home/zhangzj26/TGRS_MLUDA-2024/datasets/Houston')
paths+=list(data.glob('*.mat'))
reuse={}
for dataset in c.PLAN['datasets']:
 for arm in ('A','C'):
  for seed in c.SEEDS:
   folder=c.baseline(dataset,arm,seed);manifest=json.loads((folder/'frozen_manifest.json').read_text())
   assert manifest['checkpoints'][-1]['epoch']==200
   for record in [manifest['checkpoints'][-1],manifest['prediction']]:assert c.sha(record['path'])==record['sha256']
   paths +=[folder/'config.json',folder/'history.jsonl',folder/'frozen_manifest.json',Path(manifest['checkpoints'][-1]['path']),Path(manifest['prediction']['path'])]
   reuse['%s_%s_%d'%(dataset,arm,seed)]={'folder':str(folder),'prediction':manifest['prediction'],
            'parameters':manifest['config']['parameters'],'train_seconds':manifest['train_seconds'],
            'source_val_at_final':manifest['source_val_at_final']}
c.dump(c.OUT/'reused_CE_controls.json',reuse)
assert all(path.exists() for path in paths),[str(p) for p in paths if not p.exists()]
c.dump(c.OUT/'provenance.json',{str(p):c.sha(p) for p in sorted(set(paths))})
snapshot=c.OUT/'snapshot';snapshot.mkdir()
for p in Path(__file__).parent.glob('*.py'):shutil.copy2(p,snapshot/p.name)
link=c.ROOT/'results/cross_benchmark_lmmd_v1'
if not link.exists():link.symlink_to(c.OUT,target_is_directory=True)
(c.OUT/'PROTOCOL.txt').write_text(json.dumps(c.PLAN,indent=2)+'\n')
print('Frozen plan and all 12 CE control endpoints verified; no training started by preparation.')

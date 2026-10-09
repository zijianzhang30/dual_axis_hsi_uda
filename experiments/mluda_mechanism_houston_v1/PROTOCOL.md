# Full MLUDA mechanisms: Houston13 → Houston18

Independent four-arm study, copied from completed Pavia mechanism study but
using the unchanged native MLUDA_hu.py/config_Houston.py training recipe.
Existing Pavia/BiDA/HyRANK artifacts and tasks are untouched.
M0 full native MLUDA; M1 + Joint CNN BN; M2 + center-relative relation pool;
M3 + both. Author seeds first three in order: 1174,1370,1417. Fixed100 epochs.

Native Houston loader normalization + ILDA(pca_n=2,radius=.009),48 bands,
7×7 patches,180/class native source sampling,batch32/drop_last,38 updates/epoch.
Learning rate .01/(1+10*(epoch-1)/100)^.75; SGD(momentum=.9,wd=.0005)
recreated each epoch, unchanged feature/head groups.
Exact loss: CE_source + .01*lambda*LMMD + Con_source + Con_target + Domain_Occ.
All native augmentations and their RNG preserved. Native np.fliplr/flipud act
on B×C×H×W channel/batch axes; do not silently fix this historical behavior.
No source-CE-only, SceneShift, additional loss or target-based tuning/selection.

Only actual CNN BN1–7 join source+target for each original/noise/flip view
separately. BN9/10 unused, unchanged. Split before native CA/SA and attention.
Native momentum .1 twice/pair versus joint .19 once: old-statistic decay match,
NOT equivalent batch moments or normalization. Eval uses running statistics.
Preserve native test pairing with the final source training batch.

Replace only post-CA/SA avgpool: (actual_patch-center)/(spatial_population_std+1e-6),
center=(3,3), aligned7×7 grid, Linear48→32/ReLU→4 shared MLP,4 spatial softmaxes,
post-CA/SA CNN values,4 weighted288-D vectors averaged to288-D. All later
native cross-attention/heads unchanged. New MLP in feature_layers,independent
CPU seed+41000 initialization after entire original model; original tensors/RNG
unchanged. M2/M3 new tensors match per seed.

Historical strict Houston preprocessing/checkpoint/stream bundle not identified;
rebuild all three paired M0, never use historical multiseed means as controls.
ILDA precedes native set_seed and uses unspecified PCA RNG. This new study fixes
an independent pre-ILDA seed20261006 solely for reproducible shared preprocessing,
freezes one cube pair, and discloses this change; no accuracy-driven choice.
Native ILDA algorithm and source sample selection remain unchanged.

Before formal training: off-switch outputs/full loss/gradient/update equivalence,
uniform-pool avgpool equivalence, BN calls, target gradients through joint moments,
center/grid checks, new parameter group membership/updates and RNG invariance.
Native CUDA pooling backward lacks deterministic implementation; verification
uses explicit tolerance and verification-only backend controls. Formal training
retains native backend behavior. All12 final predictions/checkpoints frozen
before scoring. Check same-seed native initial states, exact samples, every
batch/augmentation/RNG stream, evaluated pixels and final-source test pair.
Native OA divides correct retained predictions by FULL target dataset length;
AA/Kappa use retained predictions. No drop_last/padding changes per arm.
Report OA/AA/Kappa/recall/shares/confusion,collapse(max share≥.95),zero-recall
classes,parameters,time,and paired M1−M0,M2−M0,M3−M1,M3−M2,all seeds retained.
No automatic additional experiments/retries. GPUs0/1/6/7,one task per GPU.
NAS new directory: /nas1/zhangzj26/dual_axis_hsi_uda/results/mluda_mechanism_houston_v1.

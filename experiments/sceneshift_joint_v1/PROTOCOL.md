# Noisy SceneShift x Full Joint BN: minimal 2x2

Frozen before new training/outcomes. Twelve fresh runs, arms A/B/C/D each with
seeds 2100/2101/2102. Houston13 -> Houston18, normband, 13x13 patches, batch
128/domain, source split state 23, original BiDA spatial-softmax tokenizer,
depth 3, dimension 64, four tokens, dropout .1, 376,567 trainable parameters,
SGD .01 without momentum/schedule, 18 updates/epoch, 200 epochs. Identical
initialization including unused EMA initialization, dataset/augmentation,
loader ordering and source-validation cadence (1,10,...,200). No new module.

- A: native Original sequential source -> target BN, source CE.
- B: A plus historical noisy shifted-source auxiliary CE.
- C: differentiable Full Joint BN, source CE.
- D: C plus the identical auxiliary shift/CE branch as B.

SceneShift is frozen historical alpha .7, epsilon 1e-5, per-patch multiplicative
Gaussian SD .04, additive Gaussian SD .015 smoothed by avg_pool2d(5,1,2),
clamp [0,1], weight .5. Thus B/D loss is CE(original source)+.5 CE(shifted source),
not the alpha .8 half/half pure-affine variant. Statistics are full-scene,
per-band mean/population std in current normband space, no GT mask, no padding.
Using normband instead of old ILDA is the necessary backbone input adaptation;
do not transfer historical MLUDA numerical gains to this protocol.

Auxiliary forward uses model(shifted_source, same_target)[0] in train mode,
with normal running-buffer updates (no restore). B retains sequential BN for
both forwards; D uses joint statistics for each source/target pair. Original
BN momentum .1 per domain update; Full Joint .19 per paired update, retaining
the current reference definition. The intervention is the full frozen BN
recipe, not a momentum-only or moment-only contrast. No target CE or MLUDA
LMMD/SCL/occupancy/attention/pseudo-label loss is added.

Shift noise uses separate CUDA generator seed+99173. Auxiliary dropout uses
an isolated persistent CUDA RNG stream seed+199173, inside fork_rng, so neither
noise nor auxiliary forward advances original-forward RNG. This RNG control
does not change the augmentation distribution or loss weights. Compare all
four arms' per-epoch input and pre-original-forward RNG stream hashes directly.
Save 10-epoch checkpoints, but infer only fixed epoch 200; no target selection
or oracle search. Freeze checkpoint/prediction manifest before post-hoc GT.
GT-mask target sampling follows unchanged BiDA loaders; target label values
never enter loss/model selection. No claim of wholly GT-free sampler is made.

Report OA/AA/Kappa, per-class recall, class-6 prediction ratio and >=95%
class-6-collapse count, mean +/- sample SD (ddof=1), paired B-A/C-A/D-C and
interaction (D-C)-(B-A), fixed-200 source validation, parameters and cost.
Cost includes preparation, training wall time and measured auxiliary CUDA
forward+loss milliseconds; concurrent elapsed times are not controlled speed
benchmarks. No historical A/C artifact is overwritten; compare their replay
for verification, but fresh runs are the formal 2x2 endpoints.

Operationalize user Go/No-Go before outcomes: Go only if at least one of OA/AA
has mean D-C >=1 pp and positive D-C in at least 2/3 seeds in that same metric,
with no increase in collapse count. Disclose all negative paired changes and
the other metric; this practical exploratory gate is not a significance test.
If unmet, no further SceneShift stacking, alpha sweep or pure-affine ablation.

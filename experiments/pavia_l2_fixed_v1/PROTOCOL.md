# Repaired Pavia spectral-L2 frozen validation

New experiment, no historical code or outputs overwritten. Fix only the confirmed
preprocessing return-value defect: reshape the normalized flat array back to the
original scene cube. Both real cubes must exactly match archived native normband.

A: BiDA-self + Original Mixed BN + original tokenizer.
B: BiDA-self + Full Joint BN + original tokenizer.
C: BiDA-self + Full Joint BN + current center-relative relation tokenizer.

PaviaU -> Pavia City, author-format 102 bands/7 classes; spectral L2 from raw cube;
no z-score/ILDA. Same native 95% stratified source pool/validation random_state=23,
13x13 reflect padding, existing strict native boundary mask, batch128,
original proportion shuffle without replacement, 290 updates/epoch, 200 epochs,
seeds2100/2101/2102, SGD0.01 without momentum/decay/scheduler, source CE only.
Depth3, dim64, four tokens, dropout0.1. Full Joint BN momentum0.19, target
statistics remain differentiable. Relation MLP initialized in isolated native
fork_rng(seed+41000), eps1e-6; all other initialization/RNG retained.

Native source-val calls at epoch1 and every10 remain. Checkpoints every10; formal
target score fixed200, predictions frozen before GT evaluation. No target-best
selection, hyperparameter search, pseudo labels or additional loss. Target labeled
mask used as candidate centers; train label values replaced by zero.

Preflight: same real source/target batch (128 each), archived native model vs local
source-CE path three steps; explicit Joint BN reference three steps; relation grid,
formula, values and gradients; all seeds initial states/data/indices verified.
This does NOT certify full official BiDA/EMA/loss reproduction.

Nine new runs, physical GPUs1 and6, one run per GPU. No automatic retries or extra
tasks. Report OA/AA/Kappa mean+sample std, recalls, prediction ratios, >=95% collapse,
zero-recall classes, same-seed B-A/C-B/C-A, source-val, parameters and timing.
Preserve full configs/logs/predictions/checkpoints/code+data hashes on NAS.
Historical raw-input results may only appear as separately labeled references.

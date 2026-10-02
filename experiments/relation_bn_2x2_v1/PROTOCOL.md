# Tokenizer × BN 2×2 completion: only three new training tasks

Freeze before new training/target outcomes. Reuse eligibility passed in
`results/relation_bn_2x2_v1/reuse_audit.json`: all three historical cells have
matching seeds, common initialization, data/split, 200 epochs of batch order and
pre-forward CPU/CUDA RNG; historical checkpoint/prediction hashes verified.

1. Original sequential mixed BN + original tokenizer: reuse sceneshift_joint_v1/A.
2. Original sequential mixed BN + center-relative tokenizer: three new runs only.
3. Full Joint BN + original tokenizer: reuse relation_tokenizer_v1/A.
4. Full Joint BN + center-relative tokenizer: reuse relation_tokenizer_v1/C.

Houston13→Houston18, seeds2100/2101/2102, 200 epochs, fixed200 primary. Preserve
normband, historical95% source split random_state23, reflect13×13 patches, native
augmentation/target candidate-mask sampler, batch128/domain, SGD.01 no momentum
or scheduler, depth3/dim64/K4/dropout.1, source-self CE only, no other modules/losses.
Instantiate historical BiDA and unused EMA, then call the UNMODIFIED
relation_tokenizer_v1/relation.py installer (arm C, isolated seed+41000 MLP RNG).
Do NOT install joint normalization for cell2. The saved original native BiDAnet
forward and native _tokenize execute source CNN/BN then target CNN/BN, momentum.1
with two running-buffer updates per batch. No approximate replacement BN.
Check exact initial full-state/MLP hashes against cell4 and every batch/RNG hash
against the historical three cells. No historical files/results altered.

Cell2 uses exactly r=(x−center)/(population_patch_std+1e-6), MLP48→32(ReLU)→4,
spatial softmax, original CNN values on aligned13×13 grid. No architecture search.
Pre-training gate verifies native forward identity, native BatchNorm types,
source→target hook order, two counter updates, momentum.1, numerical/gradient
sanity and unchanged initialization. Original BN does not introduce the joint
statistics gradient pathway; target source-loss gradients may therefore be zero.

## User-added ten-epoch diagnostic

Save/evaluate grid10,20,...,200. After all three new training runs finish, perform
independent frozen-checkpoint inference for all FOUR cells on the same20-point
grid, saving predictions under this NEW experiment directory, including historical
cells. Freeze all240 prediction artifacts and hashes before any target scoring.
Do not insert target-evaluation loader iteration into training's shared RNG stream.
Fixed200 remains primary. Maximum target OA on the grid is explicitly labeled
target-oracle diagnostic, with its corresponding epoch and associated AA/Kappa.
It is NOT a target-blind selection rule; do not use it for tuning, early stopping,
formal checkpoint choice or automatic version selection. Local released BiDA
train_pipeline evaluates every10 epochs AND epoch1; this diagnostic intentionally
uses the agreed20-point grid only. No additional training tasks for the grid.

## Report

All four cells' fixed200 OA/AA/Kappa, recall per class, prediction distribution,
class6≥95% and any-class≥95% collapse, zero-recall classes, source-val, parameters
and timing; same-seed 2−1 and 4−2 deltas with mean±sampleSD(ddof1). Also report
oracle-grid OA/epoch and full trajectories separately. No automatic promotion
or appended experiments; do not change/stop Pavia or other experiments.

# A1 BatchNorm transplant: diagnostic seed 2100

Frozen before training or reading new target outcomes.

## Hypothesis

Target updates to BatchNorm running statistics improve A1 target transfer while
the source CE and every trainable weight follow exactly the same trajectory.

## Arms

- N0: A1 with its three stem GroupNorm layers replaced by BatchNorm3d;
  source batches update BN statistics; source CE trains the weights.
- N1: the same A1-BN weights and source batches; after each source optimizer
  step, a target batch also updates BN statistics under `no_grad`.

N0 and N1 share one trainable parameter trajectory. Separate BN buffers are
swapped around each forward. Both buffer paths receive the same source batch;
N1 alone receives the paired target batch. Source logits must agree within
`1e-4` before every optimizer step. This makes the only arm difference the
target contribution to BN running statistics.

Houston13 → Houston18 uses the existing Strict BiDA `normband` loader,
95/5 source split, 13×13 patches, batch 128, SGD 0.01 without momentum or
schedule, and seed 2100. Both arms use source CE only, fixed epoch 200, and
target self inference. The existing A1-GN fixed epoch result is an external
reference and is not retrained. Target candidate centers still follow the
released Houston18 GT mask; target label values are ignored by training.

Predictions and checkpoint hashes are frozen before post-hoc target labels are
collected. Report OA, AA, Kappa, per-class recall, confusion matrix, target
entropy, prediction distribution, source validation OA, exact parameter hash,
and training/inference cost.

## Predeclared diagnostic gate

N1 passes if `N1 − N0` target OA is at least +2.0 percentage points, AA does
not decrease, at least five classes have nonzero recall, and the largest
predicted-class share does not rise by more than 10 percentage points. A
positive gate from one seed motivates, but does not establish, a three-seed
result. N1 must also be compared with the existing A1-GN seed-2100 result
(73.58% OA in the paired A2-lite run) before claiming a backbone improvement.

# BiDA-self BN decomposition: seed 2100

Frozen before training or inspecting new target outcomes.

## Question

How much of the BiDA-self epoch-200 result is caused by allowing target
batches to persistently update shared BatchNorm running statistics?

## Arms

- Existing control: BiDA-self mixed BN, seed 2100, epoch 200, 78.41% OA.
- New intervention: identical BiDA-self training, but after every paired
  source/target forward and its backward pass restore each BN buffer to its
  state immediately after that step's source tokenization. Target is still
  forwarded and consumes the same dropout RNG; its BN updates do not persist
  to the next step.

The source forward uses training-mode batch statistics, so this buffer-only
intervention must leave source logits, source CE, gradients, and trainable
weights unchanged relative to an exact replay of the mixed-BN control.

## Fixed protocol

Houston13 → Houston18, `normband`, 13×13 patches, batch 128, source split
`random_state=23`, seed 2100, SGD 0.01 without momentum or schedule, three
BiDA blocks, dropout 0.1, source CE only, 200 epochs, fixed epoch-200 target
self inference. Target labels enter neither training nor checkpoint selection;
the released GT mask still defines eligible target centers.

The initial model hash must equal the existing BiDA-self/Strict-BiDA hash
`dadb50a683a2bab506df66c317cbea8d08aee955bab5281013dd4604d6fa40b5`.
The new run checks its source CE against every epoch of the completed mixed-BN
history and stops if they differ beyond numerical tolerance. Predictions and
hashes are frozen before post-hoc target evaluation.

## Interpretation

Report mixed minus source-only BN for OA, AA, Kappa, per-class recall,
prediction distribution, entropy, and source validation. A difference of at
least 2 OA points with AA moving in the same direction is a material BN effect
for this seed. Otherwise the remaining BiDA-self advantage should be tested
through transformer depth next. No BN momentum or ordering search is allowed.

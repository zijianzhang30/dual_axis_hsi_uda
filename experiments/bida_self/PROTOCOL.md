# H2 BiDA-self decomposition: diagnostic seed 2100

Frozen before training or inspecting new target outcomes.

## Hypothesis

Strict BiDA's fixed-epoch advantage over A1 is mainly supplied by its self
representation, target-conditioned BatchNorm, and dropout rather than its
coupled UDA objectives.

## Intervention

Run the released BiDA architecture with its original 3D/2D stem, BatchNorm,
four semantic tokens, CLS token, three transformer blocks, and dropout. Each
training step forwards the paired source and unlabeled target batches through
the same model in the original source-then-target order, preserving target
updates to shared BN statistics. The only optimized loss is source self CE.

Disabled components:

- target coupled-to-self distillation;
- source coupled-to-self distillation;
- EMA teacher and consistency losses;
- epoch-101+ MMD.

The coupled tensors are still constructed by the unmodified model forward but
are absent from the loss. They therefore send no gradient to model weights.

## Fixed protocol

- Houston13 → Houston18, released `normband` normalization and GT-mask target
  candidate centers;
- source split fixed by `sample_gt(..., random_state=23)`;
- seed 2100, 13×13 patches, batch 128, SGD 0.01 without momentum or schedule;
- 200 epochs; fixed epoch 200 is the primary checkpoint;
- target self branch at inference;
- target labels excluded from training and checkpoint selection.

Instantiate the unused EMA model before training to preserve the released
Strict BiDA model-initialization RNG sequence, but never forward or update it.
The trained model's initial state hash must match the existing Strict BiDA
seed-2100 hash `dadb50a683a2bab506df66c317cbea8d08aee955bab5281013dd4604d6fa40b5`.

Freeze the final checkpoint, target logits, predictions, configuration, and
SHA256 hashes before collecting target labels for post-hoc metrics.

## Predeclared interpretation

Use existing fixed epoch-200 references:

- A1-GN seed 2100: 73.58% OA, 62.13% AA;
- Strict BiDA seed 2100: 78.75% OA, 66.95% AA.

If BiDA-self is within 2.0 OA points of Strict BiDA with no collapse, treat the
self representation/normalization/regularization as the dominant explanation
for this seed. If Strict BiDA exceeds BiDA-self by at least 2.0 points with no
collapse, treat the original UDA objective as having a material positive
increment worth decomposing. If neither condition holds, record the effect as
ambiguous and do not tune on Houston18.

This one-seed diagnostic does not establish a multi-seed method claim.

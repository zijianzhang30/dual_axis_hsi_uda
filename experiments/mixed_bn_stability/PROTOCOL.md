# Mixed-BN seed-instability diagnostic

Frozen before running new recalibrated predictions.

## Questions

1. How different are the final BatchNorm running means and variances across
   seeds 2100/2101/2102?
2. When does the Hybrid seed-2101 target prediction distribution concentrate
   on class 6?
3. Can changing only BN running buffers rescue a collapsed model while all
   learned weights, including the classifier, remain fixed?

## Frozen models

Analyze the three completed BiDA-self controls and the three completed Hybrid
models. Use fixed epoch 200. For the Hybrid trajectory, read the already frozen
predictions at epochs 10, 20, ..., 200; do not select a new formal checkpoint.

For each final model evaluate three buffer states:

- `original_mixed`: checkpoint BN buffers;
- `source_recalibrated`: reset BN buffers and estimate cumulative statistics
  from one deterministic pass over the fixed source-training set;
- `target_recalibrated`: reset BN buffers and estimate cumulative statistics
  from one deterministic pass over the unlabeled target set.

Only `running_mean`, `running_var`, `num_batches_tracked`, and BN momentum may
change. All trainable tensors must remain byte-identical and are checked before
and after recalibration. Recalibration calls only the stem/tokenizer path once
per image. Freeze prediction logits, class predictions, classifier-input
features, BN snapshots, and SHA256 hashes before reading target labels.

## Interpretation

If target recalibration materially improves seed 2101 with fixed learned
weights, attribute a substantial part of collapse to a BN-buffer/classifier
state mismatch. If neither source nor target recalibration restores it, infer
that the failure is encoded in learned weights or representations rather than
only the final running buffers. Cross-seed BN distances are descriptive because
each seed also has different learned stem weights; do not treat distance alone
as causal evidence.

Target labels are post-hoc only. No recalibration choice or checkpoint is tuned
using target metrics.

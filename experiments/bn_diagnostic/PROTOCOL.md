# H1 BatchNorm Diagnostic Protocol

Status: frozen before target prediction or evaluation.

## Hypothesis

Part of Strict BiDA's target transfer comes from target-conditioned BatchNorm
running statistics rather than its coupled source-target branch.

## Fixed diagnostic

- Model: unmodified Strict BiDA.
- Checkpoint: fixed epoch 200, optimization seed 2100.
- Checkpoint path: `/home/zhangzj26/IEEE_TCSVT_BiDA/experiments/bida_memory_v1/formal/strict_2100/final.pth`.
- Data and normalization: released Strict BiDA Houston13 → Houston18 loaders,
  `normband`, patch size 13.
- Inference: target self branch only.
- No optimization, gradient update, checkpoint selection, A1 change, or H2/H3 run.

Three copies of the same checkpoint are evaluated:

1. `original_mixed`: preserve checkpoint BatchNorm buffers.
2. `source_bn`: reset every BatchNorm buffer and estimate it once over the
   unaugmented Houston13 source-training patches.
3. `target_bn`: reset every BatchNorm buffer and estimate it once over the
   unaugmented Houston18 target candidate patches. Labels are ignored.

Recalibration uses cumulative statistics (`momentum=None`). Only BatchNorm
layers enter training mode. All other modules remain in evaluation mode, and
only `model._tokenize(images)` is called so each patch updates BatchNorm once.

Predictions, logits, configuration, BatchNorm buffers, and SHA256 hashes are
written before target metrics are computed. Target labels are then read in a
separate evaluation pass.

## Predeclared gate

H1 passes only if `target_bn` relative to `source_bn`:

- improves target OA by at least 2.0 percentage points;
- does not reduce target AA;
- retains nonzero recall for at least 5 of 7 classes; and
- does not increase the largest predicted-class share by more than 10
  percentage points.

If the gate fails, proceed to the separately preregistered H2 BiDA-self
diagnostic. No recalibration choices may be changed after inspecting target
metrics.

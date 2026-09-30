# V0 revised protocol: source memory teaches target self

This records the revised V0 specification supplied after the original [design](V0_DESIGN.md). Its architecture is retained: factorized 3D stem, explicit spectral axis, spatial and spectral semantic queries, bidirectional fusion, class query, and separate class × slot EMA memories. A0–A2 remain representation controls.

## A3 training paths

For each paired source and target batch, the shared network emits `logits_self` and `logits_memory` for both domains. The self logits classify fused dual-axis tokens directly. The memory logits classify the same tokens after slot-matched attention over source-class memory and residual injection. All weights are shared.

At step *n*, both domains read memory from step *n−1*. The current source batch updates memory only **after** backpropagation and the optimizer step. Uninitialized classes are masked; before any class is initialized, memory logits equal self logits. EMA updates use detached source tokens and source GT only.

For A3, with `T=1` and fixed `lambda_dist=1`:

```text
L_src_self = CE(source.logits_self, source_label)
L_src_mem  = CE(source.logits_memory, source_label)
L_tgt_dist = T² KL(softmax(stopgrad(target.logits_memory / T))
                  || softmax(target.logits_self / T))
L_source   = 0.5 * (L_src_self + L_src_mem)
L_total    = L_source + lambda_dist L_tgt_dist
```

The target memory teacher is detached in the KL term. A0–A2 use source self CE only; they consume the paired target loader to preserve the BiDA-style step organization. No target pseudo-label CE, contrastive term, MMD/LMMD, Flow Matching, OT, or external prior is used.

## Inference and reporting

The primary epoch-200 prediction is **target self**. The target memory prediction is evaluated only for changed-prediction and class-attention diagnostics. `forward` exposes both logits and class-query embeddings, spatial/spectral tokens, position attention, memory class attention, and learned injection strengths. Report OA, AA, Kappa, per-class recall, and seed mean/std.

## Protocol caveats

1. Strict BiDA's target training loader uses the Houston18 GT **mask** to define eligible unlabeled patch centers. The code discards target label values in the loss, but the mask still affects sample selection. This is retained for direct Houston development comparison. A no-target-GT sampler requires a separate protocol.
2. The source CE average matches A2's source gradient scale when the two branches coincide. Log both component CEs and their mean to audit this comparison.
3. Self and memory predictions can be nearly identical with near-zero injection coefficients, so target KL may initially be tiny. Log its per-epoch value, both learned injection coefficients, and target self/memory prediction disagreement before interpreting target OA.

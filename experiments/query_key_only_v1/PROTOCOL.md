# Key-only PE: paired seeds 2100 / 2101 / 2102

Frozen before training or new target outcomes. Compare the existing depth-0
Query + Class Query PE-on/off controls with one new variant only:
Q=q_norm(learned_query), K=kv_norm(X+PE), V=kv_norm(X). The same kv_norm
parameters are shared between K and V as in the original reader. Keep native
MultiheadAttention projections, attention dropout 0, query residual, FFN/LN,
Class Query, dropout 0.1, classifier and parameter count unchanged. No new LN,
learned PE scale, residual scale, attention module, loss or optimizer change.

Initialize full original BiDA, consume unused EMA initialization, transplant
spatial Query and install Class Query exactly as paired controls. Verify all
three initialization hashes and shared component code hashes. Rebind only the
parameter-free spatial forward; this consumes no random draws and preserves
tensor/dropout shapes and call order. Historical batches are not directly
hash-audited; loader and RNG paths are identical by construction.

Full Joint BN with native joint moments, momentum 0.19 and differentiable target
statistics, depth 0, source CE only, SGD 0.01, 200 epochs, seeds 2100/2101/2102.
Houston13 -> Houston18 normband, 13x13 patches, batch 128/domain, split state 23.
Save every 10 epochs; generate/hash all 20 target predictions and manifest
before GT metrics. Fixed epoch 200 primary; target-oracle max OA diagnostic only.

Report OA/AA/Kappa per seed and mean +/- sample SD (ddof=1), recalls, prediction
shares/counts, paired differences versus on/off, and >=95% class-6 prediction
share collapse count plus zero-recall classes and checkpoint anomalies.
Main reference Full Joint + BiDA tokenizer (depth 3) is 79.50 +/- 1.46 OA,
71.90 +/- 1.72 AA. User's practical performance gate is mean OA >= reference,
mean AA >= reference, and 0/3 class-6 collapse; use unrounded means and disclose
close differences. This is a small-seed practical gate, not a significance test.

If unsuccessful, keep the BiDA tokenizer reference; do not sweep LN, PE scales
or extra modules to rescue the tokenizer. Even a success would not alone prove
that PE-in-Value universally causes directional collapse; target weights and
feature distributions co-adapt during training. No further variant is authorized
by this protocol.

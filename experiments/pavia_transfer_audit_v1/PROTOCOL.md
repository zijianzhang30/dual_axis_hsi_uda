# Pavia transfer implementation audit and frozen three-arm validation

No changes to existing models, training scripts, data, losses or results.
No new full training unless historical reuse fails and the mismatch is reported.

A: BiDA-self + sequential Original Mixed BN + original tokenizer.
B: BiDA-self + Full Joint BN (momentum 0.19) + original tokenizer.
C: BiDA-self + Full Joint BN + center-relative relation tokenizer.
All use historical normband, 95/5 stratified split (native random_state=23),
13x13 reflected patches, batch128, SGD0.01, source CE only, seeds2100/2101/2102,
fixed epoch200, same target mask. This is a BiDA-style extension, not the
MLUDA protocol or a full official-BiDA-loss reproduction.

Checks: native archived BiDA generic normband vs local whole-image normalization;
native HSIDataset construction with only necessary self.data=image runtime fix
vs local patches; archived official BiDAnet vs local model on same real batch,
three source-CE SGD steps with RNG/state reset, logits/loss/gradients/buffers/
parameters/eval logits; explicit joint-batch reference vs installed Joint BN;
relation formula, grid, aggregation and target/source gradient checks.

The numerical audit uses a separate process, not a running experiment. It does
not certify official full-loss/EMA training equivalence or paper-score recovery.
Historical training source, data/model hashes, initial tensors, checkpoint/prediction
hashes and all200 epoch batch/RNG traces must match before result reuse.
Only frozen final predictions are scored. Report sample std, per-class recall,
prediction shares, collapse >=95%, zero-recall classes, paired B-A/C-B/C-A.
All results retained; no target-selected protocol or extra experiments.

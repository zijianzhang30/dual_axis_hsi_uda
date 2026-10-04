# User-requested epoch-100 normalization diagnostic

Added after launch, when B/C were at epochs21/24; no new target100 scores had
been inspected. Keep 200 epochs as the formal endpoint. No model selection,
hyperparameter changes, early stopping or new training tasks.

In a separate process, wait for complete B/C seed2100 epoch100 checkpoints
and history records. Infer both standardized target predictions and hash them
before scoring. Reuse the already-frozen normband B/C seed2100 epoch100
predictions, verify their checkpoint/prediction hashes and identical evaluation
indices, then score all four on the same GT order. Report OA/AA/Kappa, recall,
prediction shares, single-class collapse, zero-recall classes and per-tokenizer
zscore-minus-normband deltas. This is a single-seed mid-training diagnostic,
not a stability claim or MLUDA-protocol reproduction. Save in a separate NAS
diagnostic subdirectory without editing the frozen training code or results.

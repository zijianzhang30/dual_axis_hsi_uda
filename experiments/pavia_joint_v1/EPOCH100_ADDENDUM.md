# User-requested epoch-100 diagnostic

Added after training had begun, at the user's explicit request, before any
epoch-100 target results were available. This is not a preregistered primary
endpoint. The original PROTOCOL.md and its frozen hash remain unchanged.

Evaluate all six saved epoch-100 checkpoints in a separate process after all
six checkpoints are complete. Preserve training code, model selection, optimizer,
BN buffers and RNG state in all running training processes. Fixed epoch 200
remains the sole primary endpoint; do not tune or stop based on this diagnostic.

Freeze and hash all six prediction files before accessing target class values
for metrics. Report per-seed OA/AA/Kappa, per-class recall, prediction shares,
any-class ≥95% collapse, paired Joint−Original deltas and sample mean ± std.
Reuse the same deterministic target evaluation dataset. No target-oracle search.

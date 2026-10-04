# User-requested Pavia epoch-100 diagnostic

Added while Pavia C was at epochs 15–16, before its epoch-100 target results
were available. This is a mid-training diagnostic, not a new primary endpoint.
Keep all prescribed 200-epoch training processes and the original frozen
protocol unchanged. Do not use these metrics for checkpoint selection, tuning
or early stopping. Shanghai tasks are unchanged.

In an independent process, wait for all three complete C epoch-100 checkpoints
and histories. Compare historical B epoch-100 checkpoints against C at the
same epoch, with the matching Pavia evaluation mask. Verify frozen dependencies,
initialization and data hashes. Infer all six checkpoints and freeze all six
prediction files with checkpoint hashes before scoring. Report OA/AA/Kappa,
per-class recall, prediction distribution, single-class collapse (>=95%),
zero-recall classes and paired C−B. Output only in a new NAS diagnostic directory.

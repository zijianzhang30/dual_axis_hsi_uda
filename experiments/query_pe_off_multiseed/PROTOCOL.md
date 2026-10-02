# PE-off three-seed replication extension

Frozen before training seeds 2100 and 2102 or inspecting their PE-off outcomes.
Reuse the completed, frozen seed 2101 intervention without rerunning or changing
any old artifact. Train only seeds 2100/2102 using the identical PE-off pipeline:
BiDA stem, four A1 spatial queries, no spatial PE addition, depth 0, A1 Class
Query, retained LN/linear, dropout 0.1, Full Joint BN with momentum 0.19 and
differentiable target moments; source CE only; SGD 0.01; 200 epochs.
Houston13 -> Houston18, normband, 13x13 patches, batch 128/domain, source split
random_state 23. Same initialization recipe (including unused original EMA),
loader/evaluation cadence and RNG path as each paired PE-on depth-0 control.
Verify original, transplant and final pipeline initial hashes and all shared
component script hashes against paired PE-on. No extra module or seed search.

Save checkpoints 10:10:200; generate/hash all target predictions and manifest
before post-hoc target metrics. Fixed epoch 200 is the formal endpoint; oracle
maximum OA on the 20-checkpoint grid is separately labeled diagnostic only.

Report per-seed OA, AA, Kappa, all per-class recalls and prediction shares/counts,
three-seed mean +/- sample SD (ddof=1), and paired off-minus-on differences.
The reference Full Joint + BiDA tokenizer depth 3 is 79.50 +/- 1.46% OA and
71.90 +/- 1.72% AA. Distinguish reference comparison from a strict tokenizer
causal contrast because depth/readout differ from the PE-off pipeline.

Use the existing >=95% class-6 prediction share threshold and also report
missing-class recalls and worst-seed metrics; 0/3 class-6 collapse alone is not
proof that all minority classes are preserved. Report grid-level anomalies.
Stability here means all three fixed endpoints avoid class-6 collapse, with
coverage and variance separately inspected; do not silently redefine success
after seeing outcomes. Three seeds are limited evidence, not a robust guarantee.

Do not start Key-only PE or LN/attention changes in this extension. If the
replication supports continued work, report that as a next step with the user's
proposed K=LN(X+PE), V=LN(X) hypothesis still unproven. It is not validated merely
by PE-off improvement, which also changes keys and learned weight co-adaptation.

# No-BiDA-Transformer query pipeline: A1 class-query readout

Frozen before training. Houston13 to Houston18, same fixed split, normband,
13x13 patches, batch128 per domain. Seeds2100/2101/2102, 200 epochs, source
self CE only, SGD .01 without momentum/schedule. No new UDA loss, tuning,
residual scaling, BN change, target checkpoint selection or augmentation change.

Architecture: existing BiDA stem + Full Joint native BN (momentum .19 per
paired step, equal counts, differentiable target moments) + exact existing
A1 spatial-query tokenizer package (64-dim, four queries, 2D sinusoidal PE,
CrossBlock with cross-attention/FFN) + token dropout .1 + A1 class-query
CrossBlock (64-dim, four heads) + LayerNorm + seven-class linear classifier.
The A1 CrossBlock itself has its original default zero attention dropout.
No BiDA transformer block or original CLS/positional sequence remains.
There is still attention/FFN inside the A1 readers: depth0 means no BiDA
refinement blocks, not no attention anywhere.

For paired initialization, build original depth3 BiDA, unused EMA and spatial
query transplant in the previous RNG sequence, verifying all per-seed initial
hashes. Remove all BiDA blocks and unused cls_token, pos_embedding, token_wA/V.
Append A1 class_query (normal .02) and class_reader. Retain stem, spatial
queries/reader, final LayerNorm and linear classifier initial tensors exactly.
The retained norm+linear has A1's classifier structure with prior BiDA initialization,
not a literal transplant of A1's original 128-dim classifier.

This changes depth, readout, trainable parameter count and dropout RNG sequence;
it is an end-to-end query-pipeline control, not a pure depth causal lesion.
Success would show this pipeline can work without BiDA refinement; it would
not prove the spatial query reader alone is universally stable. Failure would
not uniquely indict that reader because the new class readout also co-adapts.

Primary fixed epoch200. Save epochs10:10:200; after training freeze/hash all20
target predictions before GT metrics. Secondary target-oracle maximum OA on
this grid, earliest tie, diagnostic only. Report OA/AA/Kappa mean ± sample SD,
per-class recall, prediction shares, entropy, source-val and class6 collapse
(>=95% predictions) both at endpoint and all20 grid points. Compare against
Full Joint BiDA tokenizer and Query depths1/3. No 79% threshold is assumed.

After outcomes, use the same fixed2048 unlabeled target sample as prior probes
to report spatial token, class-reader pre-LN and classifier-input norms,
direction concentration and local cosine to classifier w6. No probe updates.

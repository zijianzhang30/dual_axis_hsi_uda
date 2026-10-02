# Internal spatial-query diagnosis: existing seed2101 checkpoints

Two saved pipelines: query_class and query_mean from the completed2x2,
epochs10:10:200. Fixed2048 source and target samples/indices from prior probes,
no labels for sampling/grouping. No training, loss, Class Query or BN change.
Eval with existing buffers; all state and checkpoint hashes checked. Probe hooks
must preserve logits exactly. These are retrospective observations, not live
minibatch histories; source-target sample pairs have no semantic correspondence.

Actual transplant has no separate A1 spatial_projection layer. Track:
BiDA3D stem output (flatten band/channel) -> Conv2d projection preBN -> BN2D
-> ReLU stem feature -> added2D positional encoding -> KV LayerNorm ->
attention delta -> query+attention residual -> FFN LayerNorm -> FFN delta ->
FFN residual/four tokens -> classifier z. Fixed query norms are a baseline,
not learned collapse. Hook only spatial_reader, not class_reader internals.

Per stage/domain report pooled norm/concentration, average per-position or
per-query-slot concentration, cross-sample centered RMS and raw RMS/fraction,
pooled covariance participation rank (tr(C)^2/tr(C^2)), local classifier cosine
where64-dim. Fixed position additions preserve absolute centered variation;
their raw cosine increase alone cannot establish information loss.

Report distributions (mean,SD,quantiles,histograms) of4096 fixed random
within-source, within-target and source-target pooled feature cosines, and
same-coordinate cosines at fixed representative image positions/query slots.
These are geometrical similarities, not same-class feature alignment.
Cross-stage dimensions/token counts differ; averaging169 pixels into4 queries
normally changes variation and must not by itself be called collapse.

Only if the evidence supports one identifiable component, propose/freeze a
single seed2101 component intervention before training it, leaving other
components/loss/BN/readout intact. No grid of PE/FFN variants. If attribution
remains ambiguous, report that and do not pick a rescue ablation arbitrarily.

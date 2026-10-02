# Hybrid depth V1

Freeze before new training: Hybrid-V2 Query package + Full Joint BN, depth 1
versus completed depth 3; seeds 2100/2101/2102, epochs 200. Same Houston
data/split, batch128 per domain, normband, 13x13, source CE only, SGD .01
without momentum/schedule, dropout .1, native joint BN momentum .19.

Initialize the full depth-3 Hybrid with the existing RNG sequence including
unused EMA; verify existing per-seed BiDA and Hybrid initial hashes. Then
retain only block1. All retained tensors must match full initialized state.
Removing blocks changes subsequent dropout RNG consumption; this is not
an exact minibatch/dropout trajectory replay. Depth and parameter count change.

Depth 0 is deferred: native CLS would be image-independent without attention,
and the original training forward does not define fusion outputs at depth 0.
Mean pooling would introduce a separate readout intervention requiring direction.

Save epochs10:10:200; freeze/hash all20 target predictions before metrics.
Fixed200 primary, target-oracle best OA secondary diagnostic only. Report
OA/AA/Kappa mean ± sample SD, per-class recall/distribution and class6 collapse
(>=95% predictions), paired changes against Hybrid depth3 and Full Joint BiDA.
No tuning/rescue loss. No expectation of stability or 83% is assumed.

Layer probes use the identical 2048-patch unlabeled deterministic sample from
joint_tokenizer_probe. Hook actual unmodified forward: tokenizer vectors,
CLS before block, attention update, after attention residual (norm2 input),
MLP update, after MLP residual (block output), final LayerNorm/CLS.
Record norms, direction concentration and cosine to local classifier w6.
Earlier-stage w6 cosine is descriptive; final LayerNorm changes coordinates.
Additionally project each early vector through the frozen final LayerNorm and
head as an auxiliary lens, not an actual early-exit classifier or selected model.
Initial CLS is identical across samples by design; concentration1 is not collapse.
Probe saved checkpoints, not live stochastic minibatches; no backprop or updates.
Verify eval hooks preserve logits exactly and saved checkpoint hashes unchanged.
Depth1 failure would disprove deeper blocks as necessary for failure, but would
not uniquely prove an interface defect: training co-adaptation remains a confound.

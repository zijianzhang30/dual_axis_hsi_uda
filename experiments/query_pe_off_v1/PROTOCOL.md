# PE-off causal contrast: stress seed 2101 only

Frozen before intervention training or new target outcomes. Control is the
completed Query + Class Query depth-0 Full Joint BN pipeline at seed 2101.
Remove only the addition of fixed spatial sinusoidal positional encoding.
Keep the PE calculation, all initialized tensors, source/target data loaders,
batch shapes, dropout shapes/order, class-query readout, joint BN momentum
0.19 and differentiable target moments unchanged. PE consumes no random draws;
thus removing its addition does not alter the loader/dropout RNG schedule.
The historical control did not log batch hashes; identical historical batches
are supported by identical seeded code/RNG paths, not direct hash comparison.

Assert original BiDA, full transplant and final depth-0 pipeline initial hashes
match the existing control. Assert shared pipeline, A1 and BN script hashes
match. No optimizer/scheduler/seed/architecture or classifier changes.
Houston13 -> Houston18, normband, 13x13 patches, batch 128/domain, source split
random_state 23, SGD 0.01, source CE only, dropout 0.1, 200 epochs, seed 2101.

Save epochs 10:10:200. Generate and hash all predictions and manifest before
post-hoc GT metrics. Fixed epoch 200 is primary; oracle max target OA is a
separate diagnostic and never selects further settings. Report OA/AA/Kappa,
all class recalls, class distribution and class-6 share; >=95% is the existing
class-6-collapse threshold, not a general stability guarantee.

Repeat the unchanged internal probe at all saved epochs on the same fixed
2048 unlabeled source and target patches. Compare classifier-z concentration,
attention residual source/target concentration and centered variation to the
saved PE-on probe. Directional changes alone do not prove class separation.
Run no other seed/component ablation; do not tune against target outcomes.

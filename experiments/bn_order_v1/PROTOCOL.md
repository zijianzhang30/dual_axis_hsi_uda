# BN order V1: paired causal control

Frozen before new training. Seeds 2100/2101/2102, original BiDA tokenizer,
depth 3, dropout 0.1, source CE, 200 epochs. Same data, split, minibatch
sequence, parameter initialization, SGD 0.01, and unused-EMA RNG consumption
as the completed paired original Mixed-BN controls.

## Reverse-order arm

Tokenize target first, then source, so both stem BN layers update target then
source with native momentum 0.1. Cache tokens and pass them to the original
downstream forward in the original source/target branch order. Tokenization
has no stochastic layers. Preserve the original order of all dropout calls,
attention branches, source CE, and optimizer steps. Do not interchange source
labels, objectives, or branch masks. Current training batch statistics are
independent of running buffers, so source training and learned weights must
replay the paired baseline exactly. Assert epoch CE error <= 1e-7 and exact
final non-BN-buffer tensor equality.

After 200 epochs freeze and hash predictions for epochs 10:10:200 before
reading target GT. Fixed epoch 200 is primary. Report all three paired OA/AA
changes, class-6 shares, per-class recall, and sample SD. Collapse threshold
is >=95% target predictions assigned to class 6. No checkpoint/ratio search.

## Joint-Batch comparison

The completed Fixed-Mixture V1 implementation already performs native BN on
equal-sized concatenated source/target activations at both stem BN layers. It
is Joint-Batch BN with one momentum-0.19 update per pair. The 0.5/0.5 mixture
mean and variance (including between-domain mean difference) are exactly the
joint-batch moments. Verify equivalence against a separately implemented
fully concatenated stem for output tokens, source gradients, target gradients,
and running buffers. Reuse its frozen three-seed results; do not create a
fictitious distinct arm or repeat identical training.

Sequential update after source then target is
0.81*old + 0.09*source + 0.10*target;
reverse order is 0.81*old + 0.10*source + 0.09*target.
Thus order reverses a modest last-domain bias (effective fresh weights
47.37/52.63), without adding between-domain variance or shared normalization.
The reverse experiment isolates this order effect. Its comparison against
Joint-Batch does not individually isolate pooled training statistics,
between-domain variance, or gradients through target moments. Resolving those
requires a further predeclared intervention; do not overclaim this study.

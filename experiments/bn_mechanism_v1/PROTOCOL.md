# BN mechanism V1: buffers versus training statistics versus target gradients

Frozen before any new training. Seeds 2100/2101/2102, original BiDA tokenizer,
depth 3, dropout 0.1, source self CE, 200 epochs, fixed epoch-200 endpoint.
All data, split, initialization, minibatch RNG, unused EMA initialization,
source CE, and SGD settings match the completed controls.

## Buffer-only balanced

Training forward uses the original independent source/target native BN batch
normalization and downstream dropout order. Capture each layer's detached
domain activation mean and biased variance. After backward, replace the native
sequential buffer writes by one momentum-0.19 pooled update based on equal
domain weights. Include between-domain variance and pooled unbiased correction:
mu = (mu_s + mu_t)/2;
var = (var_s + var_t)/2 + (mu_s - mu_t)^2/4;
running_var uses var*N/(N-1), with N the combined channel sample count.
Source/target training normalization and gradients remain as original Mixed BN.
Assert exact epoch-CE replay and final non-buffer weight equality to original
control. Buffers-only balancing does not imply source/target intermediate
activations equal Full Joint, because prior layers normalize differently.

## Detach-target moments

Use native joint BN on concat(source activation, detached target activation)
at both BN layers, then continue source/target tokenization and downstream
branches in the original order. With source-only CE, detaching the direct target
half as well as its moment contribution is equivalent to stopping target
moment gradients: target logits have no loss or coupling-to-source self path.
It preserves Full Joint's forward values for identical weights and inputs while
removing gradients back through the target activation half of each BN layer.
Forward values will diverge after optimization because learned weights differ.
Verify target input gradient is absent and independently check the source
gradient against an explicit joint mean/variance formula with target moments
detached. Shared affine parameters remain trainable and receive source loss.

Both new arms use the same effective EMA rate 0.19 per paired step as Full
Joint. No ratios, rates, freeze epochs, or checkpoints are tuned.

## Evaluation and interpretation

Save checkpoints every 10 epochs; after training freeze all 20 predictions and
hashes before target metrics. Target labels enter neither loss nor selection.
Report fixed endpoints OA/AA/Kappa, per-class recall, class-6 collapse count
(>=95% of target predictions), sample SD, and paired changes versus original
Mixed BN and Full Joint. Full Joint is reused from BN stabilization V1.

For descriptive interpretation, call an arm close to Full Joint only if mean
OA and AA each differ by at most two points, no individual seed's OA differs
by more than two points, and all three seeds avoid class-6 collapse. This is a
preregistered descriptive threshold, not a statistical equivalence test.

If Buffer-only is close, inference buffers may explain most of the gain. If
Detach is close while Buffer-only is not, shared training normalization is
sufficient within these seeds. If Full Joint exceeds Detach, target-side
moment gradients materially affect the observed optimization path; do not
generalize necessity from only one seed or one dataset pair.

# BN stabilization V1: seeds 2100/2101/2102

Frozen before any new training or target outcomes. Use the original BiDA
tokenizer, stem, depth 3, dropout 0.1, and source self CE.

## Arms

1. Original Mixed BN: reuse the frozen BiDA-self epoch-200 controls for seeds
   2100/2101/2102. Source then target each update native BN with momentum 0.1.
   Training normalization uses each domain's own current batch statistics.
2. DSBN: shared convolution, tokenizer, transformer, classifier, and BN affine
   parameters; separate source and target running moments/counters. Each domain
   trains with its own batch moments and updates its own buffers with momentum
   0.1. Source validation uses source buffers, target inference uses target
   buffers. The target branch has no loss. Thus source CE, dropout RNG, and
   learned parameters must reproduce the paired original control; check every
   epoch and the final tensors. This is a buffers-only intervention.
3. Fixed-Mixture: at each of the two stem BN layers, compute pooled moments on
   equal-sized source/target activation batches. Both domains normalize with
   the same mixed current moments; one shared running-state update per step.
   mu = (mu_s + mu_t)/2 and
   var = (var_s + var_t)/2 + (mu_s - mu_t)^2/4, where within-domain variances are
   biased population estimates. Use native BN on concatenated activations to
   implement exactly this equal mixture. Running variance uses the native
   pooled unbiased estimate. Momentum is fixed at 0.19 = 1-(1-0.1)^2, matching
   the original two updates per step's forgetting rate. Gradients propagate
   through pooled moments, including target activations; no target label or
   explicit target loss is used. This intervention changes training
   normalization and therefore learned weights, as well as running buffers.

All models start from the exactly matching paired control initialization.
Instantiate the same unused EMA model before intervention to preserve RNG.

## Fixed training and evaluation

Houston13 to Houston18, normband, 13x13 patches, batch 128 per domain, fixed
source split random_state=23, seeds 2100/2101/2102, SGD 0.01 without momentum
or schedule, 200 epochs. Keep the released GT-mask candidate-center protocol;
target label values enter neither loss nor checkpoint selection.

Save checkpoints every ten epochs. Train all epochs before target inference.
Freeze logits, predictions, and checkpoint/prediction hashes for all twenty
checkpoints before computing target metrics. Fixed epoch 200 is the sole
primary endpoint. Trajectories are for stability diagnostics and must not
select a freeze point, checkpoint, EMA rate, mixture weight, or another arm.

## Predeclared criteria

Define class-6 collapse as at least 95% of target predictions assigned to
class 6. Report its count across three seeds first, followed by OA, AA, Kappa,
per-class recall, prediction shares, and sample SD across seeds. Also report
all-class recall coverage and the complete checkpoint trajectory.

A candidate is eligible for a further confirmation experiment only if all
three epoch-200 seeds avoid class-6 collapse, mean OA is no lower than the
paired original mean, mean AA does not decrease, OA sample SD decreases, and
mean OA on the two previously noncollapsed control seeds (2100/2102) loses no
more than two points. Report every criterion, even if only some pass.

Do not promote an arm on its best seed or target-oracle epoch. Do not tune
mixture weight, BN momentum, or a freeze point during V1.

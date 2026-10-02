# Retrospective joint-BN tokenizer probe

Read-only comparison of frozen Full Joint + BiDA tokenizer and Hybrid-V2,
seeds 2100/2101/2102, epochs 10:10:200. No retraining, checkpoint selection,
hyperparameter change or interventions. Existing target outcomes are known.

Use the identical deterministic, label-unstratified sample of 2048 source
and 2048 target patches, sampled without replacement with probe RNG 93017
from the source nonaugmented training set and target evaluation set. Discard
labels; preserve original candidate masks. Source splits must be identical
across seeds. Batch 128 per domain, fixed sample order for every checkpoint.

Two probes: eval with saved running statistics and self inference; paired
joint-training mode with current moments and dropout disabled to isolate
normalization/representation. No backward/optimizer. Restore all checkpoint
state tensors between modes and require exact state equality after restoration.
Training-mode observations are reconstructed probes, not recorded historical
training activations; no data augmentation or historical minibatch replay.

Collect each domain's per-channel BN-output mean/variance before ReLU at both
layers, token norms and normalized direction concentration, pre-LayerNorm
CLS norms, classifier-input z norms and direction concentration, cosine to
each classifier weight vector, logits, class-6 margin over the maximum other
logit, class-6 top-1 share. Classifier z is after LayerNorm: its norm alone
cannot establish absence of feature drift. Weight cosine is model-local;
raw coordinate-wise cosine between independently trained models is not used.
Record classifier weight norms/bias and BN affine scales too.

Outputs are aggregate JSON and trajectory figures. Existing full-target
class shares/OA/AA are used only to place probe signals in context. Differences
are observational, not proof of causal amplification by the tokenizer.
No normalization/residual fix follows without a separate authorized experiment.

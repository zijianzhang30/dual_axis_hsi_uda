# Pavia University → Pavia Center: frozen normalization transfer test

Frozen before inspecting any new target predictions. This is an internal paired
transfer test, not a reproduction of an official Pavia evaluation protocol.

## Arms and endpoint

- Original: BiDA-self, original sequential source/target mixed BN (momentum 0.1).
- Joint: identical model, existing differentiable Full Joint BN implementation
  (`bn_stabilization_v1/normalization.py`, `fixed_mixture`, momentum 0.19).
- Seeds 2100, 2101, 2102; 200 epochs. Fixed epoch 200 is the sole primary endpoint.
- Save every 10 epochs. Infer target only at epoch 200, freeze predictions and
  checkpoint hashes before calculating target metrics. No target model selection.

## Locked recipe

BiDA stem and original spatial-softmax tokenizer; depth 3, dim 64, four tokens,
dropout 0.1, 13×13 reflected patches, batch 128 per domain, SGD 0.01 without
momentum or schedule; source-self CE only. No SceneShift, Query tokenizer,
distillation, MMD, consistency, pseudo-labels, or other adaptation objectives.
Only dataset-dependent band count changes from 48 to 102; classes remain seven.
Instantiate the existing generic BiDAnet directly because its dataset-name
factory does not register Pavia. Preserve the unused second model initialization
from Houston to keep the same RNG recipe. No hyperparameter search.

Data: existing `TGRS_MLUDA-2024/datasets/Pavia/{paviaU,pavia}.mat` and
their existing `*_gt_7.mat` mappings. No new class remapping or spectral alignment.
Copy the Houston `normband` operation exactly (per-pixel spectral L2 normalization,
not per-band z-scoring), separately per full unlabeled image; background is -1.
Reuse the source 95%/5% stratified split with random_state=23, existing HSIDataset
augmentation/boundary rules and shared loader generator. Zip the source and
target loaders and skip unequal batch sizes as in Houston. Pavia has more source
labels, so updates per epoch and wall time need not equal Houston. No PCA/ILDA.
The existing labeled-target mask defines unlabeled target centers, as in Houston;
target class values are never used in training. This mask assumption must be
disclosed in any eventual external comparison. Source validation is monitoring
only; overlapping neighboring source patches are not an independent spatial test.

## Reporting and decision

Report each seed and sample mean ± std (ddof=1): OA, AA, Kappa, per-class recall,
all prediction ratios, class-6 ratio, source-val OA, parameter count and timing.
Collapse is any predicted class taking ≥95% of samples (class identity need not
transfer from Houston); also report zero-recall classes. Report paired Joint−Original
OA/AA/Kappa. A positive replication requires ≥2/3 positive paired seeds in the
same metric, a mean improvement ≥1 pp in OA or AA, and no increase in collapse
count. Three seeds are diagnostic evidence, not proof of statistical significance.
Record paired initialization, sample order and pre-forward dropout RNG hashes.
Do not change the method in response to target results from this test.

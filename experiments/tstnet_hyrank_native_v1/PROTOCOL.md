# TSTnet native HyRANK reproduction

Task: Dioni -> Loukia. Run full official TSTnet, not BiDA or our modules.

Official source commit: f663d42bbb63fad126a86e6b259ef70412715ca2.
Immutable official source, runnable source, compatibility diff, configuration,
environment and logs are under:
`/nas1/zhangzj26/dual_axis_hsi_uda/results/tstnet_hyrank_native_v1/`.

## Frozen settings

- Source 5% stratified split, native random_state=23, re_ratio=10.
- Official screened out68 labels, 176 bands and 12 classes.
- Native pixel-spectrum L2 normalization, 12x12 symmetric-padded patches.
- Batch 100, native source/target loader iteration and target recycling.
- 500 epochs, 10 sequential trials, seed 1233 set once as in native entry.
- Paper HyRANK learning rate 0.001, lambda1=1, lambda2=0.1.
- Native per-epoch recreation of SGD, momentum 0.9 and weight decay 1e-4.
- Native CNN/GCN CE, linear MMD, GOT and consistency loss unchanged.
- Native flip/radiation augmentation remains enabled in test dataset too.
- Target GT used for native candidate sampling and diagnostic accuracy, not
  the optimization loss. This is native-paper reproduction, not target-blind
  evaluation of our method.
- Native evaluation grid epochs 1,11,...,491 and native target-best selection.
  Do not change it to 10,20,...,500 or call extra validation during training.
- Save final-500 state without another forward or extra loader iteration;
  final state has no new target score in this run.

## Compatibility and limitations

Only source fixes: MAT v5 reading fallback (NAS files are v5, native reader
is HDF5-only) and deprecated np.bool replaced by bool. MAT arrays retain their
physical axis order. Raw source is preserved; diff is saved.

Modern environment is torch 2.4.1 / CUDA 12.1 / torch-geometric 2.6.1, whereas
authors listed torch 1.6 / CUDA 10.2 / torch-geometric 1.6.3. Dependencies added
in a separate NAS environment; shared torch environment was not changed.
No claim of bitwise reproduction of the authors' old environment.

Preflight runs in a separate process: real data dimensions, returned-array
unit norms, finite complete loss and backward checked. No training RNG replay
or extra loader iteration is injected into the native 10-trial sequence.

Instrumentation wraps original functions without rewriting their bodies;
metrics derive from predictions already returned by native test. Save OA, AA,
Kappa, recalls, confusion matrices, class shares, collapse and zero recalls.
No extra predictions are generated during training. Models/losses unchanged.

Paper Table IX (PDF page 11): OA 63.31 +/- 3.05%, Kappa 0.5537 +/- 0.0382.
Fig. 11's 63.47% is an individual illustration, not the table mean.
Compare native-best 10-trial summary to paper, with both sample and population
standard deviations reported. Do not transfer target-best selection to our
formal fixed-epoch model evaluations or tune settings to match the paper.

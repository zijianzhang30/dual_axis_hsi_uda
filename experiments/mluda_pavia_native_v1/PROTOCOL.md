# Native MLUDA Pavia reproduction

Run the complete, unchanged `MLUDA_up.py` from an independently hashed snapshot
of `/home/zhangzj26/TGRS_MLUDA-2024-main`. This is not a BiDA experiment.
Do not modify existing experiments, the original official directory, or datasets.

## Frozen published configuration

- PaviaU -> Pavia City, existing author-format 102-band / 7-class files.
- Native per-domain whole-cube sklearn z-score, then native ILDA.
- Native constant padding, patch 11, 180 source samples per class, batch 32.
- 100 epochs; original loop executes `range(1, len_source_loader)` = 38
  updates per epoch, not 39. Preserve this behavior.
- Original SGD 0.001, momentum 0.9, weight decay 0.0005, recreated each epoch.
- Full DSANSS, source CE + 0.3 * lambda * LMMD + source/target contrastive CE.
- All ten native seeds: 1622,1322,1256,1712,1535,1670,1648,1191,1384,1436.
- Evaluation only at native final epoch 100. No additional evaluations or tuning.
- Target label values exist in the native dataset but do not enter training loss.
  Target labeled-pixel mask is used for candidate centers.

## Known released behavior, not silently corrected

The evaluation loader drops its final incomplete batch. Native OA divides correct
predictions by the full target dataset length, whereas AA/Kappa use only retained
predictions. Preserve and identify this denominator difference in reports.
The evaluation forward pairs each target batch with the final training source
batch. Native best-run/map bookkeeping is retained but does not govern any new
experiment or version choice. The native ten-run summary uses population std;
additionally report sample std from the same final scores.

ILDA includes PCA before the first native set_seed call. The upstream script does
not fix this preprocessing RNG. Record the incoming RNG state without reseeding;
do not claim fully deterministic preprocessing or paper-score reproduction.

## Execution and artifacts

Queue behind both existing pavia_zscore_v1 training processes. Start only when
both have finished successfully and frozen both final prediction manifests.
Use physical GPU 6 through CUDA_VISIBLE_DEVICES=6, retaining native .cuda calls.
Do not use GPUs occupied by other jobs or stop any task. One unchanged script
executes its original ten seeds sequentially, without retries.

Snapshot all native top-level Python source files and README; symlink only the
existing Pavia dataset into the new snapshot working directory. Save source/data
hashes, environment versions, complete stdout/stderr, initial RNG states, native
ten-run metric arrays and final-run predictions/checkpoint after runpy returns.
No training/optimizer/model monkeypatches or injected validation calls.

Paper reference: Pavia OA 91.26 +/- 0.53%, AA 92.70 +/- 0.80%; these are comparison
values, not tuning targets. No automatic additional training on disagreement.
The previously proposed BiDA normalization x sampling 12-run diagnostic is not
started by this reproduction.

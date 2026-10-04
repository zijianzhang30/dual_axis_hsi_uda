# Frozen BiDA-style extension on MLUDA datasets

This is **MLUDA数据集上的BiDA-style扩展验证**, not reproduction of the
MLUDA training protocol. Historical MLUDA scores remain separate and are not
used for cross-protocol improvement calculations. HyRANK results are preserved.

Tasks: PaviaU -> PaviaC (102 bands, existing seven-class mapping), and
Shanghai -> Hangzhou (198 bands, three classes). B is BiDA-self + Full Joint BN
+ original tokenizer; C replaces only the spatial weights with the frozen
center-relative relation MLP (hidden 32, ReLU, four scores, eps 1e-6). Values
remain the original CNN grid; no interpolation. The relation source is the
preprocessed and augmented 13x13 patch, including its center.

Both arms: dim 64, depth 3, dropout 0.1, Full Joint BN momentum 0.19,
source CE only, batch 128 per domain, SGD lr 0.01, no momentum/scheduler,
200 epochs, seeds 2100/2101/2102. Instantiate the historical unused second
model. Relation MLP initialization uses the historical isolated CPU RNG with
seed+41000; all surviving backbone tensors and training RNG are preserved.
No new loss, SceneShift, center branch, gate, pseudo-label or tuning.

Use historical normband: per-image global scaling followed by per-pixel
spectral L2 normalization, NOT MLUDA per-band standardization. Source split
is 95/5 stratified, random_state=23 (native sample_gt). Reflect pad radius 6;
retain native strict center boundary rule. The labeled target mask defines
unlabeled training centers; target label values are replaced by zero before
training iteration. Both arms share source split, dataset indices, generator
recipe and target evaluation mask. Shanghai DataCube1/gt1 is source,
DataCube2/gt2 is target, labels 1/2/3 map to zero-based Water,
Ground/Building, Plant, according to class_name/common_class in DataCube.mat.

Retain pavia_joint_v1 training procedure, including source validation at epoch
1 and every 10 epochs, zip(source,target), unequal batch skipping and a single
shared loader generator. Save checkpoints every 10 epochs; target inference
only after epoch 200. Save predictions and hashes before reading labels for
metrics. Never select or stop by target accuracy. No target-oracle search.

Reuse Pavia B only after data/source code/protocol hashes, seeds, initialization,
dataset indices/counts and actual epoch batch/RNG stream checks agree. Any
failure stops the affected task without adding baseline retraining. New tasks
are exactly three Pavia C and six Shanghai B/C runs. Results and code snapshots
go to a new NAS directory; preserve all historical files. Preflight must pass
dimensions, grid, finiteness, gradients, and nonzero target-input gradients
from source CE through differentiable joint statistics for both arms/tasks.

Report fixed200 OA/AA/Kappa, per-class recall, prediction distribution,
single_class_collapse (largest prediction share >=0.95), zero-recall count and
classes, parameters, source validation and training cost. Aggregate sample SD
(ddof=1), paired C-B for each seed. No automatic winner selection or new runs.

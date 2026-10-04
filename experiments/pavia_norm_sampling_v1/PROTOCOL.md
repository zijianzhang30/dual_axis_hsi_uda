# Frozen Pavia normalization × source sampling diagnostic

P0: repaired pixel spectral L2 / uniform replacement sampling.
P1: per-band z-score / uniform replacement sampling.
P2: repaired pixel spectral L2 / inverse TRAIN-class-frequency replacement sampling.
P3: per-band z-score / inverse TRAIN-class-frequency replacement sampling.

All twelve runs: seeds 2100/2101/2102, BiDA-self + Full Joint BN (momentum
0.19) + original tokenizer, depth 3, dropout 0.1, 13×13 patches, batch 128,
SGD 0.01 without momentum/weight decay/scheduler, source CE only. Exactly
290 updates / 37,120 paired samples each epoch; 200 epochs. Keep repaired B's
95/5 stratified split (random_state=23), all 37,236 source training centers,
1,960 validation centers, and 39,348 target evaluation centers.

Normalization starts independently from original cubes. L2 reuses the repaired
loader. Z-score exactly follows the official MLUDA Pavia base loader's
sklearn.preprocessing.scale over the complete raw scene (background included),
without ILDA. No 180-per-class pool restriction. Reflect padding and native
strict boundary condition remain unchanged. Target GT nonzero mask defines
candidate centers; target class values do not enter training or sampling.

Epoch plans use private streams: source seed+10000 (uniform) / seed+20000
(balanced), target seed+30000, source augmentation seed+40000, target
augmentation seed+50000. Each epoch stream seed adds 100000×epoch. Source
uses torch.multinomial with replacement; target uses a private permutation,
first 37,120 centers. Native augmentation law: identity probability 0.5;
horizontal flip, vertical flip, rotations 90/180/270 each probability 0.1.
Plans carry augmentation codes by draw position. Norm pairs share source
plans; all four share target and augmentation plans. Loader worker generators
are separate from model RNG. Model initialization preserves native student
and unused second-model construction. This is a NEW four-arm RNG protocol,
not reuse of old B results. Compare center/code sequences, not normalized
input hashes. Log actual consumed indices, class frequencies, unique coverage,
update counts, and model RNG hashes. No additional loader audit iterations.

Source validation at epoch 1 and multiples of 10. Save checkpoints every 10
epochs. After training save/hash all target predictions before scoring.
Fixed epoch 200 is formal; the 10-epoch grid is descriptive only. No target
best selection, tuning, early stopping, version selection or added experiments.
Report OA/AA/Kappa, seven recalls, confusion matrices, prediction shares,
3→7 and 4→7 counts / true-class denominators, >=95% single-class collapse,
zero-recall count/classes, parameter count, times and actual coverage.

Results are new NAS artifacts; prior experiments and active tasks are untouched.
Implementation checks and all twelve frozen configs must precede training.

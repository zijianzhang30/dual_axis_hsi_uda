# Repaired Pavia checkpoint trajectory analysis

Read-only analysis; no training, model/loss changes, MLUDA-aligned runs, sampling
2x2, BN recalibration, class weights, best-checkpoint selection or extra tasks.

Only `results/pavia_l2_fixed_v1/{B,C}_{2100,2101,2102}`. Evaluate all existing
epoch10,20,...,200 checkpoints. Verify config/code/data/checkpoint hashes against
the frozen repaired-L2 manifests. Missing checkpoints are recorded, not rebuilt.
Use the repaired loader, fixed row-major native boundary mask and 39348 pixels.
Save logits/predictions and hashes before scoring. Report OA/AA/Kappa, all seven
recalls, confusion matrices, prediction proportions, single-class collapse,
zero-recall classes, 3->7 and 4->7 counts and percentages of each true class.
Pair C-B by same seed/epoch, with missing-pair reasons retained.

No target-oracle selection. Descriptive comparisons use predeclared windows:
early epochs10..50, middle60..140, late150..200, plus fixed10/50/100/150/200 tables.
Recall ranges describe the observed grid only, not unseen epochs. No recall
threshold is used to select models, checkpoints, parameters or class weights.

Stage probes are fixed at epoch200 for B/C and all three seeds, if available.
Reuse existing definitions: CNN activation at center(6,6), spatial mean of that
activation, mean of four semantic tokens, target CLS after final LayerNorm(z).
Same per-class source-val/target centroids, norms and cosine as the existing
B2100 probe. First validate B2100 reproduction, then describe cross-seed
patterns. Centroids/readouts differ across stages; nearest-class changes are
descriptive and not a causal attribution to CNN/tokenizer/Transformer.

Inference is a separate process, eval/inference_mode only; check state hashes
before/after. No iteration of training loaders. No historic raw experiment is
scored or used as reference. CPU preparation/inventory is safe while GPUs busy.
GPU queue defaults to waiting for authorized GPUs1/2/3/6 to be idle, unless the
user explicitly authorizes shared inference on GPU2/3.

Save inventory, source snapshots, logs, complete JSON/CSV trajectories, paired
deltas and a concise RESULTS.md in a new NAS directory. No automatic follow-up.

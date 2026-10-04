# Fix flipped anchor coordinates only: TSTnet-style center-relative tokenizer

Historical six tasks and dependency files remain untouched. Fix is scoped to
two independent new C and C_adapt tasks, seed1233/500 epochs, same frozen
source pools/target centers/optimizer/native evaluation grid. No new tuning.

Native loader augmentation flips a coordinate marker together with data and
label. Marker starts at the original labeled patch pixel (6,6) for size12.
It becomes (6,5), (5,6), or (5,5) when appropriate. Metadata depends only on
geometry, not class values. Original image/label tensors and RNG draws remain
unchanged, including native radiation noise and test-time augmentation.

The marker is not an input band. Prefetcher transfers it as separate coordinate
metadata; model uses it only to select the anchor spectrum in the already
augmented patch. Relation MLP, epsilon, CNN values and all parameters unchanged.
No extra randomness is consumed. Do not change to patch13, alter label center,
disable augmentation or add losses to fix geometry.

Both paired source/target coordinates are supplied to the student, and to EMA
in C_adapt. Both evaluation arguments receive the same actual coordinate.
Missing metadata is fatal in real training/test; synthetic unaugmented
preflight alone allows (6,6) defaults. Original forward interface/loss AST is
preserved by an explicit geometry-context wrapper, cleared after every call.

Verify all four flips for sizes12 and13; image, label and NumPy RNG identical
under 200 native random/noise replays; real constructor indices/labels/RNG
identical; unflipped scores exactly identical; flipped spatial weights match
the flip of original weights within1e-6. GPU preflight verifies joint target
gradient, student/EMA interfaces and MMD off/on at epoch100/101.

Before first training update, assert configuration and initialization hashes
match corresponding original C/C_adapt. Actual consumed source-label batch
hashes must also match existing history at corresponding epochs. Record
actual coordinates per epoch. Fix does not modify native target GT sampling.

Report fixed500 versus original C/C_adapt and versus unchanged B/B_adapt.
Report recalls, distributions/collapse, cost and grid trajectory; no target
best selection of formal endpoint, early stopping or automatic added seeds.
This measures geometry correction, not a new model innovation. Do not assume
it explains all deficits or adapt-loss AA degradation before results exist.

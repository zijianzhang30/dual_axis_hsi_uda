# Tokenizer x adaptation objective, fixed Full Joint BN

Single seed1233 HyRANK protocol diagnosis; add exactly two tasks, reuse
existing source-CE B/C only after configuration matching before first update.

| Tokenizer | Source CE | Same released BiDA adaptation objective |
| --- | --- | --- |
| Original | existing B | new B_adapt |
| Center relative | existing C | new C_adapt |

All use Full Joint BN momentum.19; same CNN, self/coupled forward interface,
depth3, dropout.1, dim64, K4. Center-relative implementation unchanged except
necessary 176-band adaptation. Both student AND EMA use the matching BN and
tokenizer; teacher parameters detached. EMA independently initialized as in
BiDA; center-relative teacher MLP uses fixed isolated seed1234 while student
uses the frozen seed1233 relation initialization. Added MLP installation does
not advance native CPU/CUDA RNG. EMA updates native parameters only, with
independently updated teacher Joint BN buffers.

Adaptation is the SAME unmodified released BiDA batch AST in both new arms:
source CE, bidirectional distillation, EMA consistency, MMD after epoch100.
lambda1=.1, lambda2=1, EMA decay=.999. Do not fix native teacher gradients,
double softmax or MMD self-pair. Native make_loss including unused CenterLoss
initialization retained. The teacher calls consume native dropout RNG, so the
adaptation-vs-CE contrast is a training-package contrast, not a claim of exact
per-forward dropout RNG counterfactual equivalence. B_adapt and C_adapt have
the same teacher-call policy. Added losses are NOT proposed innovations.

Frozen TSTnet-style setting: 5% stratified source/random_state23, repeat10,
raw-cube pixel L2, patch12/symmetric padding, batch100, 100 steps/epoch,
500 epochs. Native flip/radiation and test augmentation preserved. Source and
target native centers and labels hashes must match CE controls. Target GT
is used for native stratified center sampling and evaluation only, never loss.
SGD reset each epoch, lr=.001/(1+10*epoch/500)^.75, momentum.9, decay1e-4.

Record fixed500 OA/AA/Kappa, recalls, prediction shares, collapse>=95% and zero
recall classes, parameters/cost. Grid1,11,...491 best is diagnostic only;
formal endpoint never selected by target. Terminal evaluation after training.
Report B_adapt-B, C_adapt-C and C_adapt-B_adapt, with C-B and interaction.
No target early stop, tuning, new modules or automatically added seeds.

Preflight separately checks joint CE target gradient and the full published
batch100 loss at epoch100/101, MMD off/on, finite gradients and matching
student/teacher parameter interfaces. It cannot contaminate training RNG.
Do not modify or stop TSTnet, original-BN full BiDA or existing B/C runs.
Archive components and configs in NAS new directory; dependency snapshot is
a read-only link to the already archived BiDA source, not a migration.

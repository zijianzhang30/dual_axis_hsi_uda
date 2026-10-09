# Capacity-matched G extension, independent of existing A–F

Do not interrupt Pavia A–F or modify any old artifact. No MLUDA modifications,
new loss, Joint BN variant, gate module or HyRANK training. G is the sole new arm:
center-relative (patch-center)/(population spatial std+1e-6), same as C;
Linear(bands,32) → Identity → Linear(32,4), no ReLU/activation/norm/dropout inside
scorer. Same spatial softmax, four64-D tokens, original13×13 CNN values and
same Transformer, CLS/PE/head. Keep original Full Joint BN implementation intact.
Same independent CPU seed+41000 initialization; C/G scorer tensors and complete
initial model state exactly equal, despite different stateless activation.
Nominal scorer parameters: Houston1700, Pavia3428 for BOTH C andG; total model
Houston378011, Pavia628571. Extra affine biases are null under spatial softmax,
so do not interpret expected zero bias gradient as a detached graph.

Frozen source-self CE, SGD.01/no momentum/schedule, bs128,200epochs, same loader
protocol as tokenizer_mechanism_v2; save every10, formal final200 only. Seeds
2100/2101/2102. Compare exact sample indices, original/scorer/full initial hashes
and EACH epoch batch/augmentation/pre-forward RNG to existing fixed C. No
retraining of A–F, checkpoint/oracle search, target-best, tuning or extra seeds.
Target values enter only final evaluation and explicitly authorized stage decision.

Audit C/G exact parameter counts/layer lists, exact relation/softmax/value grid,
legacy C replay, affine-composition equivalence of G, scorer weights/SGD update,
nonzero target-input gradient through unchanged Joint BN, no detach, initial
CPU/CUDA RNG invariance. Use only new local/NAS directories, no old writes.

Sequence: wait for successful completion AND original final summary of Pavia
A–F (including their weight diagnostics); preserve their status/results. First
write independent A–F summary/conclusions before any formal G. If none of
Pavia C−B/C−E/C−F meets its originally frozen support criterion, stop and report
negative A–F evidence; no automatic G or new design. Otherwise run Houston G
only three seeds. This conservative preregistered stop implements the user's
instruction to report negative Pavia A–F evidence before continuing.

Houston C−G continuation: mean OA AND AA positive, and at least2/3 paired seeds
have BOTH OA andAA positive ("方向基本一致"). If false, stop and report, no
Pavia G. If true, run identical G on Pavia only same three seeds. No HyRANK.
All new G predictions frozen before stage scores. Preserve original A–F mean,
checkpoint and paired comparisons; new reports in this independent directory.

Report A/B/C/F/G OA/AA/Kappa mean±sample+populationSD; C−A/C−B/C−F/C−G per-seed
and paired meanSD, recall/confusion/shares, collapse(max prediction share≥.95),
zero-recall, parameter counts/time. C/G strict scorer count and initial-tensor
proof. G final entropy/diversity/tokens and same-index AND24-permutation-aligned
cosine/JS vs originalA weights; againstC as an additional requested direct check.
Use original weight-analysis metric definitions. Cross-checkpoint features
co-adapt, so weight differences are diagnostic, not a pure causal scorer proof.

Conclusions restricted to center-reference usefulness, std-normalization
usefulness, and nonlinearity under matched depth/width/nominal parameter count.
C/G positive scores support utility, not universal mathematical necessity;
three seeds do not establish statistical significance. Keep D/E analytic fact:
shared linear W,b scores differ only by spatial constant canceled by softmax.
Separately trained D/E score differences are trajectory differences, NEVER
linear center-reference benefit. No automatic new module if evidence is negative.

NAS new folder: /nas1/zhangzj26/dual_axis_hsi_uda/results/tokenizer_capacity_g_v1.
No automatic retry. Resource schedule is independent and starts only after A–F
complete; never stop unrelated GPU jobs. Old sources/data are read-only.

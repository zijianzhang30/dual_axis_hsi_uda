# BiDA-self tokenizer mechanism V2: six fixed variants

Independent study. No MLUDA files/tasks change, no new loss, no Joint BN changes,
no new gate module or automatic method extension. Houston first, optionally Pavia
under the explicit user-authorized result gate. HyRANK synthetic minimum only.
Seeds2100/2101/2102; all six variants freshly run per eligible dataset.

A native conv_a64→4/biasFalse on CNN values.
B (patch-mean)/(population spatial std+1e-6), MLP bands→32/ReLU→4.
C (patch-center)/(same std+1e-6), same MLP.
D B relation with Linear(bands,4,biasTrue), no ReLU.
E C relation with same Linear, no ReLU.
F patch-center without std division, same MLP asC.
All spatial softmax→4 weights→4×64 tokens using ORIGINAL CNN values on same13×13
spatial grid, same CNN, same depth3 Transformer/dropout.1/classifier/CLS/PE.
New modules initialize after full native student and unused EMA with independent
CPU seed+41000, preserving original shared tensors/CPU+CUDA RNG. B/C/F scorer
initial tensors identical; D/E identical. Remove unused conv_a forB–F.

Native Full Joint BN implementation copied byte-for-byte: differentiable S/T
pooled moments, momentum.19, eval native running stats. Source-self CE only,
SGD.01 no momentum/schedule/decay, bs128/domain,200epochs. No target class values
used in loss, checkpoint selection, stopping or hyperparameter choice. Source
validation epoch1 and every10, save every10, final200 only inference/scoring.
Houston: frozen native make_loaders normband/reflectpadding/source95% split;
Pavia: frozen pavia_joint_v1 loader recipe normband/reflectpadding/source95% split.
Do not substitute official MLUDA Pavia protocol. Native source split random_state23,
shared seeded loader generator, original augmentation and boundary rule retained.
Match exact per-seed sample indices, augmented batches, pre-forward CPU/CUDA RNG
for all six variants; Houston additionally audit against historical Full Joint.
A historical numeric/state replay reported; rebuilt current A always paired
control if backend differs. Historical means never substitute paired controls.

Theory: r_p=u_p-u_center. For shared linear W,b, E scores equal D scores minus
W*u_center, spatially constant per token. Therefore D/E spatial softmax is
identical in exact arithmetic; linear bias also cancels. Floating-point tests
CPU/GPU random48/102/176 bands and odd/even patches require max error<1e-6.
Double-precision input/weight gradient equivalence checked; tiny null-bias
roundoff is not evidence of a detached graph. ReLU can make B/C non-equivalent.

Center odd13=(6,6), invariant under native flips/rot90. Even patches MUST provide
actual augmented anchor coordinates; reject guessed center. Test markers under
all rotations/flips for12/13 andC/E/F, no data augmentation changes. Formal tasks
remain native odd13, no coordinate RNG/dataset changes. Test CNN value grid,
4×64 tokens, finite/nonzero scorer weight/input/target Joint BN gradients,
SGD membership/update, original initialization/RNG invariance and legacy C
formula equality before training. HyRANK only176-band geometry/equivalence tests.

Houston→Pavia operational gate (frozen before NEW target scoring): C−B, C−E or
C−F mean OA orAA≥1pp and ≥2/3 paired positive seeds in that metric; alternatively
D/E formal epoch200 OA AND AA differ by≤.1pp in EACH of three paired seeds.
Gate uses only declared comparisons, no checkpoint search. Numerical D/E equality
alone is not the formal performance-equivalence gate. If none pass, stop and
report negative results; no extra gate/scorer/module/seeds. Pavia continues only
sameA–F and same three seeds. This is a decision threshold, not significance.

Freeze all18 predictions/checkpoints per dataset before any new target scoring.
Report OA/AA/Kappa mean±population+sampleSD, all15 pairwise contrasts/per-seed,
recall/confusion/distributions, max predicted share≥95% collapse, zero-recall,
parameters and training/inference time. No target-best metric or narrative.
Analyze actual final spatial weights: entropy raw/normalized bylog(N), off-diagonal
weight cosine/JS and token cosine/effective-rank, pairwise B–F cosine/JS, and
against trainedA original-tokenizer weights. Report same token-index AND permutation
aligned comparison (K=4 all24 permutations minimizing averageJS), because token
identities are exchangeable. Cross-checkpoint weight differences include CNN
co-adaptation forA; do not claim purely causal scorer effects. Also run same-parameter
D/E weight comparisons on real evaluation patches. No token permutation changes
training. All diagnostic data are final checkpoints only.

NAS: /nas1/zhangzj26/dual_axis_hsi_uda/results/tokenizer_mechanism_v2.
Refuse overwrites, snapshot source/data hashes, no automatic retry/extra training.
GPU0/1/4/5 one task each if resources remain available; never stop unrelated jobs.

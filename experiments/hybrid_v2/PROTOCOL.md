# Hybrid-V2: A1 query transplant plus Full Joint BN

Frozen before training. Houston13 to Houston18, normband, 13x13 patches,
batch 128 per domain, fixed source split random_state=23. Seeds 2100, 2101,
2102; SGD 0.01, no momentum/schedule, 200 epochs, source self CE only.

Use the exact previously tested Hybrid transplant: BiDA stem, four learned
queries, two-dimensional sinusoidal position encoding, A1 CrossBlock with
cross-attention and FFN, depth 3 downstream transformer, dropout 0.1. This
is a tokenizer package, not an isolated learned-query intervention.
Instantiate the unused original EMA model to preserve initialization RNG.
Assert initial BiDA hash and transplanted model hash against each existing
paired seed. Full Joint installation must not change any state tensors.

Reuse the verified Full Joint implementation: at both stem BN layers concatenate
equal-count source/target convolution outputs and call native BN once. Shared
current moments, target moment gradients enabled, momentum 0.19 per paired
step, matched to the original two momentum-0.1 updates. No ratio/rate search.

Save checkpoints at epochs 10,20,...,200. After training, freeze and hash
all 20 target prediction artifacts before collecting labels for metrics.
This evaluates the same epoch grid without injecting target evaluation into
the training RNG stream. Fixed epoch 200 is the primary endpoint. Highest
target OA on this grid is a separately labeled target-oracle diagnostic;
ties select the earliest epoch. It cannot select the formal checkpoint or
guide tuning. Report oracle OA with associated epoch, AA and Kappa.

Report three-seed mean and sample SD of fixed OA, AA, Kappa; per-class recall,
prediction counts/shares, entropy, source validation, and all frozen trajectories.
Class-6 collapse is at least 95% of target predictions in class 6; additionally
report maximum predicted-class share and recall coverage, since avoiding this
specific collapse is not proof of full stability.

Primary comparison: paired Full Joint + BiDA tokenizer fixed-200 controls.
Confirmation requires mean fixed OA greater than the control and zero endpoint
class-6 collapse; also report every paired OA/AA difference and variance.
An 81% mean or 83% score is a goal, not an assumed or tuned threshold.
Three optimization seeds and one transfer pair do not establish generality.

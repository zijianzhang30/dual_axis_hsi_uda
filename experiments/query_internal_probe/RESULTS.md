# Internal Query tokenizer probe: seed 2101

Completed retrospective eval probes for Query + Class Query and Query + Mean
Pool, at all 20 saved checkpoints per model. Each uses the same fixed 2048
unlabelled source and 2048 target patches as earlier probes. No training or
target-label grouping. Checkpoint/manifest hashes, exact hook-on/off logits,
and unchanged model-state hashes passed. BN and readout remain unchanged.

## Epoch 200 geometry

Concentration is the norm of the mean unit vector across samples, using the
spatial/query-token mean at each stage. Larger means a stronger common direction;
it does not by itself establish loss of discriminative information.

| Stage | Class Query: source | Class Query: target | Mean Pool: source | Mean Pool: target |
| --- | ---: | ---: | ---: | ---: |
| Conv2d projection, before BN | 0.913 | 0.936 | 0.937 | 0.953 |
| BN2d | 0.504 | 0.660 | 0.464 | 0.662 |
| Stem ReLU | 0.797 | 0.858 | 0.786 | 0.872 |
| + Positional encoding | 0.932 | 0.969 | 0.927 | 0.972 |
| KV LayerNorm | 0.714 | 0.915 | 0.724 | 0.920 |
| Query + attention residual | 0.427 | 0.870 | 0.404 | 0.911 |
| FFN residual / four tokens | 0.371 | 0.861 | 0.135 | 0.883 |
| Classifier input z | 0.130 | 0.867 | 0.132 | 0.885 |

There is no additional A1 spatial projection in the actual transplant. The
BiDA Conv2d/BN/ReLU supplies the reader input. Constant learned queries are
identical across samples by design; their concentration of 1 is not collapse.

## Offset-aware checks

Adding fixed positional encoding increases target concentration by 0.111
(Class Query) and 0.101 (Mean Pool), but preserves cross-sample centered RMS,
pooled centered RMS, and covariance participation rank:

| Target statistic, before -> after PE | Class Query | Mean Pool |
| --- | ---: | ---: |
| Centered RMS, per spatial coordinate | 3.390 -> 3.390 | 3.167 -> 3.167 |
| Centered RMS, pooled vector | 2.082 -> 2.082 | 1.938 -> 1.938 |
| Pooled covariance participation rank | 2.913 -> 2.913 | 3.248 -> 3.248 |

Thus PE adds a common component, not direct information destruction. Its
downstream interaction with LN/attention remains a possible cause, not proven.

Across all 20 checkpoints, attention-residual concentration is 0.426--0.506
on source versus 0.852--0.889 on target for Class Query; for Mean Pool it is
0.404--0.481 versus 0.901--0.924. This is a persistent domain-asymmetric
response, not an isolated epoch-200 event. It is also not an abrupt increase
in target concentration: attention reduces it relative to KV LN in both models.

FFN decreases target concentration at epoch 200 (0.870 -> 0.861 and
0.911 -> 0.883). Target pooled centered RMS increases (1.207 -> 1.590 and
1.020 -> 1.285), while pooled covariance participation rank decreases slightly
(3.702 -> 3.441 and 3.610 -> 3.203). These mixed indicators do not support
calling FFN the unique source of collapse.

## Cosine distributions

Full per-stage distributions (mean, SD, quantiles and histograms), including
same-coordinate cosines, are saved in the probe JSONs. The table gives random
pooled-vector pair means at epoch 200. Source-target pairs have no class matching;
high cosine here is not evidence of semantic alignment.

| Model / stage | Source-source | Target-target | Source-target |
| --- | ---: | ---: | ---: |
| Class Query / stem ReLU | 0.642 | 0.739 | 0.446 |
| Class Query / + PE | 0.872 | 0.940 | 0.838 |
| Class Query / attention residual | 0.194 | 0.759 | 0.121 |
| Class Query / FFN residual | 0.149 | 0.742 | 0.100 |
| Mean Pool / stem ReLU | 0.625 | 0.763 | 0.456 |
| Mean Pool / + PE | 0.863 | 0.946 | 0.838 |
| Mean Pool / attention residual | 0.178 | 0.831 | 0.163 |
| Mean Pool / FFN residual | 0.029 | 0.780 | 0.031 |

## Conclusion and boundary

The reader separates source directions far more than target directions. PE
creates a shared offset; KV LN and cross-attention respond differently across
domains; FFN does not monotonically worsen target concentration. The present
probe does not uniquely distinguish PE, LN, learned attention or their interaction.
Do not infer a single causal culprit from the largest raw concentration or from
variance reductions across 169 spatial positions -> 4 learned tokens.

No component ablation was trained: the preregistered condition of identifying
one component was not met. A possible next *hypothesis test*, requiring an
explicit choice, is PE-off only on seed 2101, retaining Class Query, Full Joint
BN and every other component. This would test the PE/LN/attention interaction,
not confirm that PE has already been shown to be the cause.

Artifacts: `results/query_internal_probe/query_class_2101.json` and
`results/query_internal_probe/query_mean_2101.json`.

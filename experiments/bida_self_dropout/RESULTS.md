# BiDA-self dropout decomposition: seed 2100

Both arms use mixed source/target BN, three transformer blocks, source CE, and
fixed epoch-200 target self inference. The intervention changes every
`torch.nn.Dropout` probability from 0.1 to zero after verifying the complete
initialized model hash. No trainable parameter or other setting changes.

| Dropout | Target OA | AA | Kappa | Epoch-200 source-val OA |
| --- | ---: | ---: | ---: | ---: |
| 0.0 | **80.25%** | **66.53%** | **0.651** | 77.17% |
| 0.1 | 78.41% | 66.32% | 0.627 | 79.53% |

Dropout 0.1 minus dropout 0.0 is **-1.84 OA points**, **-0.20 AA points**, and
**-0.024 Kappa** at the fixed endpoint. The OA difference is below the
predeclared two-point material-effect threshold, and disabling dropout does not
reduce AA. This seed therefore does not support dropout regularization as the
main explanation for BiDA-self's performance; dropout 0 is slightly better at
the target-blind endpoint.

All 20 checkpoints and target predictions on the released-code 10-epoch grid
were frozen before target labels were collected. The separately labeled target-
oracle diagnostic peaks at epoch 90 with **81.54% OA**, **71.50% AA**, and
**0.684 Kappa**. It is 1.29 OA points and 4.98 AA points above epoch 200. This
oracle result describes training dynamics only and does not replace the formal
epoch-200 result.

Source validation does not identify the target optimum. Its best grid point is
epoch 20 (85.83% source OA), where target OA is 80.09%. The target optimum is
epoch 90, where source OA is 81.10%. Epoch 200 has only 77.17% source OA but
80.25% target OA. This supports reporting the frozen target trajectory for
released-code comparison while keeping a target-blind endpoint as the primary
result.

Dropout 0 retains nonzero recall for all seven target classes at epoch 200. Its
largest changes relative to dropout 0.1 are higher recall for classes 1
(+2.66 points), 2 (+4.98), 3 (+1.67), 6 (+3.90), and 7 (+1.73), offset by lower
class-5 recall (-13.51). Thus its OA gain is not a class-collapse artifact,
although class-balanced performance is essentially unchanged.

Training took 149.76 seconds and inference for the 20-checkpoint trajectory
took 53.26 seconds on GPU 2. Frozen artifacts are under
`results/bida_self_dropout/diagnostic_2100/`.

- Frozen manifest SHA256:
  `acb8c8798ac84912c64ab4dc9cd938ded8c5f48950e77c46ebdffcf6cd678dc4`
- Epoch-200 checkpoint SHA256:
  `d05347bb0a9cfd12099b403be0029f9903b9317ee22ade53630855f143f53c89`
- Epoch-200 prediction SHA256:
  `fb86b64962d7c4e26e062d41e0e40d6e34beca9aaf3d28d5b484da6b34967526`

This is one diagnostic seed. Target labels were used only after predictions
and hashes for every checkpoint had been frozen.

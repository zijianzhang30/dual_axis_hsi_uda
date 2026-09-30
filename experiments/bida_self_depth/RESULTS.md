# BiDA-self depth decomposition: seed 2100

Both arms use mixed source/target BN, dropout 0.1, source CE, and fixed
epoch-200 target self inference. The depth-1 model retains the exactly
initialized first block and all shared components from the depth-3 model.

| Depth | Parameters | Target OA | AA | Kappa | Epoch-200 source-val OA |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 276,983 | 77.85% | 63.69% | 0.616 | 66.93% |
| 3 | 376,567 | **78.41%** | **66.32%** | **0.627** | 79.53% |

Depth 3 improves OA by only **0.56 points**, below the predeclared two-point
material-effect threshold. AA improves by 2.64 points. Both variants retain
nonzero recall for all seven target classes and similar prediction
distributions; depth 1 does not collapse.

The deeper refinement mainly helps minority-class coverage in this seed:
class-1 recall rises from 25.06% to 40.58%, class-3 from 55.74% to 57.81%,
and class-7 from 17.52% to 22.84%, while class-2 is 2.40 points lower. The OA
effect remains small because class 6 dominates Houston18.

This result does not support three transformer blocks as the main explanation
of BiDA-self's +4.83 OA advantage over A1. One block combined with BiDA's
stem/tokenizer and mixed BN already reaches 77.85%. It does show that reporting
AA and per-class recall is necessary: depth has a larger class-balanced effect
than its OA difference suggests.

Training took 131 seconds and target inference 2.24 seconds on GPU 4. Frozen
artifacts are under `results/bida_self_depth/diagnostic_2100/`.

- Frozen manifest SHA256:
  `75172900c49d066d04880413331633fa00eb8de7620109b85d36fdff70390c2d`
- Final checkpoint SHA256:
  `83b0feb5b1c617b6b5048abb67d6deefb107be7ce666c3fd3387e7417b472adf`
- Prediction SHA256:
  `481cc802bf87aa20e047b5bac3853b3dd5ea0ae2aa96d7462e5182e45df614b6`

This is one diagnostic seed. Target labels were used only after predictions
and hashes were frozen.

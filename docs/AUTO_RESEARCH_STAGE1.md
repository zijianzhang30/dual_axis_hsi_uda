# Auto Research Stage 1

## 1. 当前实验的统一结论

### 1.1 统一协议与结果口径

- 数据为 Houston13 → Houston18，`normband` 归一化，13×13 patch，Houston13 固定的分层 95/5 source train/val 划分（`sample_gt` 内部固定 `random_state=23`），batch size 128，SGD、学习率 0.01、无 momentum、无学习率调度，训练 200 epoch。
- 正式结果以固定 epoch 200 的 target self branch 为准。Houston18 标签值不进入 loss 或 checkpoint 选择，但发布代码使用 Houston18 GT mask 决定哪些像素是 target 训练候选中心，因此这是与 Strict BiDA 一致的 transductive-mask protocol，不能表述为训练完全不访问 target GT。
- Houston18 极不平衡：七类像素数为 `[1353, 4878, 2761, 22, 5323, 32264, 6300]`，第 6 类占 60.99%。任何接近 60.99% OA、AA 约 14.29%、只预测第 6 类的结果均是多数类 collapse。

### 1.2 Backbone 结论

| 实验 | epoch-200 target OA，mean ± SD | AA，mean ± SD | 结论 |
| --- | ---: | ---: | --- |
| A0：stem + pooling | 59.64 ± 5.58 | 38.12 | 基础 stem 的迁移能力不足 |
| A1：4 spatial queries + class query | 70.38 ± 1.82 | 55.56 | 三个 seed 均超过 A0，是当前最可靠 backbone |
| A1 配对重跑 | 72.10 ± 1.39 | 60.73 ± 1.34 | 当前应锁定的 A1 参考值；405,255 参数 |
| A2：双轴 token + 双向融合 | 68.53 ± 3.31 | 58.75 | 没有稳定超过 A1 |
| A2-lite：spatial/spectral token 仅由 class query 汇合 | 68.45 ± 7.70 | 59.09 ± 0.43 | 相对配对 A1 为 −3.65 ± 6.52 pp；spectral branch 增加不稳定性 |

A1 的四个 query 是有效的空间聚合器，但不是已经证实的四种语义角色。未经约束时，target token cosine 和 attention-map cosine 多在 0.94–0.98；强行加入 diversity 后 attention cosine 降至约 0.05，但 query 主要按固定边缘位置分工，target border attention mass 从约 0.21–0.28 升到约 0.60。D1 相对 D0 的 OA 为 −1.10 ± 2.01 pp。因此“固定 slot 身份等于固定跨域语义”不成立，不能据此做 same-slot interaction。

### 1.3 已测试 UDA 机制

| 机制 | 主要结果 | 机制诊断 | 判定 |
| --- | --- | --- | --- |
| Source semantic memory + target distillation（A3） | OA 30.63 ± 33.63 | 所有 seed 的最大 epoch-mean target KL < 4.5×10⁻⁶；memory/self 仅改变 0.01%–0.025% target 预测 | teacher 信号近乎为零且训练不稳定，暂停 |
| Global MMD（B1） | OA 38.21 ± 25.12 | seed 2101 source-val 约 14%，单类 collapse | 破坏 source discriminability，暂停 |
| Slot-wise MMD（B2） | OA 36.69 ± 23.99 | 没有任何 seed 超过 B1；同样出现 collapse | slot 对齐无优势，暂停 |
| Prediction consistency（B3） | OA 53.19 ± 14.07 | source-val 仍为 100%，但三个 seed 均低于 B0 | 保住 source 拟合但损害 target 迁移，不能单独依赖 |
| Slot MMD + consistency（B4） | OA 38.07 ± 25.22 | 未修复 MMD collapse | 暂停 |
| Slot diversity（D1） | OA 69.97 ± 3.19 | 成功制造空间不重叠，却退化为边缘位置分工 | 不支持 semantic-slot 假设，暂停后续 same-slot attention |
| Full source-set cross-attention（C1） | 已完成 seed 2100/2102 均为 OA 60.989、AA 14.286 | 两个 seed 全部 52,901 个像素预测为第 6 类；cross attention 接近均匀，KL 很小，self/cross disagreement 为零 | 无条件聚合全部 source tokens 导致多数类 collapse，停止第三个 seed |

统一结论是：A1 的 self representation 目前最稳；已有失败不是“交互强度不够”的共同证据。MMD 会直接损伤判别性，memory 和 set attention 则分别表现为没有有效信号和无选择的信息平均。下一步应保留 target self path，并先验证可靠性、选择性和类先验问题，再决定是否蒸馏。

### 1.4 “72% 对 80%+”目前不是严格成立的稳定差距

- Strict BiDA 固定 epoch 200 的三个 seed OA 为 78.75、64.51、77.24，均值 73.50%；配对 A1 为 73.58、70.82、71.90，均值 72.10%。均值差约 1.40 pp，且 Strict BiDA 在 seed 2101 低于 A1 6.31 pp。固定轮次下没有稳定的 8–10 pp 优势。
- Strict BiDA 用 Houston13 source-val OA 每 10 epoch 选点时为 82.06、76.29、78.29，均值 78.88%，但 source validation 仅 127 个像素，选中 epoch 分别为 40、20、80，选择噪声较大。
- 发布训练器直接用 Houston18 target OA 每 10 epoch 选最佳模型；本地得到过 82.16%，论文结果约 81.11%。这是 target-oracle 结果，不能与 A1 固定 epoch 200 作正式比较。
- A1 自身的 target-oracle 最大值仅比同一次重跑的 epoch 200 高 1.73 pp（72.80 对 71.07）。checkpoint 口径能解释一部分差异，但不能单独解释 BiDA 的最好单次结果。

## 2. A1 与 Strict BiDA 的关键差异

| 维度 | A1 | Strict BiDA | 对 gap 的当前判断 |
| --- | --- | --- | --- |
| 参数量 | 405,255 | 376,567 | A1 参数更多，原始参数容量不是首要解释 |
| Stem | 三个 factorized 3D conv；显式保留并下采样 spectral axis 到 12；GN + GELU | 一个 3×3×3 Conv3D 后把 8×48 spectral/channel 展平，再接 3×3 Conv2D；BN + ReLU | 表征偏置不同；BiDA 更早折叠 spectral axis，A1 的显式 spectral cube 并未自动带来更高迁移精度 |
| Spatial tokenization | 169 个 spatial features 由 4 个 learnable queries 经 cross-attention 读取 | 1×1 conv 产生 4 张空间 softmax attention map并加权汇聚 | 两者都形成 4 个空间 token，但 token 生成方式不同；现有结果只证明 A1 query 聚合有效 |
| Token refinement | 一个 query reader，随后一个 class-query reader；D=128 | 4 semantic tokens + CLS，learned position；3 个 triple-branch Transformer blocks；D=64 | BiDA 参数更少但 token refinement 更深；这是“结构深度/归纳偏置”，不能归入简单容量 |
| Normalization | Stem 用 GroupNorm，token 用 LayerNorm | Stem 用 BatchNorm3d/2d，token 用 LayerNorm | BiDA 训练时共享 BN 依次看到 source 与 target，target forward 会更新 running statistics；A1 的 GN 没有这条隐式 target-conditioned adaptation |
| Dropout | attention 默认无 dropout | token、attention、MLP 均有 0.1 dropout | BiDA 正则更强，可能影响 seed 稳定性与跨域泛化 |
| Optimizer | SGD，lr 0.01，无 momentum/schedule | 相同 | 不是当前差异 |
| Source supervision | A1/A0/A2 为单一 source CE；A3 才平均两项 source CE | source self CE | 基础监督强度大体可比 |
| Target usage | A1 基准虽配对读取 target batch，但 loss 与 forward 均不使用 target；实际是 source-only 学习 | 每一步使用 target self、coupled target、EMA target；epoch >100 还在 source/target logits 上使用 MMD | 这是两者最明确的 UDA 信息量差异 |
| Cross-domain branch | A1 无；已试 memory/set fusion 均失败 | 每层由随机配对的 source/target token 产生双向 coupled logits | BiDA 的直接交互未必合理，但其辅助损失组合可能提供有效训练信号 |
| Objective | `CE(source)` | `CE(source) + 0.1×双向 coupled→self distillation + 1.0×source/target EMA consistency`；epoch >100 再加 `0.1×MMD` | BiDA 同时改变多项因素，当前结果不能把收益归因于 coupled attention |
| 实现细节 | KL teacher 显式 detach | `distill_loss` 的 teacher 没有 detach；EMA consistency 还对已经 softmax 的概率再次 softmax；MMD 第二项实际是 `MMD(source_coupled, source_coupled)`，恒为零 | Strict BiDA 是需要原样比较的代码系统，其机制解释不能直接照搬公式描述 |
| Inference | target-only A1 self | target-only BiDA self | 两者部署均不依赖 source |
| Checkpoint | 固定 epoch 200 为 primary | 原发布代码用 target OA 选 best；本地 strict 另有 source-val 选点和 epoch 200 | 必须分别报告，不能混合成一个“BiDA 82%”基线 |

差异应分为三类：

1. **网络容量：** 参数量证据不支持“BiDA 更大所以更好”。需要考察的是三层 token refinement、BN 和 dropout 等结构差异。
2. **训练策略：** 优化器相同，但 BiDA 有 EMA、dropout、多分支损失和不同 checkpoint 轨迹；这些可能改变泛化与 seed 方差。
3. **UDA 信息利用：** A1 基准没有 target 梯度，BiDA 每一步都有 target 自分支、随机 coupled 分支和后期 alignment。固定 epoch 下的均值差很小，因此应先分解这些因素，不能直接重新设计更强交互。

## 3. 最值得验证的三个下一步 hypothesis

### H1：BiDA 的部分优势来自 target-conditioned BatchNorm，而不是 coupled attention

**Hypothesis**

BiDA 的共享 BN 在每一步先后处理 source 和 target，running statistics 因此含有无标签 target 分布信息。A1 使用 GN，不存在这条隐式适配路径。BN 统计可能解释 BiDA 即使随机 coupled interaction 不可靠，target self branch 仍有时显著优于 A1。

**Why previous methods failed**

过去的 MMD、memory 和 full-set attention都在表示或预测上施加强制跨域信号，但没有先测量一个更低风险的分布适配来源。它们分别造成判别性损伤、无效 teacher 或多数类平均。

**Exact mechanism changed**

不训练新模型。对一个预先固定的 Strict BiDA epoch-200 checkpoint，分别用 source-train patches 和 unlabeled target patches 重新估计 BN running mean/variance，得到 `source-BN` 与 `target-BN` 两个评估副本，并保留原始 mixed-BN 副本。权重、推理 self branch和所有其他模块不变；BN 统计使用完整预注册 loader 一次估计，不读取标签值。

**Held fixed**

同一 checkpoint、相同权重、同一 target self inference、同一数据归一化与评估像素。target GT 仅在三个预测文件冻结后统一计算 post-hoc 指标。

**Expected observable diagnostic**

除 OA/AA/Kappa 外，记录 BN 统计偏移、每类 recall、prediction distribution、entropy 及三个副本的预测 disagreement。若 target-BN 明显改变且改善 minority recall，说明 normalization 是低风险 target information channel；若三者几乎一致，则排除该解释。

**Go / No-Go**

一个 diagnostic seed。若 target-BN 相对 source-BN 提升 OA ≥2 pp，同时 AA 不下降、至少 5/7 类 recall 非零且没有单类占比增加超过 10 pp，则进入预注册的三 seed normalization 实验；否则停止该路线。该阈值在查看新 target 指标前固定。

### H2：BiDA 与 A1 的剩余差异来自 self representation/regularization，而不是跨域 coupled branch

**Hypothesis**

BiDA 的三层 token refinement、dropout 和 CNN tokenization形成比 A1 单层 reader 更适合跨场景的 self representation；其随机 coupled branch 可能不是主要贡献。由于 A1 参数量已经更大，关键变量是表示路径和训练正则，而非参数数目。

**Why previous methods failed**

此前直接在 A1 上添加 UDA loss，同时改变了 target 梯度和跨域机制，却没有先得到 BiDA encoder 在无 UDA 情况下的对照。因此无法判断应该改 representation，还是改 adaptation。

**Exact mechanism changed**

只运行一个 `BiDA-self` diagnostic arm：保留 BiDA 原始 stem、4 tokens、CLS、3 blocks、BN、dropout、参数初始化和 source CE；每步仍前向 source 与 unlabeled target以保持 BN 更新顺序，但关闭 coupled→self distillation、EMA consistency 和 MMD。target logits不进 loss。与同 seed 的现有 Strict BiDA 和 A1 epoch-200 结果比较。

**Held fixed**

数据、source split、batch 顺序、200 epoch、SGD 0.01、checkpoint 固定 epoch 200、target-only self inference不变；不加入新结构或新超参数。

**Expected observable diagnostic**

记录 source-val OA/loss、target OA/AA/Kappa、per-class recall、prediction distribution、entropy、参数量和训练时间。`BiDA-self − A1` 估计表示/正则差异，`Strict BiDA − BiDA-self` 估计原始 UDA objective 的增量；两者必须按同 seed 解读，不用 target 指标选点。

**Go / No-Go**

先跑一个 diagnostic seed。若 BiDA-self 已比 A1 高 ≥2 pp 且 AA 不降，下一阶段优先研究 self representation 的最小移植；若 Strict BiDA 比 BiDA-self 高 ≥2 pp 且无 collapse，则优先分解辅助训练信号；若两者均未达到，视为该 seed 不支持清晰 gap，先扩展差异诊断而不设计大模型。只有出现上述明确分离才扩到三 seed。

### H3：类平衡、可拒绝的 source-supervised retrieval 可以形成可靠 auxiliary teacher

**Hypothesis**

有效跨域信息应由 target 主动检索少量兼容 source semantics，并在证据不足时拒绝指导。候选集合按 source label 类平衡，compatibility 模块只用 source episodic supervision训练，因此不会把 Houston18 的 60.99% 多数类先验直接注入 teacher，也不依赖四个 query slot 的固定身份。

**Why previous methods failed**

Source memory 把每类压成固定 slot prototype且 teacher 信号趋近零；full source-set attention对 512 个 tokens近乎均匀平均并 collapse；same-slot 方法依赖未成立的 slot 语义；MMD 则无条件移动整个分布。共同缺少的是“兼容性验证、稀疏选择、类平衡和拒绝机制”。

**Exact mechanism changed**

第一步只做 retrieval diagnostic，不蒸馏、不改 A1：冻结一个 A1 encoder，将每个样本的四个 spatial tokens视为无序集合。用 labeled source 构造 class-balanced episodic query/support；训练一个共享、置换不变的 set compatibility head 区分同类与异类。检索时每类提供相同数量候选，每个 target 只保留 top-k 关系及一个由 source-validation compatibility margin预注册的 abstention rule。先冻结 target retrieval 输出，再用 target GT 作 post-hoc 正确性诊断。

**Held fixed**

A1 encoder、classifier、数据与 checkpoint均不变；不把 source feature加到 target feature，不训练 target，不使用 target pseudo-label，不使用 target GT 设定 k、margin 或结构。首轮没有 target distillation loss。

**Expected observable diagnostic**

首先检查 source leave-one-out retrieval：top-k class precision、各类 recall、compatibility calibration。target 侧记录 retrieval entropy、top-1 类分布、各类被检索频率、abstention coverage、与 A1 self prediction 的 agreement；target GT 仅在输出冻结后用于检查高 compatibility 是否真的对应更高预测正确率。可靠 retrieval 应显著非均匀，同时保持类覆盖，且可靠性随 compatibility margin单调上升。

**Go / No-Go**

一个 diagnostic seed，且不训练 UDA 模型。只有同时满足以下预注册条件才进入下一轮单 seed auxiliary-logit teacher：source-val top-1 retrieval accuracy ≥90%；target 非 abstain coverage ≥30%；至少 5/7 类被检索；最高可靠组的 post-hoc target correctness 比全部样本高 ≥10 pp。任一条件失败即停止，不通过调 target threshold挽救。后续 teacher只能提供 detached auxiliary logits并蒸馏到 target self，不能覆盖 target feature。

## 4. 每个 hypothesis 的最小 causal experiment

| Hypothesis | 最小实验 | 新训练 | 单一因果变量 | 必存诊断 | 最大运行规模 |
| --- | --- | --- | --- | --- | --- |
| H1 target-conditioned BN | 同一 Strict BiDA checkpoint 的 original/source-recalibrated/target-recalibrated BN 三副本 | 否 | BN running statistics 的来源 | OA、AA、Kappa、per-class recall、prediction distribution、entropy、BN shift、分支 disagreement | 1 seed；通过 gate 后才 3 seeds |
| H2 self representation vs UDA objective | A1、`BiDA-self`、现有 Strict BiDA 的同 seed epoch-200 分解 | 仅训练 1 个 BiDA-self arm | 关闭 BiDA 的 distillation、EMA consistency、MMD | 完整分类指标、source-val curve、entropy、训练成本；记录参数量 | 1 seed；出现 ≥2 pp 清晰分离后才 3 seeds |
| H3 selective retrieval | 冻结 A1，只训练 source episodic set-compatibility head并冻结 target retrieval 文件 | 仅 source relation head | 从无条件 source 聚合改为类平衡、top-k、可拒绝检索 | source retrieval precision/recall/calibration；target coverage、类分布、entropy、margin-correctness 曲线 | 1 seed diagnostic；通过四项 gate 才允许一次 UDA seed |

三项实验都必须在运行前保存配置、随机种子、模型/数据哈希和 Go/No-Go 条件；target prediction 文件先冻结，再计算 post-hoc target metrics。任何实验都不以 target 最佳 epoch作为正式结果。

## 5. 推荐优先级

1. **H1：BN 统计重估。** 成本最低，不训练即可验证一个此前未隔离、且确实向 BiDA target self branch传递 target 分布信息的机制。它先回答 gap 是否部分来自 normalization，而不是误把收益归给 coupled attention。
2. **H2：BiDA-self 分解。** 这是设计新模型前必须补齐的因果对照。它把网络表示/正则与 UDA objective分开，也能判断固定 epoch 下仅约 1.4 pp 的均值差是否值得追逐。
3. **H3：类平衡 selective retrieval。** 这是当前最值得保留的方法候选，但先只验证 teacher 信息是否可靠。只有 source-supervised compatibility在 target 上表现出选择性、覆盖和可靠性，才允许加入 auxiliary distillation。

在 H1/H2 完成前不修改 A1 backbone；在 H3 diagnostic 通过前不训练新的 target distillation 模型。当前没有证据支持恢复 spectral query、semantic memory、MMD、prediction consistency、slot diversity或 full source-set fusion。

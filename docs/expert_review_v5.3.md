# MangaTrace OPSD v5.3：实验背景、现象与待讨论问题

## 1. 这份材料希望解决什么

我们在漫画页面多模态结构化理解任务上研究 on-policy self-distillation（OPSD）。当前最需要解释的现象是：模型在训练早期已经获得小幅但较一致的固定集收益，训练损失和局部更新也都正常；继续训练后，学生自主生成的轨迹却逐步恶化，表现为非法命令增多、有效命令减少、图完成率和监督覆盖下降。

我们希望您结合代码、训练日志和评测结果，判断最可能的首要因果瓶颈，并给出一个可以通过短程实验明确证伪或支持的修改方案。这里更关心“为什么代理目标持续改善而完整行为退化”，而不只是如何通过早停选出当前最好的 checkpoint。

当前最需要区分的是：

1. 普通 token-level OPSD 是否在教师未必优于学生的位置持续施加了有偏更新；
2. 多个特权教师分量的概率混合是否破坏了多峰合法候选的几何结构；
3. 学生进入非法历史后，文本历史与执行器状态的错位是否形成了自我强化的训练分布；
4. failure anchor 是否只是加速了问题，而不是问题的来源；
5. 缺少局部信赖域或保持机制，是否使共享 LoRA 参数逐渐破坏原有能力。

训练成本较高，因此不希望进行大规模超参数网格搜索。更理想的结果是先确定一个最可能的机制，再设计一个有明确判别标准的短实验。

## 2. 任务简介

输入是一张完整漫画页面图像。Qwen3-VL-4B 需要自回归生成闭合标签 DSL，用一串命令建立页面场景图。主要命令包括：

- `enter`：按顺序进入漫画分格；
- `detect`：检测角色和文字框；
- `read`：读取已检测文字框中的文本；
- `link_from/link_to`：建立说话人关系或跨分格角色身份关系；
- `ground`：生成分格描述，并将描述中的实体或文本跨度关联到已检测对象。

后续命令会引用先前注册的对象，因此这是一个强状态依赖的长序列任务。早期坐标、对象注册、命令结构或引用错误会改变后续可执行状态。模型不仅需要保持 DSL 合法，还要同时完成检测、OCR、说话人关联、角色身份聚类和 caption grounding。

固定集评测包括：

- command valid rate 和 panel recall；
- detection precision、recall 和 F1；
- OCR 1-EditDist 和 exact accuracy；
- speaker association F1；
- character identification AMI 和 pairwise F1；
- caption grounding entity/span F1。

训练执行器的“非法命令率”依赖当时的执行状态，口径比最终评测中的命令合法率更严格，两者不能直接相减或互相替代。

## 3. 为什么采用 OPSD

学生和冻结教师均从同一个 Qwen3-VL-4B DSL SFT checkpoint 初始化。标准 SFT 要求模型拟合一条给定序列，但在当前任务的许多状态中，可能同时存在多个合法且有价值的后续操作。OPSD 允许教师在学生实际访问的前缀上给出软分布，并通过特权信息使教师在多数位置上比学生更有判断力。

我们的任务与标准 OPSD 存在两个显著差异：

1. 特权信息不是单一答案，而是从当前 GT 场景图状态构造的多个可执行候选分量；
2. 一个局部命令错误可能改变之后很长一段轨迹的可执行语义，学生会快速进入冻结教师训练时很少遇到的历史。

当前实现没有把教师强制投影到完全合法的候选集合，也没有用 hard label 代替教师分布。每个分量保留其原始词表概率，包括教师自身的错误概率。这样做是为了保留 OPSD 的基本前提：教师只需总体上优于学生，而不必在每个 token 上百分之百正确。

## 4. 当前多教师监督方式

在每个因果预测位置，代码执行以下过程：

1. 根据当前执行状态生成多个局部特权教师分量；
2. 只使用已经出现的学生前缀保留仍兼容的分量，不使用正在预测的 token 或未来 token 事后选择教师；
3. 对仍兼容的分量进行等权概率混合；
4. 从各分量 top-k 的并集选择稀疏支持，再补查每个分量在保留 token 上缺失的精确概率；
5. 将稀疏支持之外的概率作为 tail event，并对学生执行 forward KL。

这种混合存在一个需要重点检查的风险：如果不同教师分别偏好不同的合法候选，而共同给同一个非法候选较高概率，算术平均会稀释各个合法 mode，却不会稀释共同错误。当前因果前缀路由能够逐步排除不兼容分量，但在命令早期仍可能保留多个分量，因此该问题不能仅靠后续路由消除。

## 5. v5.3 的轨迹处理与训练配置

仓库当前主实验的关键设置来自 `opsd_qwenvl_fsdp.sh`：

- 学生收集轨迹时使用 greedy decoding，`MANGA_STUDENT_GENERATION_TEMPERATURE=0`；
- actor 和教师用于 KL 的原始分布保持 temperature = 1；
- batch size = 16，每批数据采样一次并更新一次；
- learning rate = `1e-6`；
- LoRA rank = 64，alpha = 128；
- LoRA 参数和 Adam states 使用 FP32，主体模型使用 BF16；
- gradient clipping = 20；
- progress loss weight = 0；
- 不反复训练同一失败页面；
- 不使用 frozen-SFT preservation loss；
- 不使用 legality CE、probability transport 或 PPO/GRPO policy loss；
- 普通目标是全局 token-mean forward KL。

学生按命令生成原始 token，生成内容始终保留在文本历史中。边界检测基于解码后的文本，而不是检查单独的换行 token，因此能够处理 `>\n` 被编码成一个 token 的情况。

当一条命令第一次不再兼容任何候选时：

1. 第一个失败 token 仍接受教师监督；
2. 同一失败命令剩余后缀被屏蔽；
3. 如果能识别到可靠的命令结束边界，执行器状态保持不变，并在完整原始学生历史上继续生成下一条命令；
4. 如果不能可靠恢复边界，则停止该轨迹。

主实验还保留每条失败命令的 first-failure anchor：

```text
L = 0.8 * L_all_supervised_tokens + 0.2 * L_first_failure_anchors
```

其中 anchor loss 先在每条轨迹内平均，再跨轨迹平均，因此权重 `0.2` 不代表它只贡献总梯度的 20%。我们也进行了 `failure_anchor_weight=0` 的对照。

## 6. 仓库版本与阅读入口

本文基于 Git 提交 **`36506ee`**（`v5.3`）以及该提交中已经整理的日志和结果。建议优先阅读：

- `opsd_qwenvl_fsdp.sh`：主训练入口和实际超参数；
- `verl/experimental/agent_loop/manga_soft_opsd_agent_loop.py`：greedy 学生轨迹、失败后缀屏蔽和命令边界恢复；
- `verl/trainer/manga/raw_mixture.py`：因果分量路由、稀疏支持选择和概率混合；
- `verl/trainer/distillation/losses.py`：base OPSD 与 failure anchor 的损失组合；
- `verl/trainer/distillation/fsdp/losses.py`：稀疏 forward KL；
- `verl/workers/engine/fsdp/transformer_impl.py`：稀疏投影和 actor 更新；
- `verl/workers/engine_workers.py`：FP32 LoRA 和 optimizer state 路径；
- `verl/workers/optimizer_probe.py`：实际参数更新探针；
- `evaluation/infer_vllm.py`、`evaluation/evaluate.py`、`evaluation/metrics.py`：固定集推理与评测。

仓库中的 `data/demo.parquet` 有 512 条样例，可用于了解字段、prompt 和 GT 图结构。完整训练集和模型 checkpoint 没有随仓库提供。

## 7. step 16 固定集结果

当前 `results/` 中有四份汇总结果。每份结果均记录 `num_samples=100`，并使用同一 evaluator 指标结构。

| 指标 | 初始 SFT | v5.3 w=0.2 s16 | v5.3 w=0 s16 | 完整轨迹监督 s16 |
|---|---:|---:|---:|---:|
| Command valid rate | 0.980921 | 0.994400 | 0.993030 | **0.994789** |
| Panel recall | 0.884440 | **0.944495** | 0.941765 | 0.941765 |
| Detection F1 | 0.750383 | **0.795989** | 0.793202 | 0.769614 |
| OCR 1-EditDist | 0.643858 | **0.692679** | 0.684040 | 0.680550 |
| OCR accuracy | 0.410872 | **0.412769** | 0.405183 | 0.404551 |
| Speaker F1 | 0.530711 | **0.554609** | 0.547573 | 0.548563 |
| Character AMI | **0.390539** | 0.359753 | 0.376688 | 0.337096 |
| Character pairwise F1 | 0.395679 | 0.434719 | **0.440975** | 0.429209 |
| Caption entity F1 | 0.682411 | 0.712329 | **0.714094** | 0.708487 |
| Caption span F1 | 0.813962 | **0.862316** | 0.857241 | 0.859498 |

从这些结果可以得到三点较稳妥的结论：

1. 三种 OPSD 版本在 step 16 都已经不是“完全没有学到东西”。它们在合法率、panel recall、detection、OCR 1-EditDist、speaker 和 grounding 等多数指标上超过当前仓库中的 SFT 基线。
2. `w=0` 与 `w=0.2` 很接近，前者甚至在 character pairwise F1 和 caption entity F1 上更高。这说明 early gain 不依赖 failure anchor。
3. 完整轨迹监督与另外两组方向大体一致，但不能简单说成完全相同：它的 detection F1 比 `w=0.2` 低约 0.026，Character AMI 也更低。扩大监督覆盖没有带来更强的 step 16 收益。

当前结果文件只包含聚合 JSON，没有对应的逐样本预测 JSONL，也没有把 checkpoint、数据哈希、seed 和完整推理参数写入结果。`evaluation/infer_vllm.py` 的默认推理配置包含 temperature = 0 和 repetition penalty = 1.1，而训练入口的 rollout repetition penalty = 1.0。若实际评测命令与默认值不同，需要一并提供；否则外部分析无法完全复现实验口径。

## 8. 两个 v5.3 长程实验的共同退化

下面按每 16 个 optimizer steps 汇总两个长程实验。非法命令率按总非法命令数除以总命令数计算，有效命令/页按总有效命令数除以总页面数计算。

| 版本与步数 | 非法命令率 | 有效命令/页 | 完成率 | 提前 EOS | 监督 token 利用率 | Teacher-student overlap |
|---|---:|---:|---:|---:|---:|---:|
| w=0.2，1–16 | 0.6812 | 15.10 | 0.1758 | 0.7773 | 0.3756 | 0.6784 |
| w=0.2，17–32 | 0.6736 | 15.90 | 0.1289 | 0.8398 | 0.3871 | 0.6829 |
| w=0.2，81–96 | 0.8635 | 7.48 | 0.0391 | 0.8438 | 0.2698 | 0.5932 |
| w=0，1–16 | 0.6839 | 14.97 | 0.1602 | 0.7969 | 0.3749 | 0.6803 |
| w=0，17–32 | 0.6683 | 16.13 | 0.1367 | 0.8320 | 0.3935 | 0.6852 |
| w=0，81–96 | 0.8391 | 7.15 | 0.0391 | 0.9062 | 0.2709 | 0.6155 |

两个版本在前 32 步几乎重合，随后均明显恶化。到 step 81–96：

- 非法命令率从约 0.68 上升到 0.84–0.86；
- 有效命令/页从约 15–16 下降到约 7；
- 图完成率都下降到 0.039；
- 监督利用率下降到约 0.27。

去掉 failure anchor 只使部分区间的非法率稍低，并没有改变整体轨迹。这排除了“anchor 是后期退化的必要条件”。anchor 的归一化仍可能放大或加速局部更新，但不能单独解释主现象。

## 9. 完整轨迹监督对照能够说明什么

完整轨迹版本不再屏蔽非法命令的失效后缀，其日志目前只有 step 1–32：

| Steps | 非法命令率 | 有效命令/页 | 完成率 | 提前 EOS | 监督 token 利用率 | Teacher-student overlap |
|---|---:|---:|---:|---:|---:|---:|
| 1–16 | 0.6806 | 15.08 | 0.1641 | 0.7891 | 1.0000 | 0.7176 |
| 17–32 | 0.6763 | 15.79 | 0.1406 | 0.8320 | 1.0000 | 0.7237 |

监督覆盖从约 0.38 提高到 1.0，但 step 16 固定集收益并没有同步扩大。这说明“被屏蔽 token 太多”不是解释早期学习效率的充分原因。

该版本还记录了教师质量审计。按每步已有比例取平均：

| 审计指标 | step 1–16 | step 17–32 |
|---|---:|---:|
| Mixture top-1 legal rate | 0.7006 | 0.5672 |
| Best legal candidate probability | 0.6628 | 0.5367 |
| Best illegal candidate probability | 0.2584 | 0.3849 |
| Legal-vs-illegal top-1 margin | 0.4044 | 0.1519 |
| Suffix component top-1 agreement | 0.9937 | 0.9937 |
| Teacher assigns lower probability to sampled suffix token | 0.4145 | 0.3795 |

这些指标提示，在越来越困难或异常的状态上，教师的合法优势会明显变弱；而在失效后缀中，各分量又几乎总是给出相同 top-1。全量监督因此可能主要重复学生已经高置信度生成的后缀，而没有提供足够强的纠错方向。

这里必须保留一个证据边界：现有完整轨迹日志只到 step 32，在这 32 步内还没有出现两个 v5.3 实验在 step 48 之后的明显崩塌。因此，当前仓库不能独立支持“完整轨迹版本也已经发生相同后期退化”这一结论。若已有更长日志，应补充后再判断。

此外，提交 `36506ee` 保留的是 v5.3 的失败后缀屏蔽实现。仓库提供了完整轨迹版本的日志和 s16 结果，但没有单独保留生成该实验的代码 patch，因此该对照目前不能从仓库直接复现。

## 10. 当前最可能的机制

我们目前更倾向于把首要瓶颈放在 **base OPSD 缺少逐位置的教师优势条件** 上，而不是继续围绕 failure anchor 调整权重。

其逻辑是：

1. 训练早期，学生主要访问与 SFT 分布接近的状态，特权教师通常更有信息，因此 soft KL 能带来真实收益；
2. base OPSD 对所有可路由位置持续优化，并没有先确认该位置的教师确实优于学生；
3. 学生逐渐访问包含非法文本历史、但执行器状态未前进的异常组合，教师合法优势随之下降；
4. 多教师算术平均可能进一步稀释不同合法 mode，同时保留共同错误；
5. 即使单个位置的更新很小，持续、同向且没有 reference trust region 的共享 LoRA 更新也会逐步改变原本正确的状态；
6. 更差的策略又产生更多异常 on-policy 状态，最终形成负反馈。

这个机制能够同时解释：step 16 出现真实收益；anchor 权重设为 0 后仍然退化；扩大到全轨迹监督没有自动带来更好结果；token KL 下降却不能保证完整图质量。

这仍是工作假设，而不是已经证明的结论。最关键的是判断应当如何定义“教师在当前位置确实更好”，以及如何在不退化为 hard-label SFT 的前提下，只让这部分信号进入学生更新。

## 11. 希望重点帮助判断的问题

### A. 教师优势应如何定义和门控

是否应只在教师相对学生具有明确优势时执行 KL，例如依据可执行候选总概率、合法 top-1 margin、教师与学生的相对熵，或冻结 SFT 在该状态上的稳定性？这种门控如何避免把教师重新强制成完全正确的 oracle？

### B. 多教师概率应如何组合

当前 equal raw mixture 是否适合多峰合法动作？mixture-of-losses、几何平均、按教师优势加权或保留 mode identity 的目标，哪一种更符合 OPSD 的原始动机，并且不会使用学生未来 token 事后选教师？

### C. 非法历史应该如何进入教师上下文

失败命令保留在学生历史、执行器状态不更新、下一条命令恢复监督，这种状态表示是否会污染教师分布？如果需要隔离失败状态，应修改教师上下文、学生上下文还是损失 mask，才能避免重新回到只训练轨迹前部的问题？

### D. 如何限制无关位置上的破坏

我们不希望简单用较大的 frozen-SFT KL 把模型拉回初始状态。是否可以只在教师没有明确优势的位置施加 reference trust region，或将 adapter 更新投影到不会显著损伤固定参考状态的方向？

### E. token-level forward KL 是否缺少序列级约束

当前 loss 按 token 平均，而最终指标来自完整场景图。是否需要按命令、执行状态或页面重新归一化，或者加入一种只负责衡量执行进展、但不直接替代教师分布的序列级信号？

## 12. 希望得到的分析形式

为了减少昂贵训练的试错，希望最终建议包含：

1. 从代码和日志中指出一个最可能的首要因果瓶颈；
2. 解释它如何同时产生“step 16 小幅收益”和“继续训练后轨迹退化”；
3. 给出一个首选修改，精确到 loss、mask、教师聚合或状态处理的定义；
4. 指出本次实验中应保持不变的设置，以形成单变量比较；
5. 给出最短验证步数、固定集 checkpoint 和提前停止指标；
6. 说明正结果和负结果分别会支持或排除什么机制；
7. 如果现有证据仍不足，优先给出可在已有 checkpoint 和日志上完成的离线分析。

## 13. 仓库内随附材料

### 训练代码

- Git revision：`36506ee`
- `opsd_qwenvl_fsdp.sh`
- `verl/experimental/agent_loop/manga_soft_opsd_agent_loop.py`
- `verl/trainer/manga/raw_mixture.py`
- `verl/trainer/distillation/losses.py`
- `verl/trainer/distillation/fsdp/losses.py`
- `verl/workers/engine/fsdp/transformer_impl.py`
- `verl/workers/engine_workers.py`
- `verl/workers/optimizer_probe.py`

### 训练日志

- `logs/mangatrace_a_opsd_v5.3-failure_anchor_weight0.2.metrics.jsonl`：step 1–123；
- `logs/mangatrace_a_opsd_v5.3-failure_anchor_weight0.0.metrics.jsonl`：step 1–96；
- `logs/mangatrace_a_opsd_v5.4-full_rollout.metrics.jsonl`：完整轨迹监督，step 1–32。

### 固定集结果

- `results/qwen3vl_4b_dsl_sft_v3.1_e1.json`；
- `results/qwen3vl_4b_dsl_opsd_v5.3_w0.2_s16.json`；
- `results/qwen3vl_4b_dsl_opsd_v5.3_w0.0_s16.json`；
- `results/qwen3vl_4b_dsl_opsd_v5.4_s16.json`。

### 数据与评测

- `data/demo.parquet`：512 条字段和格式示例；
- `evaluation/infer_vllm.py`：推理入口；
- `evaluation/evaluate.py`：评测入口；
- `evaluation/metrics.py`：指标实现；
- `evaluation/dsl.py`：DSL 解析与执行。

为了完整复核，仍建议补充实际固定集推理命令、评测数据标识或哈希、四个结果对应的逐样本预测 JSONL，以及完整轨迹实验的代码 patch 和更长训练日志。

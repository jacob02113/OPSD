# MangaTrace OPSD v5.3：实验背景、现象与待讨论问题

## 1. 这份材料希望解决什么

我们在漫画页面多模态结构化理解任务上研究 on-policy self-distillation（OPSD）。当前实验同时记录到以下现象：部分固定集指标在训练早期高于初始 SFT；主实验继续训练后，多项固定集指标下降；学生自主生成轨迹中的非法命令率上升，有效命令数、图完成率和监督覆盖下降。

我们希望您结合代码、训练日志和评测结果，分析训练目标、教师分布、学生状态分布与固定集指标之间的关系，并给出可以通过短程实验验证的修改方案。讨论重点是解释训练日志与完整任务评测之间的变化是否一致，以及出现差异的环节。

当前最需要区分的是：

1. 普通 token-level OPSD 覆盖了哪些位置，以及这些位置上的教师与学生分布差异；
2. 多个特权教师分量经过概率混合后，合法与非法候选的排序如何变化；
3. 非法命令之后，文本历史发生变化而执行器状态保持不变时，后续教师分布和学生分布如何变化；
4. `failure_anchor_weight=0.2` 与 `failure_anchor_weight=0` 的训练记录和固定集结果有何差异；
5. `manga_sft_preservation_weight=0` 时，初始 SFT 分布在训练过程中的变化。

训练成本较高，因此希望每项修改都对应一个明确问题，并预先确定短程检查点和判别指标。

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

学生和冻结教师均从同一个 Qwen3-VL-4B DSL SFT checkpoint 初始化。标准 SFT 要求模型拟合一条给定序列，但在当前任务的许多状态中，可能同时存在多个合法后续操作。当前 OPSD 实现让教师在学生实际访问的前缀上给出软分布，并通过 GT 场景图构造特权信息。

我们的任务与标准 OPSD 存在两个显著差异：

1. 特权信息不是单一答案，而是从当前 GT 场景图状态构造的多个可执行候选分量；
2. 一个局部命令错误可能改变之后很长一段轨迹的可执行语义，学生会快速进入冻结教师训练时很少遇到的历史。

当前实现没有把教师投影到完全合法的候选集合，也没有用 hard label 代替教师分布。每个分量保留其原始词表概率，包括教师在非法候选上的概率。

## 4. 当前多教师监督方式

在每个因果预测位置，代码执行以下过程：

1. 根据当前执行状态生成多个局部特权教师分量；
2. 只使用已经出现的学生前缀保留仍兼容的分量，不使用正在预测的 token 或未来 token 事后选择教师；
3. 对仍兼容的分量进行等权概率混合；
4. 从各分量 top-k 的并集选择稀疏支持，再补查每个分量在保留 token 上缺失的精确概率；
5. 将稀疏支持之外的概率作为 tail event，并对学生执行 forward KL。

算术平均在以下情况下会改变候选排序：若三个教师分别给三个不同合法候选 0.6 的概率，并都给同一个非法候选 0.4 的概率，混合后每个合法候选的概率为 0.2，而该非法候选仍为 0.4。当前因果前缀路由会随学生前缀增长排除不兼容分量；在命令早期，仍可能同时保留多个分量。

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

本文基于 Git 提交 **`36506ee`**（`v5.3`）及当前仓库中的日志和结果。相关实现位于：

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

## 7. 固定集结果

当前 `results/` 中有八份汇总结果。每份结果均记录 `num_samples=100` 和 `missing_predictions=892`，并使用同一指标结构。

### v5.3 failure anchor weight = 0.2

| 指标 | 初始 SFT | s16 | s32 | s48 | s64 |
|---|---:|---:|---:|---:|---:|
| Command valid rate | 0.980921 | 0.994400 | 0.995329 | 0.994310 | 0.995523 |
| Panel recall | 0.884440 | 0.944495 | 0.944495 | 0.945405 | 0.927207 |
| Detection F1 | 0.750383 | 0.795989 | 0.797409 | 0.759190 | 0.688201 |
| OCR 1-EditDist | 0.643858 | 0.692679 | 0.692045 | 0.686973 | 0.669210 |
| OCR accuracy | 0.410872 | 0.412769 | 0.412769 | 0.410872 | 0.400759 |
| Speaker F1 | 0.530711 | 0.554609 | 0.549553 | 0.545810 | 0.534874 |
| Character AMI | 0.390539 | 0.359753 | 0.352216 | 0.346072 | 0.368145 |
| Character pairwise F1 | 0.395679 | 0.434719 | 0.424529 | 0.425241 | 0.422130 |
| Caption entity F1 | 0.682411 | 0.712329 | 0.689012 | 0.653943 | 0.616660 |
| Caption span F1 | 0.813962 | 0.862316 | 0.827854 | 0.787633 | 0.736315 |

在该序列中，command valid rate 在 s16–s64 之间为 0.994310–0.995523。Detection F1 在 s32 为 0.797409，在 s48 为 0.759190，在 s64 为 0.688201。Caption entity F1 从 s16 的 0.712329 依次变为 0.689012、0.653943 和 0.616660；caption span F1 从 0.862316 依次变为 0.827854、0.787633 和 0.736315。

### failure anchor 与完整轨迹对照

| 指标 | v5.3 w=0.2 s16 | v5.3 w=0 s16 | 完整轨迹 s16 | 完整轨迹 s32 |
|---|---:|---:|---:|---:|
| Command valid rate | 0.994400 | 0.993030 | 0.994789 | 0.972094 |
| Panel recall | 0.944495 | 0.941765 | 0.941765 | 0.933576 |
| Detection F1 | 0.795989 | 0.793202 | 0.769614 | 0.763881 |
| OCR 1-EditDist | 0.692679 | 0.684040 | 0.680550 | 0.684210 |
| OCR accuracy | 0.412769 | 0.405183 | 0.404551 | 0.406448 |
| Speaker F1 | 0.554609 | 0.547573 | 0.548563 | 0.552560 |
| Character AMI | 0.359753 | 0.376688 | 0.337096 | 0.405020 |
| Character pairwise F1 | 0.434719 | 0.440975 | 0.429209 | 0.441542 |
| Caption entity F1 | 0.712329 | 0.714094 | 0.708487 | 0.681960 |
| Caption span F1 | 0.862316 | 0.857241 | 0.859498 | 0.822317 |

`w=0` 与 `w=0.2` 的 s16 数值接近。完整轨迹版本从 s16 到 s32 时，command valid rate 从 0.994789 变为 0.972094，detection F1 从 0.769614 变为 0.763881，caption entity F1 从 0.708487 变为 0.681960，caption span F1 从 0.859498 变为 0.822317；同期 OCR、speaker、Character AMI 和 pairwise F1 的变化方向不同。

这些结果文件只包含聚合 JSON，没有对应的逐样本预测 JSONL，也没有记录 checkpoint 路径、评测数据哈希、seed 和完整推理命令。`evaluation/infer_vllm.py` 的默认值为 temperature = 0、repetition penalty = 1.1；训练入口的 rollout repetition penalty = 1.0。结果文件本身不能确认实际评测时是否使用了这些默认值。

## 8. 两个 v5.3 长程实验的轨迹记录

下面按每 16 个 optimizer steps 汇总两个长程实验。非法命令率按总非法命令数除以总命令数计算，有效命令/页按总有效命令数除以总页面数计算。

| 版本与步数 | 非法命令率 | 有效命令/页 | 完成率 | 提前 EOS | 监督 token 利用率 | Teacher-student overlap |
|---|---:|---:|---:|---:|---:|---:|
| w=0.2，1–16 | 0.6812 | 15.10 | 0.1758 | 0.7773 | 0.3756 | 0.6784 |
| w=0.2，17–32 | 0.6736 | 15.90 | 0.1289 | 0.8398 | 0.3871 | 0.6829 |
| w=0.2，81–96 | 0.8635 | 7.48 | 0.0391 | 0.8438 | 0.2698 | 0.5932 |
| w=0，1–16 | 0.6839 | 14.97 | 0.1602 | 0.7969 | 0.3749 | 0.6803 |
| w=0，17–32 | 0.6683 | 16.13 | 0.1367 | 0.8320 | 0.3935 | 0.6852 |
| w=0，81–96 | 0.8391 | 7.15 | 0.0391 | 0.9062 | 0.2709 | 0.6155 |

两个版本在前 32 步的各项统计接近。到 step 81–96：

- 非法命令率从约 0.68 上升到 0.84–0.86；
- 有效命令/页从约 15–16 下降到约 7；
- 图完成率都下降到 0.039；
- 监督利用率下降到约 0.27。

在 `failure_anchor_weight=0` 的记录中，同样出现了非法命令率上升、有效命令/页下降、完成率下降和监督利用率下降。

## 9. 完整轨迹监督对照的现有记录

完整轨迹版本不再屏蔽非法命令的失效后缀，其日志目前只有 step 1–32：

| Steps | 非法命令率 | 有效命令/页 | 完成率 | 提前 EOS | 监督 token 利用率 | Teacher-student overlap |
|---|---:|---:|---:|---:|---:|---:|
| 1–16 | 0.6806 | 15.08 | 0.1641 | 0.7891 | 1.0000 | 0.7176 |
| 17–32 | 0.6763 | 15.79 | 0.1406 | 0.8320 | 1.0000 | 0.7237 |

该版本的监督 token 利用率为 1.0；两个 v5.3 版本在 step 1–16 的监督 token 利用率分别为 0.3756 和 0.3749。对应的 step 16 固定集结果见第 7 节。

该版本还记录了教师质量审计。按每步已有比例取平均：

| 审计指标 | step 1–16 | step 17–32 |
|---|---:|---:|
| Mixture top-1 legal rate | 0.7006 | 0.5672 |
| Best legal candidate probability | 0.6628 | 0.5367 |
| Best illegal candidate probability | 0.2584 | 0.3849 |
| Legal-vs-illegal top-1 margin | 0.4044 | 0.1519 |
| Suffix component top-1 agreement | 0.9937 | 0.9937 |
| Teacher assigns lower probability to sampled suffix token | 0.4145 | 0.3795 |

从 step 1–16 到 step 17–32，mixture top-1 legal rate 从 0.7006 变为 0.5672，best legal candidate probability 从 0.6628 变为 0.5367，best illegal candidate probability 从 0.2584 变为 0.3849，legal-vs-illegal top-1 margin 从 0.4044 变为 0.1519。Suffix component top-1 agreement 在两个区间均约为 0.9937。

现有完整轨迹日志和固定集结果均只到 step 32，不能与两个 v5.3 实验的 step 48–96 区间作同长度比较。

此外，提交 `36506ee` 保留的是 v5.3 的失败后缀屏蔽实现。仓库提供了完整轨迹版本的日志以及 s16、s32 结果，但没有单独保留生成该实验的代码 patch，因此不能仅使用当前代码复现该对照。

## 10. 希望重点帮助判断的问题

### A. 未获教师优势的位置是否需要保持约束

当前 `manga_sft_preservation_weight=0`，没有 frozen-reference preservation。希望判断是否需要在教师与学生差异较小或教师优势不明确的位置加入约束，以及应如何定义这些位置。

### B. token-level forward KL 与完整图指标的训练单位

当前 loss 按 token 平均，固定集指标按完整场景图计算。希望判断是否需要按命令、执行状态或页面重新归一化，以及是否需要单独的序列级信号。

## 11. 希望得到的分析形式

为了减少昂贵训练的试错，希望最终建议包含：

1. 区分现有材料能够确认的事实、由这些事实支持的推断以及仍缺少证据的问题；
2. 分析固定集指标、on-policy 轨迹指标和训练 loss 之间的关系；
3. 给出一个优先验证的修改，精确到 loss、mask、教师聚合或状态处理的定义；
4. 指出本次实验中应保持不变的设置，以形成单变量比较；
5. 给出最短验证步数、固定集 checkpoint 和提前停止指标；
6. 说明正结果和负结果分别会支持或排除什么机制。

## 12. 仓库内随附材料

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
- `results/qwen3vl_4b_dsl_opsd_v5.3_w0.2_s32.json`；
- `results/qwen3vl_4b_dsl_opsd_v5.3_w0.2_s48.json`；
- `results/qwen3vl_4b_dsl_opsd_v5.3_w0.2_s64.json`；
- `results/qwen3vl_4b_dsl_opsd_v5.3_w0.0_s16.json`；
- `results/qwen3vl_4b_dsl_opsd_v5.4_s16.json`；
- `results/qwen3vl_4b_dsl_opsd_v5.4_s32.json`。

### 数据与评测

- `data/demo.parquet`：512 条字段和格式示例；
- `evaluation/infer_vllm.py`：推理入口；
- `evaluation/evaluate.py`：评测入口；
- `evaluation/metrics.py`：指标实现；
- `evaluation/dsl.py`：DSL 解析与执行。

当前仓库未包含实际固定集推理命令、评测数据标识或哈希、各结果对应的逐样本预测 JSONL、完整轨迹实验的代码 patch，以及 step 32 之后的完整轨迹训练日志。

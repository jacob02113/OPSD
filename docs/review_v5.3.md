# MangaTrace OPSD v5.3–v5.5：实验背景、现象与待讨论问题

## 1. 这份材料希望解决什么

我们在漫画页面多模态结构化理解任务上研究 on-policy self-distillation（OPSD）。v5.3 实验记录到以下现象：部分固定集指标在训练早期高于初始 SFT；继续训练后，多项固定集指标下降；学生自主生成轨迹中的非法命令率上升，有效命令数、图完成率和监督覆盖下降。随后进行的 v5.5 实验把教师分布逐分量投影到当前状态的合法 next-token 集合，但该修改没有降低在线非法命令率，也没有阻止固定集检测指标继续退化。

我们希望您结合[代码](https://github.com/jacob02113/OPSD)、训练日志和评测结果，分析训练目标、教师分布、学生状态分布与固定集指标之间的关系，并给出可以通过短程实验验证的改进方案，使得模型能够在训练中稳定获得提升。

训练成本较高，因此希望每项修改都对应一个明确问题，并预先确定短程检查点和判别指标。

## 2. 任务背景

漫画页面由按阅读顺序排列的多个分镜组成，每个分镜中包含人物、对话文字及其交互。我们研究的任务是将这些视觉内容转化为可执行、可核验的结构化标注。输入是一张漫画页面，以及各分镜的位置和对应的文字描述；模型需要定位人物与文字区域、识别区域内的文字、判断每段对话由谁说出、关联不同分镜中属于同一角色的人物实例，并将给定描述中提及的人物对应到图像中的具体对象。例如，对于一个包含两名人物和两个对话框的分镜，模型不仅要找出四个区域并读出对话，还要确定对话与人物之间的归属关系，以及描述中的“戴帽子的男子”对应哪一个人物。分镜位置与描述由输入提供，模型负责生成其余视觉和关系标注。

这一任务的困难来自多个预测之间的依赖，以及跨分镜上下文的持续变化。文字识别需要先定位文字区域，说话人判断需要结合对话与人物，描述定位则要求相关人物已经被检测并能够被引用。同一角色跨分镜出现时，其姿态、尺度和可见外观可能变化，模型还需要将新出现的人物与此前的角色联系起来。因此，前面遗漏或错误识别一个对象，可能影响后续多个任务；即使各项局部预测形式正确，模型也可能漏掉尚未处理的文字或关系。另一方面，这些操作并不只有一种正确顺序：同一分镜中的多个人物可以先后检测，互不依赖的文字识别也可以交换顺序。模型既要满足必要的依赖约束，又要自主安排操作，完整覆盖需要标注的内容。

为统一表达上述任务，我们将漫画理解构造为有向无环图（DAG）上的命令执行过程。模型通过五类命令完成标注：`enter` 进入分镜，`detect` 定位人物或文字，`read` 识别已检测的文字，`link` 建立说话人或角色身份关系，`ground` 将给定描述中的人物提及对应到已检测对象。图中的节点表示具体操作，边表示前置依赖，例如“识别某个文字区域”必须在“检测该区域”之后执行。执行器维护已产生的标注及完成状态，模型每步选择一个当前可执行、尚未完成的操作，并在完成当前分镜内能够解决的任务后进入下一分镜。这种表达将各项视觉任务组织为共享状态的连续过程，在保留多种合法执行顺序的同时，使操作依赖、对象引用和任务遗漏能够被明确检查。

这一任务强调的能力，与围绕最终答案展开的推理后训练有所不同。在答案导向的推理任务中，成功通常体现为找到一条能够导向正确结论的推理路径；在我们的任务中，每个必要节点的产出都是最终结果的一部分，走到最后一个分镜并不能弥补此前遗漏的对象、文字或关系。模型需要在满足依赖约束的同时完整覆盖任务图，而同一状态往往存在多个同样有效的下一步。这使在线自蒸馏（OPSD）的分布监督具有直接的适用价值：学生可以学习当前状态下不同合法续写的相对概率，而不必只模仿参考序列中的单一选择。我们由此关注 OPSD 的一个具体用途——将教师对多个可执行候选的判断转化为学生可学习的分布，并随任务状态的变化持续更新这一监督。

## 3. 当前多教师监督方式

在每个因果预测位置，代码执行以下过程：

1. 根据当前执行状态生成多个局部特权教师分量；
2. 只使用已经出现的学生前缀保留仍兼容的分量，不使用正在预测的 token 或未来 token 事后选择教师；
3. 对仍兼容的分量进行等权概率混合；
4. 从各分量 top-k 的并集选择稀疏支持，再补查每个分量在保留 token 上缺失的精确概率；
5. 将稀疏支持之外的概率作为 tail event，并对学生执行 forward KL。

当前因果前缀路由会随学生前缀增长排除不兼容分量；在命令早期，仍可能同时保留多个分量。

## 4. 轨迹处理和训练配置

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

其中 anchor loss 先在每条轨迹内平均，再跨轨迹平均，因此权重 `0.2` 不代表它只贡献总梯度的 20%。

v5.5 保持上述 v5.3 `w=0.2` 设置不变，唯一方法改动是逐分量教师合法性投影。对学生因果前缀 `h` 的活跃教师分量 `j`，以 `V_j(h)` 表示当前执行器状态下仍可延伸为可执行命令的 next-token 集，先计算：

```text
Z_j(h) = sum_{t in V_j(h)} q_j(t|h)
q_j^legal(t|h) = q_j(t|h) * 1[t in V_j(h)] / Z_j(h)
q_mix^legal(t|h) = (1 / |A(h)|) * sum_j q_j^legal(t|h)
```

合法集合只由当前学生前缀和执行器状态决定，不读取正在预测的 token 或未来 token。各分量先在自身合法域归一化，再保持等权混合；后续仍使用 top-k 并集、缺失概率补查、tail event 和 token-mean forward KL。v5.5 同时记录教师原始合法质量、低合法质量行、零支持分量和 failure-anchor 更新前后的概率变化。该实验运行到 step 55，并在 step 16、32、48 使用同一 100 页固定集评测。

## 5. 仓库版本与阅读入口

相关实现位于：

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

## 6. 固定集结果

本文汇总的固定集结果均记录 `num_samples=100`，并使用同一指标结构。

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

### v5.5 逐分量合法教师投影

下表沿用前述 100 页固定集与 `repetition_penalty=1.0` 的初始 SFT 基线。三个 v5.5 JSONL 各包含 100 个无重复样例，并与 v5.3 s16 JSONL 的 sample ID 集合完全相同。

| 指标 | 初始 SFT | v5.5 s16 | v5.5 s32 | v5.5 s48 |
|---|---:|---:|---:|---:|
| Command valid rate | 0.980921 | 0.993530 | 0.977440 | 0.993137 |
| Panel recall | 0.884440 | 0.929026 | 0.929936 | 0.919017 |
| Detection F1 | 0.750383 | 0.756480 | 0.751187 | 0.705721 |
| OCR 1-EditDist | 0.643858 | 0.671992 | 0.678274 | 0.673021 |
| OCR accuracy | 0.410872 | 0.400759 | 0.402023 | 0.400759 |
| Speaker F1 | 0.530711 | 0.542958 | 0.556903 | 0.543083 |
| Character AMI | 0.390539 | 0.355872 | 0.360981 | 0.366433 |
| Character pairwise F1 | 0.395679 | 0.418278 | 0.429611 | 0.419341 |
| Caption entity F1 | 0.682411 | 0.702630 | 0.709656 | 0.693694 |
| Caption span F1 | 0.813962 | 0.852708 | 0.852708 | 0.828319 |

v5.5 s16 相对初始 SFT 提高了 command validity、panel recall、detection、OCR 1-EditDist、speaker、character pairwise 和两项 caption 指标，但没有复现 v5.3 s16 的固定集增益。与相同步数的 v5.3 相比，v5.5 s16 的 Detection F1、Character detection F1、OCR 1-EditDist、Caption entity F1 和 Caption span F1 分别低 0.039509、0.057299、0.020687、0.009699 和 0.009609。

v5.5 s32 的 caption entity、caption span、speaker 和 character pairwise F1 分别比 v5.3 s32 高 0.020644、0.024853、0.007350 和 0.005082，但 Detection F1 和 Character detection F1 分别低 0.046222 和 0.073112；command valid rate 只有 0.977440，对应 212 条无效命令，而 v5.3 s32 为 0.995329 和 42 条无效命令。到 s48，v5.5 的 caption entity/span F1 比 v5.3 s48 高 0.039751/0.040685，但 Detection F1 和 Character detection F1 低 0.053469/0.079529。相对改善主要表现为 v5.5 的 caption 指标退化较慢，而不是超过初始 SFT。

v5.5 后期的 detection 下降主要来自 character 假阳性增加。Character precision 从 s16 的 0.658962 降至 s48 的 0.537203，recall 仅从 0.788820 变为 0.778656；同期 character FP 从 723 增至 1188，而 TP 从 1397 变为 1379。因此，s48 的主要错误不是漏掉更多角色，而是生成了更多错误角色框。

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

## 7. v5.3 与 v5.5 的训练轨迹记录

### 两个 v5.3 长程实验

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

### v5.5 合法教师投影实验

v5.5 日志共有 55 个连续 optimizer steps。下表仍按总计数汇总非法命令率、有效命令/页、完成率、提前 EOS 和监督 token 利用率；step 49–55 只有 7 个 batch，因此单独列为不完整窗口。

| Steps | 非法命令率 | 有效命令/页 | 完成率 | 提前 EOS | 监督 token 利用率 | Sparse support overlap |
|---|---:|---:|---:|---:|---:|---:|
| 1–16 | 0.6800 | 15.10 | 0.1562 | 0.7969 | 0.3791 | 0.2953 |
| 17–32 | 0.6736 | 16.19 | 0.1445 | 0.8242 | 0.3912 | 0.2872 |
| 33–48 | 0.6834 | 13.88 | 0.1289 | 0.8516 | 0.3737 | 0.2954 |
| 49–55 | 0.7444 | 12.23 | 0.0536 | 0.9018 | 0.3381 | 0.2860 |

v5.5 在前 32 步没有改善 v5.3 的在线行为。v5.5 step 1–16 和 17–32 的非法命令率为 0.6800 和 0.6736，v5.3 `w=0.2` 对应窗口为 0.6812 和 0.6736；有效命令/页、完成率、提前 EOS 和监督利用率也处于相同量级。到 step 33–48，有效命令/页降至 13.88，完成率降至 0.1289，提前 EOS 升至 0.8516。最后 7 步的非法命令率进一步升至 0.7444，完成率降至 0.0536。由于合法投影改变了教师稀疏支持，表中的 sparse support overlap 不应与 v5.3 的旧 overlap 数值直接解释为同一难度下的独立质量增益。

投影审计显示实现没有因为空合法域而系统性丢失监督：

| Steps | 教师平均合法质量 | 低合法质量行比例 | 零支持分量 | 零支持行 | Raw KL | 失败 token 概率变化 | 保留合法质量变化 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1–16 | 0.9657 | 0.0309 | 3 | 0 | 0.7028 | -0.000141 | 0.000029 |
| 17–32 | 0.9616 | 0.0350 | 4 | 0 | 0.6804 | -0.000713 | 0.000539 |
| 33–48 | 0.9657 | 0.0312 | 1 | 0 | 0.5323 | -0.001448 | 0.001235 |
| 49–55 | 0.9594 | 0.0369 | 0 | 0 | 0.6549 | -0.002223 | 0.001409 |

教师平均合法质量稳定在 0.9594–0.9657，全部 55 步均没有零支持行。与此同时，failure anchor 对局部概率的作用越来越明显：失败 token 的平均概率变化绝对值持续增大，保留合法候选的质量变化从接近零转为正值。局部训练目标因此按设计生效，但完整轨迹的非法命令率、有效命令数、完成率和提前 EOS 没有同步改善。这一分叉说明，训练 loss 或 first-failure 局部概率改善不能单独作为序列级任务效果的证据。

v5.5 还没有通过在实验前设定的短程判别条件：

1. s16 Detection F1 比 v5.3 s16 低 0.039509，超过 0.02 的立即停止阈值；
2. step 17–32 非法命令率为 0.6736，没有达到 `<=0.62` 的目标；
3. s32 command valid rate 为 0.977440，低于 0.99；
4. s48 相对 v5.5 s16 的 Detection F1 和 Caption span F1 分别下降 0.050759 和 0.024389，均超过 step 48 所允许的 0.02。

因此，v5.5 结果不支持“教师在当前学生状态上的非法 token 质量是退化的充分原因”。合法投影减缓了相对 v5.3 的后期 caption grounding 下降，却没有降低在线非法命令率，也没有阻止 character 假阳性和 detection 退化。

## 8. 完整轨迹监督对照的现有记录

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

## 9. 希望得到的分析形式

为了减少昂贵训练的试错，希望最终建议包含：

1. 分析固定集指标、on-policy 轨迹指标和训练 loss 之间的关系；
2. 给出一个优先验证的修改，精确到 loss、mask、教师聚合或状态处理的定义；
3. 指出本次实验中应保持不变的设置，以形成单变量比较；
4. 给出最短验证步数、固定集 checkpoint 和提前停止指标；
5. 说明正结果和负结果分别会支持或排除什么机制。

## 10. 仓库内随附材料

### 训练代码

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
- `logs/mangatrace_a_opsd_v5.4-full_rollout.metrics.jsonl`：完整轨迹监督，step 1–32；
- `logs/mangatrace_a_opsd_v5.5.metrics.jsonl`：合法教师投影实验，step 1–55。

### 固定集结果

- `results/qwen3vl_4b_dsl_sft_v3.1_e1.json`；
- `results/qwen3vl_4b_dsl_opsd_v5.3_w0.2_s16.json`；
- `results/qwen3vl_4b_dsl_opsd_v5.3_w0.2_s32.json`；
- `results/qwen3vl_4b_dsl_opsd_v5.3_w0.2_s48.json`；
- `results/qwen3vl_4b_dsl_opsd_v5.3_w0.2_s64.json`；
- `results/qwen3vl_4b_dsl_opsd_v5.3_w0.0_s16.json`；
- `results/qwen3vl_4b_dsl_opsd_v5.4_s16.json`；
- `results/qwen3vl_4b_dsl_opsd_v5.4_s32.json`；
- `results/qwen3vl_4b_dsl_opsd_v5.5_s16.json` 与对应 JSONL；
- `results/qwen3vl_4b_dsl_opsd_v5.5_s32.json` 与对应 JSONL；
- `results/qwen3vl_4b_dsl_opsd_v5.5_s48.json` 与对应 JSONL。

### 数据与评测

- `data/demo.parquet`：512 条字段和格式示例；
- `evaluation/infer_vllm.py`：推理入口；
- `evaluation/evaluate.py`：评测入口；
- `evaluation/metrics.py`：指标实现；
- `evaluation/dsl.py`：DSL 解析与执行。

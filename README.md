# Qwen2.5-32B LoRA GRPO 训练记录

基于 GRPO（Group Relative Policy Optimization）和 LoRA 对 **Qwen2.5-32B-Instruct** 进行强化学习微调，任务场景为医学循证医学（EBM）Agent。

---

## 任务背景

训练一个能够回答临床问题的医学 Agent，要求输出符合 **PICO 结构**（Population/Intervention/Comparison/Outcome）并包含文献引用。

- **输入：** 临床问题（如"某治疗方案对某人群的效果是什么？"）
- **输出：** 包含 background/method/result/conclusion 各节、并引用检索到的证据的结构化回答
- **奖励函数：** PICO 格式分（关键词检测，最高 1.0）× 0.5 + Reward Model 打分（sigmoid 归一化）× 0.5
- **数据：** 900 条训练问题，来自医学文献检索场景

---

## 硬件与环境

| 项目 | 配置 |
|------|------|
| 服务器 | 117.50.171.247:23 |
| GPU | 4 × NVIDIA A800-SXM4-80GB（共 320GB 显存） |
| 内存 | 925GB |
| 存储 | 571GB（训练中稳定在 35% 使用率） |
| Python 环境 | `/root/lora_grpo` venv，Python 3.10 |
| 主要依赖 | torch 2.12.0+cu126、trl 1.4.0、peft、bitsandbytes、accelerate 1.13.0 |

---

## 训练框架选择：为什么用 TRL 而不是 VERL

最初计划使用 **VERL 0.7.1 + AgentLightning + vLLM HYBRID 模式**（vLLM TP worker 与 FSDP Actor 共享 GPU），经历约一天调试（共修复 13 个 bug）后发现：

> **vLLM 0.21.0 V1 引擎的 `vLLMColocateWorkerExtension` 在这台 A800 机器上存在底层 CUDA/NCCL 兼容性问题**，`update_weights` 调用会触发 NCCL AllGather 污染 vLLM TP worker 的 CUDA stream，导致后续任何推理崩溃。VERL 0.7.1 硬编码使用 V1 引擎，无法切换到 V0。

最终切换至 **TRL GRPOTrainer**：
- 使用 HuggingFace `generate()` 做 rollout，完全不依赖 vLLM
- DDP（`torchrun --nproc_per_node=4`），无需 FSDP/HYBRID
- QLoRA 4-bit 量化，每卡只占 ~22GB

详细调试过程见 [docs/environment_setup_journey.md](docs/environment_setup_journey.md)。

---

## 实验总览

| 实验 | 步数 | N_ROLLOUTS | lora_r | max_completion_len | LR | 后100步均值 | 最高 Reward | 时长 | evidence_ii | 状态 |
|------|------|------------|--------|-------------------|-----|------------|------------|------|-------------|------|
| [v1](runs/v1/) | 280 | 2 | 8 | 512 | 1e-4 | 0.707 | 0.779 | 8.4h | — | 完成（未收敛）|
| [v2](runs/v2/) | 448 | 4 | 8 | 512 | 1e-4 | 0.754 | 0.808 | 13h | — | 完成（未收敛）|
| [v3](runs/v3/) | 672 | 4 | 8 | 512 | 1e-4 | 0.761 | 0.801 | 18.6h | 64.1 | 完成（收敛）|
| [v4](runs/v4/) | 672 | 4 | **16** | 512 | 1e-4 | **0.786** | **0.824** | 20.7h | **65.5** | 完成（收敛）|
| [v5](runs/v5/) | 672 | 4 | 16 | **1024** | 1e-4 | 0.770 | 0.821 | 18.5h | 65.4 | 完成（收敛）|
| [v6](runs/v6/) | 672 | **8** | 16 | 512 | 1e-4 | **0.797（最高）** | **0.838** | 20.0h | **58.4（最低）** | 完成（奖励欺骗）|

> v4 唯一变更：`lora_r` 8→16，确认更大 rank 提升收敛平台（+0.025），但 evidence_ii 增益有限（+1.4），提示单纯扩容收益递减。  
> v5 唯一变更：`max_completion_length` 512→1024，证伪"生成长度是瓶颈"假说——v4 末期 mean_length=414 是奖励驱动的主动选择而非被截断，v5 放开后模型反而写得更短。  
> **v6 唯一变更：`N_ROLLOUTS` 4→8。发现奖励欺骗（Reward Hacking）：训练 reward 全系列最高（0.797），但 evidence_ii 全系列最低（58.4）。更强的优化器找到了奖励函数的漏洞——引用前置策略（82% 的回答将所有引用堆在结论，正文写"未检索到"），GPT-4.1 评审发现不一致性，证据回答一致性维度从 44.0 骤降至 14.5（-29.5）。**

---

## 四次实验核心对比

### 关键指标

| 指标 | v1（N_ROLLOUTS=2） | v2（N_ROLLOUTS=4） | v3（N_ROLLOUTS=4） | v4（lora_r=16） |
|------|-------------------|-------------------|--------------------|----------------|
| 步数 | 280 | 448 | 672 | 672 |
| lora_r | 8 | 8 | 8 | **16** |
| max_completion_len | 512 | 512 | 512 | 512 |
| 最终 reward | 0.713 | 0.748 | 0.738 | **0.788** |
| 最高 reward | 0.779 | 0.808 | 0.801 | **0.824**（step 631）|
| 后 50 步均值 | 0.707 | 0.753 | 0.762 | **0.786** |
| 后 100 步均值 | 0.707 | 0.754 | 0.761 | **0.786** |
| 后 100 步 std | — | — | 0.015 | **0.016（收敛）** |
| KL 散度（final） | 0.016 | 0.060 | 0.066 | **0.073** |
| 最终 loss | 0.00162 | 0.00681 | 0.00705 | ~0.008 |
| 每步耗时 | 104s | 104s | 99s | **125s** |
| 训练时长 | 8.4h | 13h | 18.6h | **20.7h** |
| evidence_ii 得分 | — | — | 64.1 | **65.5** |
| 是否收敛 | 否 | 否 | 是（step 448+）| **是（step 505+）** |

### Reward 分段进展对比（每 56 步）

| 步数段 | v1 | v2 | v3 |
|--------|----|----|-----|
| 0–55 | 0.685 | 0.695 | 0.695 |
| 56–111 | 0.706 | 0.747 | 0.721 |
| 112–167 | 0.710 | 0.745 | 0.721 |
| 168–223 | 0.706 | 0.758 | 0.754 |
| 224–279 | 0.707 | 0.754 | 0.740 |
| 280–335 | — | 0.769 | 0.729 ← KL spike 余震 |
| 336–391 | — | 0.757 | 0.752 |
| 392–447 | — | 0.753 | 0.761 |
| 448–503 | — | — | 0.765 |
| 504–559 | — | — | 0.759 |
| 560–615 | — | — | 0.761 |
| 616–671 | — | — | **0.761（收敛）** |

v2/v3 起点完全一致（0.695），step 392 后 v3 超过 v2 并进入稳定收敛区间。

---

## 深入分析

### 1. N_ROLLOUTS 的核心作用

GRPO 的 advantage 估计公式：

```
A_i = (r_i - mean(r_1...r_n)) / std(r_1...r_n)
```

- **N=2**：baseline 只有 2 个样本均值，std 很小，advantage 估计极粗糙；梯度步长微小，策略几乎不动
- **N=4**：4 个样本提供更稳定的 mean/std，advantage 区分度更高，策略更新更有方向性

直接体现：**KL 散度从 0.016 → 0.060**（+275%），说明 N=4 让策略真正开始偏离参考模型，而不是在原地踏步。

### 2. 为什么 Clip Ratio 始终为 0

两次实验 clip_ratio 均为 0，**这是 TRL 在线 GRPO 的预期行为，不是 bug**：

TRL 的 GRPOTrainer 每次 rollout 后只做 **1 个优化步**。clip 比率计算的是新旧策略的概率比，但由于只做 1 步更新，新旧策略差距极小，比率始终在 1.0 附近，无法超过 ε=0.2 的阈值。

触发 clip 需要：
- 多 epoch 优化（PPO 风格）——TRL 默认不支持
- 或者 N_ROLLOUTS ≥ 8，advantage 足够大才能推动更大的策略更新

在线 GRPO 中，**KL 散度是真正有效的正则化手段**，clip 只是辅助保护机制。

### 3. Loss 大幅上升是好事

v1 最终 loss 0.00162 → v2 最终 loss 0.00681（+320%）

原因：`loss = policy_gradient_loss + β × KL`

v1 中 KL ≈ 0.016，β=0.1 → KL 项贡献 0.0016（可忽略）  
v2 中 KL ≈ 0.068，β=0.1 → KL 项贡献 **0.0068**（主导 loss）

Loss 升高说明 KL 惩罚项在真正发挥作用，策略在有效地偏离参考模型同时受到约束——这是训练进入正轨的标志。

### 4. 生成长度的下降趋势

v2 中生成长度从训练初期的 ~350 tokens 降至末期的 ~284 tokens（下降 19%），同时 reward 在上升。

这说明模型在**主动学习用更简洁的语言拿到更高分**——奖励函数对 PICO 结构关键词的检测比对文字量更敏感，模型发现了这个模式。副作用：`transformers.generate` 时间从 ~110s 降至 ~60s，使得 N_ROLLOUTS=4 的每步总时间（104s）与 v1（104s）基本相同。

### 5. v2 未收敛的证据

后 200 步（步 248–447）的 reward 均值：

| 步数段 | 均值 | 趋势 |
|--------|------|------|
| 280–335 | 0.769 | 局部高点 |
| 336–391 | 0.757 | 回落 |
| 392–447 | 0.753 | 仍在震荡 |

曲线未拉平，仍以 ~+0.002/56步 的斜率缓慢上升，预计在 step 600–700 附近收敛至 ~0.78–0.80 平台。

---

## 收敛结论（v3 验证）

v3（672步）在 step 448 后明显收敛，后 224 步四个分段均值均在 0.759–0.765：

```
448–503: 0.765
504–559: 0.759
560–615: 0.761
616–671: 0.761（std=0.015，< 0.02 收敛阈值）
```

**v3 结论：该配置（N_ROLLOUTS=4，LR=1e-4，LoRA r=8）的 reward 上限约为 0.76。**

**v4 验证（lora_r=16）：** 收敛平台提升至 0.786（+0.025），evidence_ii 65.5（+1.4）。更大 rank 确有帮助，但边际收益已开始递减——v1→v3 training reward 涨 0.054，eval 涨 ~4 分；v3→v4 涨 0.025，eval 只涨 1.4 分。

**v4 观察到的新约束：** v4 末期 mean_length 达 414/512，生成长度接近 `max_completion_length` 上限，引用完整性和信息全面性受限可能与此直接相关。

**v5 方向（进行中）：** `max_completion_length` 512→1024，其余参数与 v4 完全相同，验证长度上限是否是当前瓶颈。

---

## 超参数设计说明

| 参数 | 值 | 选择理由 |
|------|-----|---------|
| LoRA r | 8 | 32B 模型 LoRA r=8 可训参数约 4600 万（0.14%），reward 上升未见瓶颈，无需加大 |
| LoRA alpha | 16 | alpha/r=2，有效 LR 放大 2×，标准设置 |
| N_ROLLOUTS | 4 | v1 用 2 导致 advantage 估计粗糙；4 是平衡效果与速度的合理选择（8 会使步时翻倍）|
| LR | 1e-4 | LoRA GRPO 常用范围 5e-5–2e-4；32B 模型取中间偏低值；v2 曲线平滑，无震荡，确认合适 |
| beta | 0.1 | 标准 KL 约束系数；v2 中 KL 维持在 0.06–0.08，处于健康范围 |
| epsilon | 0.2 | 标准 PPO clip 范围；TRL 在线 GRPO 中实际未触发，不影响训练 |
| 量化 | QLoRA NF4 4-bit | 32B 模型全精度需 64GB+/卡（OOM），4-bit 压缩至 ~22GB/卡，4 卡可行 |
| warmup_ratio | 0.03 | ~13 步 warmup，防止训练初期 LR 过大破坏预训练能力 |
| batch size | 4 GPU×1×grad_accum 4=16 | 显存受限下的合理选择 |

---

## 仓库结构

```
├── README.md                           # 本文档（中文，持续更新）
├── scripts/
│   ├── train_32b_lora_grpo_trl.py     # 主训练脚本（TRL GRPOTrainer）
│   └── run_32b_lora_grpo_trl.sh       # 启动脚本（env var 配置）
├── runs/
│   ├── v1/
│   │   ├── README.md                   # v1 运行总结
│   │   ├── hyperparameters.md          # 超参数详情
│   │   ├── training_log_analysis.md    # 逐步指标分析
│   │   └── metrics.csv                 # 280 步完整数据
│   ├── v2/
│   │   ├── README.md                   # v2 运行总结（含 v1 vs v2 对比）
│   │   ├── hyperparameters.md          # 参数变更记录
│   │   ├── training_log_analysis.md    # 逐步指标分析（448 步）
│   │   └── metrics.csv                 # 448 步完整数据
│   ├── v3/
│   │   ├── README.md                   # v3 运行总结（含三跑对比，收敛验证）
│   │   ├── hyperparameters.md          # 参数说明
│   │   ├── training_log_analysis.md    # 逐步指标分析（672 步，含 spike 分析）
│   │   └── metrics.csv                 # 672 步完整数据
│   └── v4/
│       ├── README.md                   # v4 运行总结（lora_r=16，evidence_ii 65.5）
│       ├── hyperparameters.md          # 与 v3 唯一变更：lora_r 8→16
│       ├── training_log_analysis.md    # 逐步指标分析（672 步）
│       └── metrics.csv                 # 672 步完整数据
└── docs/
    └── environment_setup_journey.md    # VERL→TRL 迁移全过程（13 个 bug 记录）
```

---

## 快速启动

```bash
ssh -p 23 root@117.50.171.247
cd /workspace/post_train/sql_agent
source /root/lora_grpo/bin/activate

# 查看当前训练进度（v5）
tail -f logs/v5_*.log | grep -E "reward|step|wandb"

# 启动 v5（max_completion_length=1024，其余同 v4）
LORA_R=16 LORA_ALPHA=32 MAX_STEPS=672 LR=1e-4 GRAD_ACCUM=4 N_ROLLOUTS=4 \
  MAX_RESPONSE_LEN=1024 OUTPUT_DIR=/workspace/outputs/32B-LoRA-GRPO-TRL-v5 \
  RUN_VERSION=v5 bash run_32b_lora_grpo_trl.sh
```

---

## WandB 监控

- **项目地址：** `http://103.139.212.228:3005/johnson/32b-lora-grpo-trl`
- v1 run：`qwen32b-lora-r8-trl`（未版本化，历史记录）
- v2 run：`qwen32b-lora-r8-rollout4-v2`（run ID: f9wu6by1）
- v3 run：`qwen32b-lora-r8-rollout4-v3`（run ID: 5fc1638k，已完成）
- v4 run：`qwen32b-lora-r16-rollout4-v4`（run ID: 4rusviq8，已完成）
- v5 run：`qwen32b-lora-r16-rollout4-v5`（训练中）

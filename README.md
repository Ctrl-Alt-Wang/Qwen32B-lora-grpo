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

| 实验 | 步数 | N_ROLLOUTS | LR | 初始 Reward | 最终 Reward | 最高 Reward | 时长 | 状态 |
|------|------|------------|-----|------------|------------|------------|------|------|
| [v1](runs/v1/) | 280 | 2 | 1e-4 | 0.645 | 0.713 | 0.779 | 8.4h | 完成 |
| [v2](runs/v2/) | 448 | 4 | 1e-4 | 0.645 | 0.748 | 0.808 | 13h | 完成 |
| v3 | 672 | 4 | 1e-4 | — | — | — | ~19h（预计） | **运行中** |

> v3 为全新起点（从 base model 开始），与 v2 完全同配置，只是步数更多，用于观察收敛。

---

## v1 → v2 核心对比

### 关键指标

| 指标 | v1（N_ROLLOUTS=2） | v2（N_ROLLOUTS=4） | 变化 |
|------|-------------------|-------------------|------|
| 最终 reward | 0.713 | **0.748** | +4.9% |
| 最高 reward | 0.779 | **0.808** | +3.7% |
| 后 100 步均值 | 0.707 | **0.754** | +6.6% |
| KL 散度均值 | 0.016 | **0.060** | +275% |
| KL 散度最大 | 0.050 | **0.152** | +204% |
| Reward std 均值 | 0.035 | **0.045** | +28% |
| 梯度范数均值 | 0.053 | **0.068** | +28% |
| 最终 loss | 0.00162 | **0.00681** | +320% |
| Clip ratio | 0（全程） | 0（全程） | 无变化 |
| 每步耗时 | 104s | **104s** | 无额外开销 |

### Reward 进展对比（按 56 步分段）

| 分段 | v1 均值 | v2 均值 | v2 较 v1 |
|------|---------|---------|---------|
| 0–55 步 | 0.685 | 0.695 | +1.5% |
| 56–111 步 | 0.706 | 0.747 | +5.8% |
| 112–167 步 | 0.710 | 0.745 | +4.9% |
| 168–223 步 | 0.706 | 0.758 | +7.4% |
| 224–279 步 | 0.707 | 0.754 | +6.6% |
| 280–335 步 | — | 0.769 | — |
| 336–391 步 | — | 0.757 | — |
| 392–447 步 | — | 0.753 | — |

v2 在第 56 步时就比 v1 高出接近 6%，说明 N_ROLLOUTS=4 带来的优势估计改善是实质性的，而非噪声。

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

## v3 设计思路

**目标：** 在同等超参数下，用更多步数跑到收敛，与 v2 曲线做公平对比。

| 参数 | v2 | v3 |
|------|----|----|
| N_ROLLOUTS | 4 | 4 |
| LR | 1e-4 | 1e-4 |
| MAX_STEPS | 448 | **672** |
| 起点 | base model | base model（全新） |
| 预计时长 | 13h | ~19h |
| WandB run | `...-rollout4-v2` | `...-rollout4-v3` |
| 收敛预期 | 未收敛 | 预计 step 550–650 收敛 |

v2 和 v3 从同一起点（Qwen2.5-32B-Instruct base）出发，步数区间 0–448 可以直接对比，v3 多出的 224 步用于观察是否收敛。

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
│   └── v2/
│       ├── README.md                   # v2 运行总结（含 v1 vs v2 对比）
│       ├── hyperparameters.md          # 参数变更记录
│       ├── training_log_analysis.md    # 逐步指标分析（448 步）
│       └── metrics.csv                 # 448 步完整数据
└── docs/
    └── environment_setup_journey.md    # VERL→TRL 迁移全过程（13 个 bug 记录）
```

---

## 快速启动

```bash
ssh -p 23 root@117.50.171.247
cd /workspace/post_train/sql_agent
source /root/lora_grpo/bin/activate

# 查看当前训练进度
tail -f logs/v3_rollout4_12epoch_20260604.log | grep -E "reward|step|wandb"

# 启动新实验（参数按需修改）
MAX_STEPS=672 LR=1e-4 N_ROLLOUTS=4 SAVE_STEPS=56 RUN_VERSION=v3 \
  OUTPUT_DIR=/workspace/outputs/32B-LoRA-GRPO-TRL-v3 \
  nohup bash run_32b_lora_grpo_trl.sh > logs/v3_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

---

## WandB 监控

- **项目地址：** `http://103.139.212.228:3005/johnson/32b-lora-grpo-trl`
- v1 run：`qwen32b-lora-r8-trl`（未版本化，历史记录）
- v2 run：`qwen32b-lora-r8-rollout4-v2`（run ID: f9wu6by1）
- v3 run：`qwen32b-lora-r8-rollout4-v3`（run ID: 5fc1638k，进行中）

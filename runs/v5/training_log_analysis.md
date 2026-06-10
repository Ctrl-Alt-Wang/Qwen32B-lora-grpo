# Run v5 — 训练日志分析

日志文件：`logs/v5_max1024_20260609_193952.log`（672 步）  
WandB：run ID `6skuwv6c`，项目 `32b-lora-grpo-trl`

---

## 启动命令

```bash
LORA_R=16 \
LORA_ALPHA=32 \
MAX_STEPS=672 \
LR=1e-4 \
GRAD_ACCUM=4 \
N_ROLLOUTS=4 \
MAX_RESPONSE_LEN=1024 \
OUTPUT_DIR=/workspace/outputs/32B-LoRA-GRPO-TRL-v5 \
RUN_VERSION=v5 \
bash run_32b_lora_grpo_trl.sh
```

---

## 训练过程时间线

| 时间 | 事件 |
|------|------|
| 2026-06-09 19:39 | `torchrun --nproc_per_node=4` 启动 |
| 2026-06-09 19:40 | Step 1 完成，reward=0.6438，step_time=77s |
| 2026-06-10 ~01:40 | Step ~168（Epoch 1 结束），reward 均值≈0.760 |
| 2026-06-10 ~07:40 | Step ~336（Epoch 2 结束），reward 均值≈0.775 |
| 2026-06-10 ~11:40 | Step 400，reward 均值≈0.776 |
| 2026-06-10 ~12:40 | Step 194，峰值 reward=0.8207 |
| 2026-06-10 06:08 | Step 672 完成，训练结束，reward=0.7471 |

总训练时长：**66,480 秒（约 18.5 小时）**（比 v4 少 2.2h，原因：平均生成长度更短）

---

## reward 演进

```
Step   1:  0.644  初始
Step  50:  0.707
Step 100:  0.776
Step 194:  0.821  ← 峰值（远早于 v4 的 step 631）
Step 200:  0.773
Step 300:  0.791
Step 400:  0.776
Step 500:  0.752
Step 600:  0.773
Step 671:  0.778
Step 672:  0.747  最终
────────────────────────
后 100 步均值: 0.770  std: 0.016
后 200 步均值: 0.768  std: 0.017
```

---

## 关键指标观察

### 1. 收敛平台低于 v4（0.770 vs 0.786）

这是本次实验最反直觉的发现。扩大 `max_completion_length` 512→1024 后，训练 reward 的收敛平台**下降了 0.016**，而非预期的持平或上升。

可能原因：
- 奖励函数（格式分 + RM 分）对 400-500 token 长度的回答"最优"，这是 v3/v4 训练强化出的隐式偏好
- 放开 1024 后，模型在早期尝试了长回答（step 2 max_length=585），但这些长回答拿到的奖励并不更高，因此策略反向收缩至更短的输出
- 最终模型学会了用 270-325 token 的简洁回答最大化奖励，反而偏离了 v4 找到的 400-450 token 平衡点

### 2. 峰值出现极早（step 194 vs v4 step 631）

v5 的最高 reward 0.8207 出现在 step 194（约 Epoch 1 结束），之后 reward 振荡下行直到在 0.770 附近收敛。这与 v4 的单调上升曲线完全不同，表明 v5 陷入了一个局部最优，之后的 KL 惩罚逐渐将策略拉回。

### 3. 生成长度先升后降（与 v4 完全相反）

| 阶段 | v4 mean_length | v5 mean_length |
|------|---------------|---------------|
| Step 1 | 259 | 248 |
| Step 100 | ~320 | 296 |
| Step 200 | ~345 | 327（局部高点）|
| Step 400 | ~370 | 273 |
| Step 672 | 414 | 307 |

v4 中 mean_length 单调增长（259→414），说明模型在奖励驱动下主动学习"写更多内容"。  
v5 中 mean_length 先涨后跌（248→327→273），说明模型尝试过更长的回答但发现奖励更低，最终退缩到更短的输出。

**这直接证明：v4 的 414 token 均值并非被 512 上限截断，而是模型在奖励驱动下主动学习的最优长度。**

### 4. KL 散度整体偏高

| 阶段 | v4 kl | v5 kl |
|------|--------|--------|
| step 50 | ~0.030 | **0.073** |
| step 100 | ~0.040 | **0.079** |
| step 300 | ~0.055 | **0.130** |
| step 672 | 0.073 | **0.094** |

v5 的 KL 值在整个训练过程中都高于 v4，说明 v5 的策略偏移更大，但这种偏移并未带来更高的奖励。高 KL + 低 reward 是策略在搜索空间中"迷路"的典型特征。

### 5. 步时减少（符合预期）

| 版本 | 平均步时 |
|------|---------|
| v4 | ~111s |
| v5 | ~99s |

v5 生成更短（270-325 vs 414），autoregressive 解码更快，每步节省约 12s，总时长减少 2.2h。

---

## checkpoint 情况

| Checkpoint | 路径 | 对应步数 |
|-----------|------|---------|
| checkpoint-670 | `/workspace/outputs/32B-LoRA-GRPO-TRL-v5/checkpoint-670` | step 670 |
| checkpoint-672（final） | `/workspace/outputs/32B-LoRA-GRPO-TRL-v5/` | step 672 |

评测使用 **final checkpoint（step 672）**。

---

## vllm 部署配置（评测用）

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m vllm.entrypoints.openai.api_server \
    --model /workspace/models/Qwen2.5-32B-Instruct \
    --enable-lora \
    --lora-modules grpo_v5=/workspace/outputs/32B-LoRA-GRPO-TRL-v5 \
    --port 8001 \
    --dtype bfloat16 \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.90 \
    --tensor-parallel-size 4 \
    --trust-remote-code \
    --enable-auto-tool-choice \
    --tool-call-parser hermes
```

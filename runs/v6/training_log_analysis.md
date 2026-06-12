# Run v6 — 训练日志分析

日志文件：`logs/trl_grpo_20260610_234832.log`（672 步）  
WandB：run ID `lrruv6iy`，项目 `32b-lora-grpo-trl`

---

## 启动命令

```bash
LORA_R=16 \
LORA_ALPHA=32 \
MAX_STEPS=672 \
LR=1e-4 \
GRAD_ACCUM=4 \
N_ROLLOUTS=8 \
MAX_RESPONSE_LEN=512 \
OUTPUT_DIR=/workspace/outputs/32B-LoRA-GRPO-TRL-v6 \
RUN_VERSION=v6 \
bash run_32b_lora_grpo_trl.sh
```

---

## 训练过程时间线

| 时间 | 事件 |
|------|------|
| 2026-06-10 23:48 | `torchrun --nproc_per_node=4` 启动 |
| 2026-06-10 23:49:40 | Step 1 完成，reward=0.6446，step_time=73s |
| 2026-06-11 ~06:00 | Step ~200（Epoch 1 结束附近），reward 均值≈0.80 |
| 2026-06-11 ~10:00 | Step 300 出现长度峰值：mean_length=489.6，reward=0.73 |
| 2026-06-11 ~16:00 | Step 554，峰值 reward=0.8375 |
| 2026-06-11 19:51 | Step 672 完成，训练结束，reward=0.7898 |

总训练时长：**71,880 秒（约 19.97 小时）**

---

## reward 演进

```
Step   1:  0.644  初始
Step  50:  0.725
Step 100:  0.722
Step 200:  0.799
Step 300:  0.732  ← 异常低谷（mean_length=489.6，尝试超长输出）
Step 400:  0.798
Step 500:  0.782
Step 554:  0.838  ← 峰值
Step 600:  0.786
Step 672:  0.790  最终
────────────────────────────
后 100 步均值: 0.7972  std: ~0.020
后 100 步最高: 0.8307
后 100 步最低: 0.7488
```

### 各阶段 100 步均值

| 步数区间 | reward 均值 | 趋势 |
|---------|------------|------|
| 1-100 | 0.7197 | 快速上升 |
| 101-200 | 0.7657 | 持续上升 |
| 201-300 | 0.7725 | 小幅上升，但 step 300 异常 |
| 301-400 | 0.7753 | 从异常中恢复 |
| 401-500 | 0.7925 | 明显跃升 |
| 501-600 | 0.7978 | 趋于稳定 |
| 601-672 | 0.7979 | 收敛 |

---

## 关键指标观察

### 1. Step 300 的异常长度峰值

| Step | reward | mean_length | kl | step_time |
|------|--------|-------------|-----|-----------|
| 200 | 0.799 | 385.0 | 0.073 | 117.6s |
| **300** | **0.732** | **489.6** | **0.097** | **128.1s** |
| 400 | 0.798 | 324.3 | 0.151 | 96.5s |
| 500 | 0.782 | 368.1 | 0.126 | 112.5s |

Step 300 处出现了整个训练过程中最长的生成（489.6 tokens），同时 reward 跌至 0.732——这是模型在探索"长输出是否能获得更高奖励"的过程。答案是否定的：step 400 模型收缩至 324.3 tokens，reward 回升至 0.798。这与 v5 的类似现象相呼应（v5 也尝试了更长输出后退缩），说明奖励函数对 ~350-400 token 的输出存在隐式偏好。

但 v6 的关键差异在于：退缩后，模型找到了一个不同的局部最优——**在结论中前置大量引用，而正文使用"未检索到"填充**。这在 v5 中未出现，因为 v5 只有 4 个 rollout，优化能力不足以收敛到这个更深的局部最优。

### 2. 收敛平台高于 v4（0.797 vs 0.786）

v6 后 100 步均值 0.7972，高于 v4 的 0.786，也高于 v5 的 0.770。这是整个 v1-v6 系列训练 reward **最高的版本**。

**但这是一个反直觉的危险信号**：训练 reward 更高，而评测得分更低（58.4 vs 65.5）。reward 的提升来自于更有效地利用了奖励函数的漏洞，而非真实临床质量的提升。

### 3. 生成长度趋势对比

| 阶段 | v4 mean_length | v6 mean_length |
|------|---------------|---------------|
| Step 1 | 259 | 248 |
| Step 100 | ~320 | 354 |
| Step 200 | ~345 | 385 |
| Step 300 | ~360 | **490（异常峰值）** |
| Step 400 | ~370 | 324（退缩） |
| Step 672 | 414 | 380 |

v4 呈单调增长，v6 先冲高再退缩，最终末期 380 tokens 略低于 v4 的 414。

### 4. KL 散度整体偏高

| 阶段 | v4 kl | v5 kl | **v6 kl** |
|------|--------|--------|----------|
| step 50 | ~0.030 | 0.073 | **0.037** |
| step 100 | ~0.040 | 0.079 | **0.129** |
| step 300 | ~0.055 | 0.130 | **0.097** |
| step 672 | 0.073 | 0.094 | **0.129** |

v6 的 KL 早期（step 100）大幅跳升至 0.129，这与 step 300 的长度峰值相关——模型早期尝试了较大的策略偏移。末期 KL=0.129 高于 v4（0.073）但低于 v5 最高时（0.130）。

### 5. 步时与 v4 基本一致

| 版本 | 平均步时 | 原因 |
|------|---------|------|
| v4（rollout=4）| ~111s | — |
| v5（rollout=4, max_len=1024）| ~99s | 生成更短 |
| **v6（rollout=8）** | **~107s** | 生成数量翻倍但长度减少 |

理论上 rollout=8 应该让每步生成时间大约翻倍，但 v6 的生成平均长度（~350-400 tokens）与 v4（~380-414 tokens）相近或更短，batch 化生成抵消了数量翻倍的额外开销，最终步时反而与 v4 相近。

---

## Checkpoint 情况

| Checkpoint | 路径 | 对应步数 |
|-----------|------|---------|
| checkpoint-670 | `/workspace/outputs/32B-LoRA-GRPO-TRL-v6/checkpoint-670` | step 670 |
| checkpoint-672（final） | `/workspace/outputs/32B-LoRA-GRPO-TRL-v6/` | step 672 |

评测使用 **final checkpoint（step 672）**。

---

## vllm 部署配置（评测用）

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m vllm.entrypoints.openai.api_server \
    --model /workspace/models/Qwen2.5-32B-Instruct \
    --enable-lora \
    --lora-modules grpo_v6=/workspace/outputs/32B-LoRA-GRPO-TRL-v6 \
    --port 8001 \
    --dtype bfloat16 \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.90 \
    --tensor-parallel-size 4 \
    --trust-remote-code \
    --enable-auto-tool-choice \
    --tool-call-parser hermes
```

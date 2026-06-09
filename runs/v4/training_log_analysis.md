# Run v4 — 训练日志分析

日志文件：`logs/v4_r16_rollout4_20260608.log`（1438 行）  
WandB：run ID `4rusviq8`，项目 `32b-lora-grpo-trl`

---

## 启动命令

```bash
LORA_R=16 \
LORA_ALPHA=32 \
MAX_STEPS=672 \
LR=1e-4 \
GRAD_ACCUM=4 \
N_ROLLOUTS=4 \
OUTPUT_DIR=/workspace/outputs/32B-LoRA-GRPO-TRL-v4 \
RUN_VERSION=v4 \
bash run_32b_lora_grpo_trl.sh
```

---

## 训练过程时间线

| 时间 | 事件 |
|------|------|
| 2026-06-08 08:07 | `torchrun --nproc_per_node=4` 启动 |
| 2026-06-08 08:09 | Step 1 完成，reward=0.6425，step_time=78s |
| 2026-06-08 ~14:00 | Step ~168（Epoch 1 结束），reward 均值≈0.725 |
| 2026-06-08 ~20:00 | Step ~336（Epoch 2 结束），reward 均值≈0.751 |
| 2026-06-09 ~00:30 | Step ~504（Epoch 3 半程），reward 均值≈0.778 |
| 2026-06-09 ~03:30 | Step 631，峰值 reward=0.8238 |
| 2026-06-09 04:51 | Step 672 完成，训练结束，reward=0.788 |

总训练时长：**74,642 秒（约 20.7 小时）**

---

## reward 演进

```
Step   1:  0.643  初始
Step  50:  ~0.712
Step 100:  ~0.730
Step 200:  ~0.745
Step 300:  ~0.758
Step 400:  ~0.770
Step 500:  ~0.779
Step 600:  ~0.785
Step 631:   0.824  ← 峰值
Step 672:   0.788  最终
────────────────────────
后 100 步均值: 0.786  std: 0.016
后 200 步均值: 0.786  std: 0.015
```

---

## 关键指标观察

### 1. 收敛平台更高（+0.025 vs v3）

v4 后 100 步均值 **0.786** vs v3 **0.761**。更大的 rank（16 vs 8）提供了更多可学习参数（7 个 target_modules，每个 rank 从 8→16，可训练参数约翻倍），使模型能更精细地拟合奖励信号。

### 2. 峰值出现在末期（step 631）

最高 reward 0.8238 出现在倒数第 41 步，表明模型在最后阶段仍有持续学习能力，未出现明显过拟合。v3 的峰值分布更均匀，v4 则呈现末期上冲特征。

### 3. 步时随训练轻微增长

| 阶段 | 典型 step_time |
|------|--------------|
| 初期（step 1） | 78s |
| 中期（step 100-400） | ~108s |
| 后期（step 600-672） | ~125s |

增长来自 KL 计算和更长的生成序列（mean_length 从初期 ~259 增至后期 ~414）。

### 4. KL 散度控制良好

| 阶段 | kl 值 |
|------|------|
| 初始（step 1） | 0.000 |
| step 100 | ~0.010 |
| step 400 | ~0.055 |
| step 672（final） | 0.073 |

beta=0.1 的 KL 惩罚有效约束策略偏移，未出现 KL 爆炸。

### 5. entropy 温和上升

从初始 0.33 升至最终 0.43，说明模型输出多样性略有提升，策略探索充分但未失控。

### 6. 生成长度增长

| 步数 | mean_length |
|------|------------|
| step 1 | 259 |
| step 200 | ~320 |
| step 400 | ~370 |
| step 672 | 414 |

回答越来越长，说明模型学会了用更详细的内容来争取更高奖励。

---

## checkpoint 情况

| Checkpoint | 路径 | 对应步数 |
|-----------|------|---------|
| checkpoint-616 | `/workspace/outputs/32B-LoRA-GRPO-TRL-v4/checkpoint-616` | step 616 |
| final adapter | `/workspace/outputs/32B-LoRA-GRPO-TRL-v4/adapter_model.safetensors` | step 672 |

评测使用 **final checkpoint（step 672）**，由 vllm `--lora-modules grpo_v4=<path>` 加载。

---

## vllm 部署配置（评测用）

```bash
# 启动脚本：/workspace/scripts/start_vllm_qwen32b_grpo_v4.sh
source /root/lora_grpo/bin/activate

CUDA_VISIBLE_DEVICES=0,1,2,3 python -m vllm.entrypoints.openai.api_server \
    --model /workspace/models/Qwen2.5-32B-Instruct \
    --enable-lora \
    --lora-modules grpo_v4=/workspace/outputs/32B-LoRA-GRPO-TRL-v4 \
    --port 8001 \
    --dtype bfloat16 \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.90 \
    --tensor-parallel-size 4 \
    --trust-remote-code \
    --enable-auto-tool-choice \
    --tool-call-parser hermes
```

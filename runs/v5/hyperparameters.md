# Run v5 — 超参数

## 与 v4 的唯一变更

| 参数 | v4 | **v5** | 说明 |
|------|----|--------|------|
| `max_completion_length` | 512 | **1024** | 允许模型生成更长的回答，探索长度上限效应 |

## 完整超参数表

| 参数 | 值 |
|------|----|
| 基础模型 | Qwen2.5-32B-Instruct |
| 量化方式 | QLoRA 4-bit（NF4，double quant） |
| `lora_r` | 16 |
| `lora_alpha` | 32 |
| `lora_dropout` | 0.05 |
| `target_modules` | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj |
| `max_steps` | 672 |
| `max_completion_length` | **1024** |
| `learning_rate` | 1e-4 |
| `lr_scheduler` | cosine |
| `warmup_ratio` | 0.03 |
| `per_device_train_batch_size` | 1 |
| `gradient_accumulation_steps` | 4 |
| `num_generations`（N_ROLLOUTS） | 4 |
| `beta`（KL 系数） | 0.1 |
| `epsilon`（clip ratio） | 0.2 |
| `loss_type` | grpo |
| `bf16` | true |
| `gradient_checkpointing` | true |
| WandB run name | `qwen32b-lora-r16-rollout4-v5` |
| WandB run ID | `6skuwv6c` |

## 训练环境

| 项目 | 配置 |
|------|------|
| 服务器 | 117.50.171.247:23 |
| GPU | 4 × NVIDIA A800-SXM4-80GB（共 320GB） |
| 启动方式 | `torchrun --nproc_per_node=4` (DDP) |
| 环境 | `/root/lora_grpo` venv，Python 3.10，TRL 1.4.0 |
| 训练脚本 | `train_32b_lora_grpo_trl.py` |
| 启动脚本 | `run_32b_lora_grpo_trl.sh`（`LORA_R=16 LORA_ALPHA=32 MAX_RESPONSE_LEN=1024 ... bash run_32b_lora_grpo_trl.sh`） |

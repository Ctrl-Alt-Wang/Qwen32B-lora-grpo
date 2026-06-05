# v3 超参数

与 v2 完全相同，仅修改 MAX_STEPS、OUTPUT_DIR 和 RUN_VERSION。

## 与 v2 的变化

| 参数 | v2 | v3 | 原因 |
|------|----|----|------|
| MAX_STEPS | 448 | **672** | 跑到收敛（预计需 600+ 步） |
| OUTPUT_DIR | `32B-LoRA-GRPO-TRL-v2` | `32B-LoRA-GRPO-TRL-v3` | 独立实验 |
| RUN_VERSION | v2 | **v3** | WandB 区分 |
| 起点 | base model（全新） | base model（全新） | 公平对比 |

## 完整参数

### 模型
| 参数 | 值 |
|------|----|
| Base model | Qwen2.5-32B-Instruct |
| 量化 | QLoRA NF4 4-bit，double quant，bf16 compute |

### LoRA
| 参数 | 值 |
|------|----|
| r | 8 |
| alpha | 16（有效放大 2×） |
| dropout | 0.05 |
| target_modules | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj |

### GRPO
| 参数 | 值 |
|------|----|
| num_generations | 4 |
| max_completion_length | 512 |
| beta（KL 系数） | 0.1 |
| epsilon（clip） | 0.2 |
| loss_type | grpo |

### 训练
| 参数 | 值 |
|------|----|
| learning_rate | 1e-4 |
| lr_scheduler_type | cosine |
| warmup_ratio | 0.03 |
| max_steps | **672** |
| per_device_train_batch_size | 1 |
| gradient_accumulation_steps | 4 |
| 有效 batch size | 4 GPU × 1 × 4 = 16 prompts/step |
| save_steps | 56 |
| save_total_limit | 2 |
| bf16 | True |
| gradient_checkpointing | True |

## 启动命令

```bash
cd /workspace/post_train/sql_agent
source /root/lora_grpo/bin/activate
MAX_STEPS=672 LR=1e-4 N_ROLLOUTS=4 SAVE_STEPS=56 RUN_VERSION=v3 \
  OUTPUT_DIR=/workspace/outputs/32B-LoRA-GRPO-TRL-v3 \
  nohup bash run_32b_lora_grpo_trl.sh \
  > logs/v3_rollout4_12epoch_20260604.log 2>&1 &
```

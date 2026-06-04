# v2 Hyperparameters

Identical to v1 except `N_ROLLOUTS`, `MAX_STEPS`, output dir, and run name.

## Changes from v1

| Parameter | v1 | v2 | Reason |
|-----------|----|----|--------|
| N_ROLLOUTS | 2 | **4** | Better advantage estimation |
| MAX_STEPS | 280 | **448** | 8 equivalent epochs |
| OUTPUT_DIR | `32B-LoRA-GRPO-TRL` | `32B-LoRA-GRPO-TRL-v2` | Separate experiment |
| run_name | `qwen32b-lora-r8-trl` | `qwen32b-lora-r8-rollout4-v2` | Versioned |
| save_total_limit | not set | **2** | Prevent checkpoint accumulation |

## Full Parameter Table

### Model
| Parameter | Value |
|-----------|-------|
| Base model | Qwen2.5-32B-Instruct |
| Quantization | QLoRA NF4 4-bit, double quant, bf16 compute |

### LoRA
| Parameter | Value |
|-----------|-------|
| r | 8 |
| alpha | 16 |
| dropout | 0.05 |
| target_modules | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj |

### GRPO
| Parameter | Value |
|-----------|-------|
| num_generations (N_ROLLOUTS) | **4** |
| max_completion_length | 512 |
| beta (KL coeff) | 0.1 |
| epsilon (clip) | 0.2 |
| loss_type | grpo |

### Training
| Parameter | Value |
|-----------|-------|
| learning_rate | 1e-4 |
| lr_scheduler_type | cosine |
| warmup_ratio | 0.03 |
| max_steps | 448 |
| per_device_train_batch_size | 1 |
| gradient_accumulation_steps | 4 |
| Effective batch | 4 GPU × 1 × 4 accum = 16 prompts/step |
| save_steps | 56 |
| save_total_limit | 2 |
| bf16 | True |
| gradient_checkpointing | True |

## Launch Command

```bash
cd /workspace/post_train/sql_agent
source /root/lora_grpo/bin/activate
MAX_STEPS=448 LR=1e-4 N_ROLLOUTS=4 SAVE_STEPS=56 RUN_VERSION=v2 \
  OUTPUT_DIR=/workspace/outputs/32B-LoRA-GRPO-TRL-v2 \
  nohup bash run_32b_lora_grpo_trl.sh \
  > logs/v2_rollout4_8epoch_20260603.log 2>&1 &
```

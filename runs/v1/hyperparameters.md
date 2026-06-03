# v1 Hyperparameters

## Model

| Parameter | Value |
|-----------|-------|
| Base model | Qwen2.5-32B-Instruct |
| Model path | `/workspace/models/Qwen2.5-32B-Instruct` |
| Quantization | QLoRA NF4 4-bit, double quant |
| Compute dtype | bfloat16 |
| Attn implementation | sdpa (scaled dot-product attention) |

## LoRA

| Parameter | Value |
|-----------|-------|
| r | 8 |
| alpha | 16 |
| dropout | 0.05 |
| target_modules | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj |
| bias | none |
| task_type | CAUSAL_LM |
| Effective alpha/r scale | 2.0× |

## GRPO / Training

| Parameter | Value |
|-----------|-------|
| Algorithm | GRPO (Group Relative Policy Optimization) |
| loss_type | grpo |
| num_generations (N_ROLLOUTS) | **2** |
| max_completion_length | 512 tokens |
| beta (KL coefficient) | 0.1 |
| epsilon (clip ratio) | 0.2 |
| learning_rate | 1e-4 |
| lr_scheduler_type | cosine |
| warmup_ratio | 0.03 (≈8 steps) |
| per_device_train_batch_size | 1 |
| gradient_accumulation_steps | 4 |
| Effective batch size | 4 GPU × 1 × 4 accum = **16 prompts/step** |
| max_steps | 280 |
| save_steps | 56 (every epoch) |
| save_total_limit | not set (all 6 checkpoints kept) |
| bf16 | True |
| gradient_checkpointing | True (use_reentrant=False) |
| dataloader_num_workers | 0 |

## Data

| Parameter | Value |
|-----------|-------|
| Training set | 900 prompts (medical EBM questions) |
| Format | System prompt + user question → PICO answer with citations |
| Max prompt length | 2048 tokens |
| Source | `/workspace/post_train/sql_agent/data/train.parquet` |

## Reward Function

| Component | Weight | Description |
|-----------|--------|-------------|
| Format reward | 0.5 | PICO structure detection (background/method/result/conclusion/intervention/population/outcome keywords, max 1.0) |
| RM score | 0.5 | Reward model at `http://117.50.48.176:8400/score`, sigmoid-normalized: `1/(1+exp(-(raw-(-3.4))/8.0))` |
| Total range | [-2.0, 2.0] | Clipped |

## Infrastructure

| Item | Value |
|------|-------|
| GPUs | 4 × A800-SXM4-80GB |
| GPU memory per card | ~22GB (QLoRA 4-bit) |
| Training launcher | `torchrun --nproc_per_node=4 --master_port=29600` |
| Environment | `/root/lora_grpo` venv |
| WandB project | `32b-lora-grpo-trl` |
| WandB server | `http://103.139.212.228:3005` |
| Output dir | `/workspace/outputs/32B-LoRA-GRPO-TRL` |
| Checkpoints saved | checkpoint-1, 56, 112, 168, 224, 280 (6 total) |
| Checkpoint size | ~397MB each (LoRA adapter only) |

## Launch Command

```bash
cd /workspace/post_train/sql_agent
source /root/lora_grpo/bin/activate
MAX_STEPS=280 LR=1e-4 N_ROLLOUTS=2 SAVE_STEPS=56 \
  nohup bash run_32b_lora_grpo_trl.sh \
  > logs/trl_grpo_5epoch_lr1e4_20260602_145101.log 2>&1 &
```

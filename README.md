# Qwen2.5-32B LoRA GRPO Training

Medical evidence-based medicine (EBM) agent trained with GRPO (Group Relative Policy Optimization) using LoRA on Qwen2.5-32B-Instruct.

## Task

SQL-based evidence retrieval agent that answers clinical questions in PICO structure with citations. Reward signal: PICO format score + reward model (RM) score.

## Hardware

- Server: `117.50.171.247:23`
- GPUs: 4 × NVIDIA A800-SXM4-80GB
- RAM: 925GB
- Disk: 571GB

## Training Framework

**TRL GRPOTrainer** (HuggingFace) with QLoRA 4-bit quantization + DDP via `torchrun`.

> Originally attempted VERL 0.7.1 + AgentLightning with vLLM HYBRID mode, but encountered deep incompatibility between vLLM 0.21.0 V1 engine and FSDP colocated workers on this machine. Switched to TRL after exhaustive debugging. See [`docs/environment_setup_journey.md`](docs/environment_setup_journey.md).

## Runs

| Run | Steps | Epochs | LR | N_ROLLOUTS | Reward Start | Reward End | Duration |
|-----|-------|--------|-----|------------|-------------|------------|----------|
| [v1](runs/v1/) | 280 | ~2.5 | 1e-4 | 2 | 0.645 | 0.713 | 8.4h |
| [v2](runs/v2/) | 448 | ~4 | 1e-4 | 4 | — | — | ~21h (running) |

## Repository Structure

```
├── README.md
├── scripts/
│   ├── train_32b_lora_grpo_trl.py   # Main training script
│   └── run_32b_lora_grpo_trl.sh     # Launch script
├── runs/
│   ├── v1/                           # First run analysis
│   │   ├── README.md                 # Run summary
│   │   ├── hyperparameters.md        # All hyperparameter details
│   │   └── training_log_analysis.md  # Detailed metrics analysis
│   └── v2/                           # Second run (N_ROLLOUTS=4)
├── docs/
│   └── environment_setup_journey.md  # VERL→TRL migration story
```

## Quick Start

```bash
ssh -p 23 root@117.50.171.247
cd /workspace/post_train/sql_agent
source /root/lora_grpo/bin/activate

# v2 training (N_ROLLOUTS=4, 8 epochs)
MAX_STEPS=448 LR=1e-4 N_ROLLOUTS=4 SAVE_STEPS=56 \
  OUTPUT_DIR=/workspace/outputs/32B-LoRA-GRPO-TRL-v2 \
  nohup bash run_32b_lora_grpo_trl.sh > logs/v2_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

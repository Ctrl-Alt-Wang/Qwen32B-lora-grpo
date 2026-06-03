#!/usr/bin/env bash
set -euo pipefail
export BASE_DIR="${BASE_DIR:-/workspace/post_train/sql_agent}"
export MODEL_PATH="${MODEL_PATH:-/workspace/models/Qwen2.5-32B-Instruct}"
export OUTPUT_DIR="${OUTPUT_DIR:-/workspace/outputs/32B-LoRA-GRPO-TRL}"
export LORA_R="${LORA_R:-8}"
export MAX_STEPS="${MAX_STEPS:-20}"
export LR="${LR:-1e-5}"
export GRAD_ACCUM="${GRAD_ACCUM:-4}"
export N_ROLLOUTS="${N_ROLLOUTS:-2}"
export REWARD_MODEL_URL="${REWARD_MODEL_URL:-http://117.50.48.176:8400/score}"
export ENABLE_RM_REWARD="${ENABLE_RM_REWARD:-1}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8
export WANDB_BASE_URL="http://103.139.212.228:3005"
export WANDB_API_KEY="local-f2ca8cd44276ac92ca0a2c12641a6902beb6847d"
export WANDB_PROJECT="32b-lora-grpo-trl"
source /root/lora_grpo/bin/activate
cd "${BASE_DIR}"
mkdir -p logs
LOG="logs/trl_grpo_$(date +%Y%m%d_%H%M%S).log"
echo "=== TRL GRPO 32B  steps=${MAX_STEPS}  lr=${LR}  log=${LOG} ==="
torchrun --nproc_per_node=4 --master_port=29600 train_32b_lora_grpo_trl.py 2>&1 | tee "${LOG}"

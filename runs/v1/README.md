# Run v1 — Baseline (N_ROLLOUTS=2, 5 epochs)

**Date:** 2026-06-02 14:51 → 23:17  
**Duration:** 8 hours 26 minutes (30,330 seconds)  
**Status:** Completed

## Summary

First successful LoRA GRPO training run on Qwen2.5-32B-Instruct. Reward improved from **0.645 → 0.713** (+10.5%), but training signal was weak due to `N_ROLLOUTS=2` — advantage estimation too coarse to trigger KL divergence or PPO clipping.

## Key Results

| Metric | Value |
|--------|-------|
| Initial reward | 0.645 |
| Final reward | 0.713 |
| Max reward (step 276) | 0.779 |
| Average reward (last 50 steps) | 0.707 |
| train_loss | 0.001618 |
| Total steps | 280 |
| Effective epochs | ~2.5 (TRL counts N_ROLLOUTS×samples) |
| Avg step time | ~104 seconds |
| KL divergence | 0.01–0.02 (small but nonzero) |
| Clip ratio | **0 throughout** (never triggered) |
| Grad norm | avg 0.053, max 0.186 |
| Mean completion length | 297.5 tokens |

## Reward Curve by Segment

| Steps | Avg Reward | Trend |
|-------|------------|-------|
| 0–55   | 0.685 | Fast rise |
| 56–111 | 0.706 | Moderate rise |
| 112–167 | 0.710 | Slowing |
| 168–223 | 0.706 | Plateau |
| 224–279 | 0.707 | Near-plateau |

Reward rose sharply in the first ~100 steps then flattened. Not yet converged.

## Hyperparameters

See [`hyperparameters.md`](hyperparameters.md) for full details.

Key settings:
- **Model:** Qwen2.5-32B-Instruct, QLoRA 4-bit (NF4, double quant)
- **LoRA:** r=8, alpha=16, dropout=0.05, all projection layers
- **N_ROLLOUTS:** 2 (identified as main bottleneck)
- **LR:** 1e-4, cosine decay, 3% warmup
- **Batch:** 4 GPU × 1 sample × grad_accum=4 = 16 prompts/optimizer step
- **Beta (KL coeff):** 0.1
- **Epsilon (clip):** 0.2

## Identified Issues

### 1. N_ROLLOUTS=2 — Main Bottleneck
With only 2 rollouts per prompt, advantage estimates are extremely noisy. GRPO needs at least 4–8 rollouts for stable advantage normalization. Evidence:
- `clip_ratio = 0` throughout all 280 steps (policy updates too small to trigger ε=0.2 clip)
- KL stayed in 0.01–0.02 range (should grow as policy diverges from reference)
- Reward plateau after step 100 — signal too weak to push further

### 2. cosine LR Scheduler Decayed to Zero
At step 280 (end of training), `learning_rate ≈ 3.36e-9` (effectively 0). Cannot simply resume training — scheduler must be reset.

### 3. WandB Run Name Not Versioned
Run was logged under `run_name="qwen32b-lora-r8-trl"` with no version suffix, making it hard to distinguish from future runs in WandB dashboard.

## Decision: Run v2 with N_ROLLOUTS=4

- N_ROLLOUTS: 2 → **4** (better advantage estimation)
- MAX_STEPS: 280 → **448** (8 epochs equivalent)
- Output dir: new separate directory `32B-LoRA-GRPO-TRL-v2`
- WandB run_name: versioned as `qwen32b-lora-r8-rollout4-v2`
- Added `save_total_limit=2` to avoid checkpoint accumulation
- Expected duration: ~21 hours (each step ~170s with doubled rollouts)

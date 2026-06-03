# v1 Training Log Analysis

**Log file:** `trl_grpo_5epoch_lr1e4_20260602_145101.log`  
**Total lines:** 2151  
**Steps parsed:** 280

## Overall Statistics

| Metric | Value |
|--------|-------|
| Total training time | 8h 26min (30,330 seconds) |
| Total steps | 280 |
| Avg seconds/step | ~104s |
| Min step time | ~90s |
| Max step time | ~124s |
| Total tokens processed | ~6.06M |
| train_samples_per_second | 0.148 |
| train_loss (final) | 0.001618 |
| Effective epochs (TRL counting) | 2.5 |

> **Note on epoch count:** TRL's GRPOTrainer counts epochs by iterating through `num_generations × dataset_size` total samples. With N_ROLLOUTS=2 and 900 prompts, one "TRL epoch" = 900×2=1800 samples, but only 112 optimizer steps (not 56). Thus 280 steps ≈ 2.5 TRL epochs, not 5 data epochs as originally planned.

## Reward Metrics

| Metric | Value |
|--------|-------|
| Initial reward (step 0) | 0.645 |
| Final reward (step 279) | 0.713 |
| Max reward | 0.779 (step 276) |
| Min reward | 0.641 |
| Avg first 50 steps | 0.683 |
| Avg last 50 steps | 0.707 |
| Total improvement | +10.5% |

### Reward by Training Segment

| Steps | Avg Reward | Notes |
|-------|------------|-------|
| 0–55   | 0.685 | Rapid initial learning |
| 56–111 | 0.706 | Continued improvement |
| 112–167 | 0.710 | Slowing down |
| 168–223 | 0.706 | Slight oscillation, plateau |
| 224–279 | 0.707 | Near-plateau, not converged |

### Reward Samples (selected steps)

| Step | Reward | LR | KL | Grad Norm |
|------|--------|----|----|-----------|
| 0 | 0.645 | 1.00e-4 | — | — |
| 1 | 0.648 | ~9.9e-5 | 0.016 | 0.047 |
| 10 | ~0.700 | ~9.5e-5 | — | — |
| 50 | ~0.701 | ~8.5e-5 | — | — |
| 100 | ~0.712 | ~6.5e-5 | — | — |
| 200 | ~0.708 | ~2.0e-5 | — | — |
| 279 | 0.713 | 3.36e-9 | 0.011 | 0.042 |

## KL Divergence

| Metric | Value |
|--------|-------|
| Average KL | 0.01618 |
| Max KL | 0.04985 |
| Steps with KL > 0.001 | 272 / 280 |
| Trend | Small positive values throughout; expected to grow with N_ROLLOUTS=4 |

KL was very small (0.01–0.05) — consistent with conservative policy updates due to N_ROLLOUTS=2. The reference policy (frozen base model) was barely diverged from, suggesting the 0.1 KL penalty was barely active.

## Clip Ratio

```
clip_ratio/low_mean  = 0 (all 280 steps)
clip_ratio/high_mean = 0 (all 280 steps)
clip_ratio/region_mean = 0 (all 280 steps)
```

**Never triggered.** With N_ROLLOUTS=2, advantage estimates are near-zero for most prompts (baseline ≈ mean of 2 samples), so policy gradient steps are tiny and never cross the ε=0.2 clip boundary. This is the primary bottleneck.

## Gradient Norm

| Metric | Value |
|--------|-------|
| Average | 0.053 |
| Max | 0.186 |
| Trend | Stable, no spikes |

Healthy gradient norms — no exploding gradients despite 1e-4 LR.

## Completion Lengths

| Metric | Value |
|--------|-------|
| Average mean length | 297.5 tokens |
| Typical range | 200–465 tokens |
| Max completion limit | 512 tokens |
| Clipped ratio | 0 (model never hit the 512-token limit) |

Model generates 200–450 token answers, well within the 512 limit. No truncation issues.

## Learning Rate Schedule (cosine)

| Step | LR |
|------|----|
| 0 | 1.00e-4 |
| 8 (warmup end) | 1.00e-4 |
| 56 | ~8.5e-5 |
| 112 | ~6.0e-5 |
| 168 | ~3.8e-5 |
| 224 | ~2.0e-5 |
| 280 (end) | ~3.4e-9 ≈ 0 |

LR reached near-zero by end of training. **Cannot resume** without resetting the scheduler.

## Observations & Conclusions

### What worked well
1. Training was stable throughout — no NaN, no loss spikes
2. Reward improved monotonically in the early phase (+6% in first 100 steps)
3. QLoRA 4-bit kept GPU memory to 22GB/card — 4 GPUs sufficient
4. RM server integration functioned correctly (mean reward ~0.63–0.72)
5. PICO format reward + RM score combination gave useful gradient signal

### What needs improvement
1. **N_ROLLOUTS=2 is insufficient** — advantage normalization over 2 samples is too noisy. GRPO works best with 4–8+ rollouts.
2. **Clip ratio never triggered** — the policy barely moved. More rollouts → better advantage estimates → larger/more selective updates → clip engages → KL grows meaningfully.
3. **Training not converged** — reward curve still rising at step 280. Need at least 400–500 steps to plateau.
4. **No save_total_limit** — 6 checkpoints accumulated (each ~397MB). Should add `save_total_limit=2`.

### Plan for v2
- N_ROLLOUTS: 2 → **4** (double rollouts per prompt)
- MAX_STEPS: 280 → **448** (8 equivalent data epochs)
- save_total_limit: **2** (keep only last 2 checkpoints)
- New output dir: `32B-LoRA-GRPO-TRL-v2`
- WandB run_name versioned: `qwen32b-lora-r8-rollout4-v2`
- Expected: clip_ratio will finally trigger, KL will grow to 0.05–0.2 range, reward may reach 0.75+

# v2 Training Log Analysis

**Log:** `v2_rollout4_8epoch_20260603.log`  
**Steps:** 448  
**Duration:** ~13h (01:51 → 14:51, 2026-06-03)

## Summary Statistics

| Metric | Value |
|--------|-------|
| Total steps | 448 |
| Total time | ~13h |
| Avg step time | 104.1s |
| Min / Max step time | 68.9s / 130.3s |
| Total tokens processed | ~9.82M |
| Final train_loss | 0.006809 |
| Final epoch (TRL) | 1.991 |

## Reward

| Metric | Value |
|--------|-------|
| Initial (step 0) | 0.646 |
| Final (step 447) | 0.748 |
| Max (step ~330) | **0.808** |
| Min | 0.637 |
| Avg steps 0–50 | 0.691 |
| Avg steps 100–200 | 0.749 |
| Avg steps 200–350 | 0.762 |
| Avg last 50 | 0.753 |
| Avg last 100 | 0.754 |

### Reward Progression by 56-step Blocks

| Block | Steps | Avg | Max | KL avg | Notes |
|-------|-------|-----|-----|--------|-------|
| 1 | 0–55 | 0.695 | 0.756 | 0.020 | Fast initial rise |
| 2 | 56–111 | 0.747 | 0.780 | 0.037 | Big jump +7.5% |
| 3 | 112–167 | 0.745 | 0.803 | 0.069 | KL stabilises |
| 4 | 168–223 | 0.758 | 0.789 | 0.077 | KL peak |
| 5 | 224–279 | 0.754 | 0.788 | 0.061 | Plateau ~0.755 |
| 6 | 280–335 | 0.769 | **0.808** | 0.075 | Local high |
| 7 | 336–391 | 0.757 | 0.783 | 0.074 | Oscillating |
| 8 | 392–447 | 0.753 | 0.784 | 0.069 | Still oscillating |

**Convergence not reached** — reward is still oscillating ±0.02 at end. More steps needed.

## KL Divergence

| Metric | v1 | v2 |
|--------|----|----|
| Average | 0.016 | **0.060** |
| Maximum | 0.050 | **0.152** |
| Minimum | ~0 | 0.000 |

N_ROLLOUTS=4 activated real policy divergence. KL grew from ~0.02 (early steps) to a stable range of 0.06–0.08. The beta=0.1 penalty contributes ~0.006–0.008 to the total loss, meaningfully regularizing the policy.

**KL trajectory:**
- Steps 0–50: 0.020 (warming up)
- Steps 50–150: growing to 0.07 (policy diverging from reference)
- Steps 150–448: stable 0.06–0.08 (equilibrium between policy gradient and KL penalty)

## Clip Ratio

```
clip_ratio/high_mean = 0   (all 448 steps)
clip_ratio/low_mean  = 0   (all 448 steps)
clip_ratio/region_mean = 0 (all 448 steps)
```

**Expected behavior for TRL online GRPO.** TRL performs 1 optimization step per rollout batch. The ratio of new vs old policy (computed in the same forward pass) stays near 1.0, rarely exceeding ε=0.2. This is a fundamental property of online/on-policy GRPO, not a misconfiguration.

The KL divergence (avg 0.060) serves as the primary regularizer in this setting.

## Loss

| Metric | v1 | v2 |
|--------|----|----|
| Final loss | 0.00162 | **0.00681** |
| Interpretation | Mainly policy gradient | Policy gradient + active KL |

The 4× loss increase is healthy — it reflects `β × KL = 0.1 × 0.068 = 0.0068` now contributing significantly. v1's near-zero KL meant the KL term added nothing.

## Gradient Norm

| Metric | v1 | v2 |
|--------|----|----|
| Average | 0.053 | **0.068** |
| Maximum | 0.186 | **0.523** |

Larger grad norms in v2 reflect stronger learning signal. No gradient explosion (single spike to 0.52 isolated). Cosine LR schedule keeps overall gradients under control.

## Entropy

| Metric | Value |
|--------|-------|
| Average | 0.365 |
| Final | 0.358 |
| Trend | Decreasing ~0.40 → ~0.36 |

Gradual entropy decrease is healthy — model becoming more focused on its preferred PICO answer style without collapsing (entropy never approaches 0).

## Completion Length

| Metric | Value |
|--------|-------|
| Average across run | 313.7 tokens |
| Early steps | ~350 tokens |
| Final steps | ~284 tokens |
| Clipped ratio | 0 (never hit 512-token limit) |

**Notable trend:** Length decreases monotonically as training progresses. The model learns that concise, well-structured PICO answers score higher than verbose ones. This is a positive reward signal effect.

## Step Time & Profiling

| Component | Time |
|-----------|------|
| `transformers.generate` | 60–110s (decreases as responses shorten) |
| `reward_fn` (RM + format) | ~0.6–0.9s |
| `compute_loss` | ~0.6–1.4s |
| `_get_per_token_logps_and_entropies` | ~0.4–1.2s |
| `_calculate_rewards` | ~0.4–0.9s |
| `_prepare_inputs` | ~8 μs (negligible) |

Generation accounts for >90% of step time. The **decreasing generate time** (110s → 60s) corresponds to the model learning to write shorter answers, resulting in actual speedup during training.

**N_ROLLOUTS=4 added no net overhead** (avg 104s vs v1's 104s), because:
1. 4 completions per prompt fit in a single `generate()` batch call
2. 4 GPU parallelism handles 4×16 = 64 sequences simultaneously
3. Shorter completions compensated for the 2× rollout count

## Selected Step Snapshots

| Step | Reward | KL | Grad Norm | LR | Length |
|------|--------|----|-----------|----|--------|
| 0 | 0.646 | 0.000 | — | 1.00e-4 | — |
| 1 | 0.648 | 0.016 | 0.050 | ~9.9e-5 | 264 |
| 56 | ~0.745 | 0.038 | — | ~8.5e-5 | ~310 |
| 112 | ~0.750 | 0.069 | — | ~6.0e-5 | ~320 |
| 224 | ~0.758 | 0.077 | — | ~2.0e-5 | ~300 |
| 335 | ~0.769 | 0.075 | — | ~5.0e-6 | ~290 |
| 447 | 0.748 | 0.068 | 0.078 | 1.31e-9 | 284 |

## Comparison: v1 vs v2

| Metric | v1 | v2 | Improvement |
|--------|----|----|-------------|
| Max reward | 0.779 | **0.808** | +3.7% |
| Final reward | 0.713 | **0.748** | +4.9% |
| Avg last 50 | 0.707 | **0.753** | +6.5% |
| KL avg | 0.016 | **0.060** | +275% |
| Reward_std avg | 0.035 | **0.045** | +28% |
| Loss (KL active) | no | **yes** | |
| Clip triggered | no | no | same |
| Convergence | no | no | needs more steps |

N_ROLLOUTS=4 clearly improved training quality. Clip not triggering is expected for online GRPO.

## Outlook for v3

Based on the segment data (blocks 5–8 still at 0.753–0.769):
- Reward slope at end: ~+0.002 per 56 steps
- Estimated convergence plateau: ~0.78–0.80
- Steps to convergence from scratch: ~600–700
- v3 plan: MAX_STEPS=672, all other hyperparameters identical, fresh start from base model

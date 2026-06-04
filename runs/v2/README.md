# Run v2 — N_ROLLOUTS=4, 8 Epochs

**Date:** 2026-06-03 01:51 → 14:51  
**Duration:** ~13 hours (448 steps × avg 104s/step)  
**Status:** Completed  
**WandB:** `http://103.139.212.228:3005/johnson/32b-lora-grpo-trl/runs/f9wu6by1`  
**Run name:** `qwen32b-lora-r8-rollout4-v2`

## Motivation

v1 identified `N_ROLLOUTS=2` as the key bottleneck: clip_ratio=0 throughout 280 steps, KL barely active (0.016). Doubling to N_ROLLOUTS=4 to get better advantage estimates and stronger training signal.

## Key Results

| Metric | v1 (N_ROLLOUTS=2) | v2 (N_ROLLOUTS=4) | Change |
|--------|-------------------|-------------------|--------|
| Initial reward | 0.645 | 0.645 | — |
| Final reward | 0.713 | **0.748** | +4.9% |
| Max reward | 0.779 | **0.808** | +3.7% |
| Avg last 50 steps | 0.707 | **0.753** | +6.5% |
| Avg last 100 steps | — | **0.754** | — |
| KL avg | 0.016 | **0.060** | +275% |
| KL max | 0.050 | **0.152** | +204% |
| Reward_std avg | 0.035 | **0.045** | +28% |
| Grad norm avg | 0.053 | **0.068** | +28% |
| Final loss | 0.00162 | **0.00681** | +320% (KL active) |
| Clip ratio | 0 | **0** | unchanged |
| Avg step_time | 104s | **104s** | no overhead |
| Total steps | 280 | 448 | +60% |
| Effective TRL epochs | 2.5 | **2.0** | — |

## Reward Curve

```
Segment (56 steps)  Avg Reward  Max      KL avg
steps   0– 55:      0.695       0.755    0.020   ← fast initial rise
steps  56–111:      0.747       0.779    0.037   ← big jump (+7.5%)
steps 112–167:      0.745       0.803    0.069   ← peak, KL stabilizes
steps 168–223:      0.758       0.789    0.077   ← plateau ~0.76
steps 224–279:      0.754       0.788    0.061
steps 280–335:      0.769       0.808    0.075   ← local high
steps 336–391:      0.757       0.783    0.074
steps 392–447:      0.753       0.784    0.069   ← still oscillating
```

**Not yet converged** — reward oscillates between 0.75–0.77 in last 200 steps without clear plateau. Curve still has room to improve.

## Detailed Metric Analysis

### Reward
- Rapid improvement in steps 0–100 (+0.06)
- Plateau phase from step 100 onward (~0.75)
- Still oscillating at end — not converged

### KL Divergence
- v1: avg 0.016 (nearly inactive)
- **v2: avg 0.060** — 4× increase confirms N_ROLLOUTS=4 enables real policy divergence
- KL grew from ~0.020 (early) → ~0.075 (mid) → ~0.069 (final)
- beta=0.1 KL penalty is now meaningfully active (contributing 0.1×0.068=0.0068 to loss)

### Clip Ratio — Still Zero
All 448 steps: `clip_ratio/high_mean = clip_ratio/low_mean = 0`

**This is expected behavior for TRL's online GRPO**, not a bug. In TRL's GRPOTrainer, rollout and optimization happen in the same step (1 optimization epoch per rollout). The "old" policy used for the ratio is the policy at generation time; with only 1 gradient step the ratio stays near 1.0 and rarely crosses ε=0.2.

Clip_ratio triggering requires either:
- Multi-epoch optimization per rollout (like PPO), or
- Very large advantage estimates (N_ROLLOUTS ≥ 8 helps)

The KL divergence (0.06) is the actual effective regularizer in this online setting.

### Loss
- v1 final: 0.00162 (mostly policy gradient, KL near-zero)
- **v2 final: 0.00681** — 4× higher because KL loss (β=0.1) now contributes
- This increase is healthy: KL regularization is doing its job

### Gradient Norm
- avg: 0.068 (vs v1: 0.053) — larger updates due to stronger reward signal
- max: 0.523 — no gradient explosion
- Stable throughout

### Entropy
- avg: 0.365, decreasing from ~0.40 → ~0.36
- Healthy decrease: model becoming more deterministic in its PICO answer style
- Not collapsed (entropy > 0 throughout)

### Completion Length
- avg: 313.7 tokens (v1: 297.5)
- Trend: **decreasing** over training — from ~350 early to ~284 at end
- Model learning to write more concise answers while maintaining reward
- No truncation: clipped_ratio = 0 throughout (well below 512-token limit)

### Step Time
- avg: **104.1s** (same as v1's 104s despite 2× rollouts!)
- N_ROLLOUTS=4 runs 4 completions per prompt instead of 2, but TRL batches them efficiently on 4 GPUs
- min: 69s, max: 130s

### Profiling Breakdown
From WandB profiling panel (approximate):
| Phase | Time |
|-------|------|
| `transformers.generate` | 60–110s (drops over training as model generates shorter responses) |
| `reward_fn` | ~0.6–0.9s |
| `compute_loss` | ~0.6–1.4s |
| `_get_per_token_logps_and_entropies` | ~0.4–1.2s |
| `_calculate_rewards` | ~0.4–0.9s |
| `_prepare_inputs` | ~8–16 μs (negligible) |

Generation dominates (>90% of step time).

### Epoch Count (TRL behavior)
- Epoch at step 448: **1.991** (≈ 2 TRL epochs)
- TRL's GRPO epoch counting: 1 epoch = N_ROLLOUTS × dataset_size samples consumed
- With N_ROLLOUTS=4, 900 prompts: 1 TRL epoch = 3600 samples = 224 optimizer steps
- So 448 steps ≈ 2 data passes × 4 rollouts = 8 equivalent rollout epochs

### Checkpoints
Only last 2 kept (save_total_limit=2):
- `checkpoint-392` (397MB)
- `checkpoint-448` (397MB, final)
- `adapter_model.safetensors` (129MB) — final merged adapter

Total output: ~1.1GB

## Conclusions

### What improved vs v1
1. **Reward plateau lifted**: 0.71 → 0.75 final (+5.6%)
2. **KL now active**: 0.016 → 0.060 (4× increase — real policy divergence happening)
3. **Reward_std healthier**: 0.035 → 0.045 (better diversity in per-prompt rewards)
4. **Grad norms larger**: 0.053 → 0.068 (stronger learning signal)
5. **No step-time overhead**: parallel 4-rollout batching as efficient as 2-rollout

### Remaining issues
1. **Clip ratio still zero** — expected for TRL online GRPO, not fixable without multi-epoch optimization or much larger N_ROLLOUTS
2. **Not converged** — reward curve still oscillating at step 448, needs more steps
3. **2 TRL epochs only** — despite planning "8 epochs," TRL's counting means only 2 data passes per rollout cycle

## Decision: Run v3 (convergence run)

v2 shows the hyperparameters are correct. The only remaining question is how many steps to converge.

**v3 plan:**
- Same config as v2 (N_ROLLOUTS=4, LR=1e-4, all hyperparameters identical)
- MAX_STEPS=**672** (3 TRL epochs = 12 equivalent data epochs)
- Fresh start from base model (for clean experimental comparison with v2)
- Expected convergence around step 500–600 based on reward slope
- New output dir: `32B-LoRA-GRPO-TRL-v3`
- WandB run: `qwen32b-lora-r8-rollout4-v3`
- Estimated duration: **~19 hours**

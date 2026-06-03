# Environment Setup Journey: VERL → TRL

This document records the full debugging process of setting up LoRA GRPO training on this server. It took approximately **2 full days** before training ran successfully.

**Server:** `117.50.171.247:23` (root), 4×A800-80GB, CUDA 13.0, Driver 580.105.08  
**Target:** Qwen2.5-32B-Instruct + LoRA GRPO for medical EBM agent

---

## Phase 1: VERL + AgentLightning Attempt (2026-06-01 to 2026-06-02)

### Initial Setup

The original plan was to use **VERL 0.7.1 + AgentLightning** with vLLM HYBRID mode (colocated vLLM TP workers + FSDP actors on same GPUs). Framework: `train_32b_lora_grpo.py`.

Environment: `/root/lora_grpo` venv, Python 3.x, built on top of a pre-existing `sft-qwen` environment.

> Note: Previous successful runs (E1-E11) on this server used **Swift SFT**, not GRPO. This was the server's **first ever LoRA GRPO attempt**.

### Bug 1: Python/Conda Not in PATH

SSH default shell had empty PATH. All `python`/`conda` commands failed.

**Fix:** Use full path `/root/lora_grpo/bin/python` or source the venv in each command.

### Bug 2: vLLM ABI Compatibility — 5-Layer Nested Issue

`vllm_torch_compat.so` (ABI patch shim) used `dlsym(RTLD_DEFAULT, ...)` to find new torch 2.12.0 `Library::def` symbols. But `libtorch_cpu.so` was loaded with `RTLD_LOCAL`, making `dlsym` always return NULL → all vLLM custom ops registration failed.

**Fixes applied (5 files):**
1. `vllm/platforms/cuda.py`: Load `libtorch_cpu.so`/`libtorch_cuda.so` as `RTLD_GLOBAL`; change attention backend priority to FLASHINFER > FLASH_ATTN
2. `vllm/model_executor/layers/activation.py`: Add `hasattr` check + native fallback for `SiluAndMul`
3. `vllm/_custom_ops.py`: Add native fallback for `rms_norm` and `fused_add_rms_norm`
4. torchvision: Upgraded 0.26.0 → 0.27.0+cu126 (match torch 2.12.0)
5. torch: Restored to 2.12.0+cu126

Result: vLLM smoke test with TP=4 passed (`[PASS] vLLM TP=4 OK`).

### Bug 3: VERL/AgentLightning API Mismatch

AgentLightning was built against VERL 0.6.x API, but VERL 0.7.1 changed:
- `RayPPOTrainer.__init__` removed `reward_fn` constructor argument
- `NaiveRewardManager` removed `num_examine` parameter
- `RolloutConfig` added required `name` field
- `HFModelConfig` removed `bf16` and `lora_dropout` fields
- `lr_warmup_steps_ratio` replaced `warmup_ratio`

**Fixes:** Patched `AgentLightningTrainer.__init__` to intercept and drop removed parameters; added missing fields.

### Bug 4: trl 1.4.0 Removed `AutoModelForCausalLMWithValueHead`

VERL's `monkey_patch.py` tried to import this class, which was removed in trl 1.x.

**Fix:** Added backward-compatible stub in VERL's monkey_patch.py with try/except.

### Bug 5: FSDP2 FSDP Set Indexing Error

`set` does not support subscript indexing → assertion failed during FSDP2 wrapping.

**Fix:** Changed set to list in the assertion.

### Bug 6: AgentLoopManager Missing `wake_up()` / `sleep()`

VERL trainer called `self.async_rollout_manager.wake_up()` in `_validate()`, but `AgentLoopManager` had no such method.

**Fix:** Added `@auto_await async def wake_up(self)` and `sleep(self)` proxy methods that called `asyncio.gather(*[replica.wake_up() for replica in self.rollout_replicas])`.

### Bug 7: `wake_up()` Raises ValueError in HYBRID Mode

vLLM `vllm_async_server.py` explicitly raised `ValueError` for `wake_up()` in HYBRID (colocated) mode.

**Fix:** Changed to skip (no-op) for HYBRID mode.

### Bug 8: vLLM CUDA Graph Error on First Rollout

`CUDA error: invalid argument` during attention metadata construction. Attempted fix: `enforce_eager=True`.

### Bug 9: LoRA punica CUDA Error

vLLM's `punica_base.py` line 193: `_token_lora_indices.copy_(base_indices)` — device mismatch in HYBRID TP mode.

**Attempted fix:** `lora.merge=True` (merge LoRA into base model before passing to vLLM, bypassing `--enable_lora`).

### Bug 10: `update_weights` Contaminates CUDA State

Added initial `actor_rollout_wg.update_weights(global_steps=0)` call before first rollout to sync real FSDP weights to vLLM (otherwise vLLM starts with random dummy weights). But this caused `actor.engine.to("cpu")` → NCCL AllGather → contaminated vLLM TP workers' CUDA stream state → all subsequent rollouts crashed.

### Root Cause Confirmed

**vLLM 0.21.0 V1 engine's `vLLMColocateWorkerExtension` (HYBRID mode) is fundamentally incompatible with this A800 machine's CUDA/NCCL setup.** The V1 engine is hardcoded in VERL 0.7.1 (`from vllm.v1.engine.async_llm import AsyncLLM`) — cannot switch to V0.

After 9 restart attempts and ~1 full day of debugging, we confirmed: this is not a fixable code-logic bug, but a platform-level incompatibility.

---

## Phase 2: TRL GRPOTrainer (2026-06-02)

Switched to **TRL 1.4.0 GRPOTrainer** — uses HuggingFace `generate()` for rollouts, DDP via `torchrun`, no vLLM, no FSDP.

### Bug 11: GRPOConfig Parameter Names Changed in TRL 1.4.0

- `max_prompt_length` → removed (use `generation_kwargs`)
- `model_init_kwargs` with `load_in_4bit` → not valid in GRPOConfig; must load model manually with `BitsAndBytesConfig`

**Fix:** Remove `model_init_kwargs` from GRPOConfig; load QLoRA model explicitly before passing to `GRPOTrainer`.

### Bug 12: N_ROLLOUTS=1 Invalid

GRPO requires `num_generations >= 2`.

**Fix:** Set `N_ROLLOUTS=2` minimum.

### Bug 13: Reward Function Receives List-of-Dicts

TRL 1.4.0 passes completions as `list[dict]` (chat message format) not `str`.

**Fix:** Added extraction: `a = " ".join(m.get("content","") for m in c if isinstance(m, dict))`.

### Training Started Successfully

After 3 small bug fixes, TRL training ran stably:
- Model loaded (QLoRA 4-bit, 22GB/GPU) in ~10 seconds
- Step 1/20 completed in 104s
- `reward mean=0.633`, `grad_norm=0.047` — normal values

### Smoke test → Full 5-epoch run

After 20-step smoke test passed:
- Launched `MAX_STEPS=280 LR=1e-4 SAVE_STEPS=56`
- 4 GPUs × 22GB/card
- ~104s/step, total 8.4 hours
- Reward: 0.645 → 0.713 (+10.5%)

---

## Key Lessons

1. **VERL HYBRID mode requires careful version pinning.** vLLM 0.21.0 + VERL 0.7.1 HYBRID on A800 has a known CUDA stream contamination bug when colocating FSDP + vLLM TP workers. Test on a single GPU first.

2. **TRL GRPOTrainer is the path of least resistance for LoRA GRPO.** No vLLM, no FSDP, just DDP + HF generate. Slower per step but reliable.

3. **AgentLightning targets VERL 0.6.x.** If using AgentLightning, pin VERL to 0.6.1 (or patch AgentLightningTrainer extensively as done here).

4. **trl 1.x breaking changes:** `AutoModelForCausalLMWithValueHead` removed, `GRPOConfig` params changed, reward function receives list-of-dicts.

5. **Start with smoke test (N_STEPS=20).** Validates entire pipeline in <35 minutes before committing to 8+ hour runs.

---

## Final Working Environment

```
Python: /root/lora_grpo/bin/python
torch: 2.12.0+cu126
transformers: 4.x
trl: 1.4.0
peft: latest
bitsandbytes: latest
accelerate: 1.13.0
```

**No vLLM needed for TRL-based training.**

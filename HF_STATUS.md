# RWKV-7 HF Adapter Status

This document is the contributor-facing status page for the **Hugging Face /
Transformers adapter** track. The repository is intentionally scoped to HF
loading, generation, training, PEFT/TRL compatibility, HF state-cache helpers,
quantized inference, and reproducible benchmark evidence.

vLLM, SGLang, DFlash, and standalone serving engines are follow-up projects and
must not block this HF adapter deliverable.

## Current summary

| Area | Status | Notes |
|---|---|---|
| HF load / save / generate | Done | `AutoConfig`, `AutoTokenizer`, `AutoModelForCausalLM`, `save_pretrained`, `from_pretrained`, and `generate(use_cache=True)` are covered. |
| Official checkpoint conversion | Done | Official `.pth` checkpoints can be converted to HF `safetensors`; shape inference covers published model sizes. |
| Correctness alignment | Done for smoke baseline | 0.1B V100 alignment against the official `rwkv` path passes top-k / cosine / greedy-window gates. |
| PEFT | Done for smoke + adapter lifecycle | LoRA forward/backward and adapter save/load/merge smokes exist. |
| Trainer / TRL | Done for small smoke | HF Trainer plus TRL SFT/DPO/GRPO smoke tests exist and verify finite loss plus trainable parameter deltas. |
| DeepSpeed ZeRO | Done for small 2xV100 smoke | ZeRO-2 and ZeRO-3 HF Trainer smoke rows are recorded on 2 x V100. |
| HF recurrent cache helpers | Done for current adapter | `RWKV7StateCache` supports select/reorder/drop/compact, offload/restore, chunked prefill, and cache telemetry. |
| Quantized loading | Functional | bitsandbytes 8-bit and 4-bit loading/generation work and reduce footprint; speed is still an open production gap. |
| Native/no-FLA backend | Experimental | Useful for upstream/AMD/CPU fallback work; not yet a production replacement for the wrapper path. |
| Production performance | Partial | V100 fast-token/native-graph paths improve decode, but Albatross-level and quantized-speed gates are not fully closed. |
| Cross-card validation | Partial | V100 is the baseline; 2xV100 and some Blackwell evidence exists. More cards are needed. |

## Already completed

### 1. HF model API

- `RWKV7Config`, `RWKV7Model`, and `RWKV7ForCausalLM` exist.
- Remote-code loading works with:
  - `AutoConfig.from_pretrained(..., trust_remote_code=True)`
  - `AutoTokenizer.from_pretrained(..., trust_remote_code=True)`
  - `AutoModelForCausalLM.from_pretrained(..., trust_remote_code=True)`
- `generate(..., use_cache=True)` works.
- `save_pretrained` / reload roundtrip is covered.
- `labels`, `attention_mask`, `past_key_values`, and generation preparation paths
  are covered by smoke/API tests.

Primary evidence:

- `tests/smoke_hf_generate.py`
- `tests/test_hf_api_contract.py`
- `tests/test_reload_roundtrip.py`
- `tests/test_sync_hf_adapter_code.py`

### 2. Conversion and correctness

- Official RWKV-7 `.pth` checkpoints can be converted to HF-style directories.
- Conversion infers model dimensions from weight shapes instead of hardcoding the
  0.1B layout.
- Batch conversion can write a reproducible manifest with source paths, hashes,
  options, commands, and status.
- The 0.1B V100 baseline aligns with the official `rwkv` path on top-k,
  cosine-similarity, and greedy-window checks.

Primary evidence:

- `scripts/convert_rwkv7_to_hf.py`
- `scripts/batch_convert_rwkv7_to_hf.py`
- `tests/test_convert_config.py`
- `tests/test_batch_convert_manifest.py`
- `tests/test_official_alignment.py`
- `BENCHMARK.md`

### 3. HF training ecosystem

Small-model smoke coverage already exists for the main HF training stack:

- PEFT LoRA forward / loss / backward.
- PEFT adapter save / load / merge.
- HF Trainer causal-LM training smoke.
- TRL `SFTTrainer` smoke.
- TRL `DPOTrainer` smoke.
- TRL `GRPOTrainer` smoke.
- Trainer checkpoint resume smoke for the native/no-FLA path.
- DeepSpeed ZeRO-2 and ZeRO-3 HF Trainer smoke on 2 x V100.

Primary evidence:

- `tests/test_peft_lora.py`
- `tests/test_hf_training_smoke.py`
- `tests/test_hf_rl_training_smoke.py`
- `tests/test_native_trainer_smoke.py`
- `tests/test_native_sft_smoke.py`
- `tests/test_native_dpo_smoke.py`
- `tests/test_native_grpo_smoke.py`
- `tests/test_native_peft_save_load_merge.py`
- `tests/test_native_trainer_resume_smoke.py`
- `tests/test_deepspeed_configs.py`
- `tests/test_deepspeed_training_smoke.py`

### 4. HF serving-style cache helpers

The HF adapter exposes RWKV recurrent-state operations needed by serving-like
callers without making this repository a vLLM/SGLang implementation:

- Cache allocation and reuse.
- Batch select/reorder/drop/compact.
- CPU offload and restore.
- Detach and dtype/device movement.
- Chunked prefill correctness checks.
- Dynamic-batch telemetry and cache hit-rate metrics.

Primary evidence:

- `tests/test_fast_cache.py`
- `tests/test_fast_decode_api.py`
- `tests/test_batch_cache.py`
- `tests/test_dynamic_batch_cache.py`
- `tests/test_chunked_prefill.py`
- `tests/test_native_graph_cache.py`
- `bench/bench_dynamic_batch.py`
- `bench/bench_chunked_prefill.py`

### 5. Quantized inference functionality

- bitsandbytes 8-bit and 4-bit loading works.
- Quantized generation smoke exists.
- Memory-footprint reduction is recorded.
- Current W8/W4 speed is not production-complete; fused/native quantized
  kernels are still required for the target of being no slower than fp16.

Primary evidence:

- `tests/test_quantized_inference.py`
- `tests/test_native_bnb_quant_smoke.py`
- `bench/bench_quantization.py`
- `bench/bench_native_quant_gemv.py`
- `bench/bench_native_quant_w4_gemv.py`
- `bench/bench_native_quant_rkv.py`
- `bench/bench_native_quant_w4_rkv.py`

## Hardware / card adaptation status

V100 is the active development and regression baseline. The goal is not merely
"it runs on one card"; the HF adapter should have clear behavior across common
professional and consumer cards.

| Hardware target | Current status | What contributors can add |
|---|---|---|
| 1 x V100 32GB | Primary baseline | Keep regression rows green for correctness, generation, cache, quant loading, and small training. |
| 2 x V100 32GB | ZeRO smoke recorded | Add ZeRO checkpoint-resume and larger-model ZeRO rows. |
| RTX 50-series / Blackwell | Some validation exists | Re-run current acceptance scripts and add decode/prefill/quant rows. |
| RTX 4090 / Ada | Needed | Add fp16/bf16 speed, memory, quant, and PEFT smoke rows. |
| A100 / Ampere | Needed | Add production-style batch-size sweeps and ZeRO rows. |
| H100 / Hopper | Needed | Add high-end throughput, bf16, quant, and large-model rows. |
| Pascal / older NVIDIA | Needed where feasible | Verify fallback behavior, fp16 constraints, and quant policy. |
| AMD / ROCm | Open | Start with native/no-FLA pure-PyTorch compatibility, then optional kernels. |
| CPU fallback | Partial / experimental | Keep no-CUDA import and tiny native tests green. |

When adding a card result, include at least:

- GPU name and count.
- Driver, CUDA/ROCm, PyTorch, Transformers, PEFT, TRL, and DeepSpeed versions.
- Model size and dtype.
- Command used.
- `bench/results.jsonl` rows when the command supports `--results`.
- A short note in `BENCHMARK.md` or the PR body.

## Production readiness checklist

| Requirement | Status | Next proof needed |
|---|---|---|
| Installable HF adapter | Mostly done | Clean fresh-env install docs and smoke command. |
| `from_pretrained` / `generate` | Done | Keep API tests green. |
| Official-weight conversion | Done | Keep all published-size conversion tests green. |
| Correctness vs official RWKV path | Done for 0.1B smoke | Add more model sizes and dtype/card combinations. |
| PEFT LoRA | Done for smoke | Add larger-model PEFT matrix and QLoRA training smoke. |
| SFT/DPO/GRPO | Done for smoke | Add larger-model matrix, longer steps, checkpoint save/load. |
| ZeRO-2/ZeRO-3 | Done for small smoke | Add checkpoint resume and larger-model rows. |
| `device_map` / PP direction | Partial | Add more models/cards and CPU offload smoke. |
| Quantized W8/W4 memory | Done for smoke | Keep memory gates recorded for each model/card. |
| Quantized W8/W4 speed | Open | Add fused/native W8/W4 path that is no slower than fp16. |
| Dynamic batch/cache helpers | Done for HF helpers | Keep correctness plus hit-rate telemetry gates. |
| Chunked prefill | Functional | Add production batch-size and long-context sweeps. |
| One-click acceptance | Open | Add scripts that run the HF acceptance matrix. |
| CI coverage | Partial | Add no-CUDA import/API tests and optional GPU workflows. |
| Cross-card matrix | Partial | Add A100/H100/4090/5090/Pascal/AMD evidence. |

## Current open gaps

1. **Large-model HF training matrix**: small-model smoke exists; production
   evidence still needs 0.4B / 1.5B / 2.9B / 7B PEFT, SFT, DPO/GRPO, and ZeRO
   rows where feasible.
2. **ZeRO checkpoint resume**: ZeRO-2/3 can train in smoke form, but resume from
   saved DeepSpeed checkpoints needs a dedicated test and recorded rows.
3. **Quantized speed**: W8/W4 loading and memory reduction are functional, but
   generic bnb paths are not yet fast enough. This needs fused/native quantized
   serving kernels.
4. **Albatross/RWKV-LM production performance**: HF fast paths are improving,
   but the repository still needs broader prefill/decode/batch evidence and
   higher speed ratios across cards.
5. **Card coverage**: V100 evidence is strong enough for development, not enough
   for production. More professional and consumer GPUs must be added.
6. **One-click reproducibility**: contributors need scripts that run acceptance,
   training-matrix, and hardware-matrix checks with consistent output.

## Where to work next

Use [`HF_TODO.md`](HF_TODO.md) for the prioritized contributor roadmap and
[`BENCHMARK.md`](BENCHMARK.md) for the current numeric evidence. Use
[`HF_CRITERIA.md`](HF_CRITERIA.md) as the high-level acceptance criteria and
[`FUSED_BACKEND.md`](FUSED_BACKEND.md) for the performance-kernel track.

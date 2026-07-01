# RWKV-7 HF Adapter TODO / Contributor Roadmap

This is the practical TODO list for contributors. It is intentionally **HF
adapter only**: Transformers loading/generation, Trainer, PEFT, TRL, DeepSpeed,
HF state-cache helpers, quantized HF inference, hardware validation, and
production-readiness evidence.

Do not put native vLLM/SGLang work in this TODO. Those are separate projects.

## Contribution rules

1. Keep changes scoped to the HF adapter unless the PR explicitly says it is a
   benchmark or documentation-only update.
2. Add or update tests for every behavior change.
3. Record reproducible evidence for GPU work in `bench/results.jsonl` when the
   command supports `--results`.
4. Hardware PRs must state card name, driver, CUDA/ROCm, PyTorch, dtype, model
   size, and exact command.
5. Do not regress the V100 baseline while optimizing for newer cards.
6. If a test is optional because a GPU/library is missing, make the skip explicit
   and keep CPU/no-CUDA import paths green.

## P0: close the HF acceptance evidence

### 1. Large-model training matrix

Small-model PEFT/Trainer/TRL smokes exist. The next acceptance step is to run
and record a model-size matrix.

| Model size | PEFT | SFT | DPO | GRPO | ZeRO-2 | ZeRO-3 | Notes |
|---|---|---|---|---|---|---|---|
| 0.4B | Needed | Needed | Needed | Needed | Needed | Needed | First real training target beyond 0.1B. |
| 1.5B | Needed | Needed | Needed | Optional | Needed | Needed | Good V100 stress target. |
| 2.9B | Needed | Needed | Optional | Optional | Needed | Needed | May require shorter sequence / grad accumulation tuning. |
| 7B | Optional | Optional | Optional | Optional | Optional | Needed | Tiny ZeRO-3 smoke is enough initially. |

Definition of done:

- finite loss;
- trainable parameters change;
- no silent NaN/Inf;
- command and model path recorded;
- result row appended to `bench/results.jsonl` when supported;
- summary added to `BENCHMARK.md` or PR body.

### 2. ZeRO checkpoint resume

Add a dedicated smoke test for DeepSpeed resume behavior:

1. initialize HF Trainer + PEFT LoRA under ZeRO-2;
2. train one step;
3. save checkpoint;
4. reinitialize model/trainer;
5. resume from checkpoint;
6. train one more step;
7. assert finite loss, expected global step, and trainable parameter delta;
8. repeat for ZeRO-3.

Suggested file:

- `tests/test_deepspeed_resume_smoke.py`

Suggested result type:

- `deepspeed_resume_smoke`

### 3. One-click HF acceptance scripts

Add scripts so a new contributor can reproduce the current acceptance state
without reading every test file.

Suggested scripts:

- `scripts/run_hf_acceptance.sh`
- `scripts/run_hf_training_matrix.sh`
- `scripts/run_zero_resume_smoke.sh`
- `scripts/run_hardware_smoke.sh`

Definition of done:

- scripts accept `MODEL`, `RESULTS`, `CUDA_VISIBLE_DEVICES`, and dtype-related
  overrides;
- scripts print environment metadata;
- scripts fail fast on real failures but allow explicit optional skips;
- docs show the minimal invocation.

### 4. Card adaptation matrix

Build a reproducible card matrix. The goal is production confidence across
common professional and consumer hardware, not only one server.

Minimum per-card smoke:

```bash
python tests/smoke_hf_generate.py --model /path/to/model
python tests/test_hf_api_contract.py --model /path/to/model
python tests/test_quantized_inference.py --model /path/to/model --device cuda
python bench/bench_speed.py --hf-dir /path/to/model --backend hf --dtype fp16 --device cuda --results bench/results.jsonl
python bench/bench_batch_sweep.py --hf-dir /path/to/model --dtype fp16 --device cuda --results bench/results.jsonl
```

Training-capable cards should also run:

```bash
python tests/test_peft_lora.py --model /path/to/model --device cuda --attn-mode fused_recurrent
python tests/test_hf_training_smoke.py --model /path/to/model --device cuda --attn-mode fused_recurrent --backend both --results bench/results.jsonl
python tests/test_hf_rl_training_smoke.py --model /path/to/model --device cuda --attn-mode fused_recurrent --backend dpo --results bench/results.jsonl
```

Multi-GPU cards/nodes should run:

```bash
torchrun --standalone --nproc_per_node=2 tests/test_deepspeed_training_smoke.py \
  --model /path/to/model \
  --zero-stage both \
  --train-dtype fp32 \
  --max-steps 1 \
  --batch-size 1 \
  --gradient-accumulation-steps 1 \
  --max-length 32 \
  --results bench/results.jsonl
```

Card targets:

| Priority | Card family | Goal |
|---|---|---|
| P0 | V100 1x/2x | Keep baseline green; add ZeRO resume and large-model smoke. |
| P0 | A100 | Add Ampere production throughput, bf16, quant, ZeRO rows. |
| P0 | RTX 4090 | Add common consumer Ada evidence. |
| P1 | H100 | Add Hopper high-end throughput and bf16/quant rows. |
| P1 | RTX 5090 / 50-series | Add Blackwell consumer validation and regression rows. |
| P1 | Pascal/Turing | Verify fallback behavior and older-card constraints. |
| P2 | AMD ROCm | Start native/no-FLA compatibility and document gaps. |
| P2 | CPU | Keep tiny native/no-FLA import and API tests working. |

## P1: productionize the HF user experience

### 5. Accelerate / `device_map` / offload

Needed work:

- `device_map="auto"` smoke;
- manual multi-GPU layer placement smoke on larger models;
- CPU offload smoke;
- clear docs for when fast-token shortcuts are disabled by sharding;
- examples for single-GPU, multi-GPU, and offload loading.

### 6. PEFT / QLoRA matrix

Needed work:

- document recommended LoRA target modules;
- verify adapter merge then `generate()`;
- add QLoRA 8-bit and 4-bit training smoke where bitsandbytes supports the card;
- record memory deltas for QLoRA loads.

### 7. TRL training hardening

Needed work:

- longer SFT/DPO/GRPO smoke runs;
- checkpoint save/load for each trainer type;
- clearer handling of fp16/bf16/fp32 training dtype behavior;
- small public toy dataset examples.

### 8. Hub and examples

Needed work:

- minimal inference example;
- minimal LoRA example;
- minimal SFT example;
- minimal DPO/GRPO examples;
- model-card notes explaining RWKV recurrent state cache vs Transformer KV cache;
- `trust_remote_code=True` loading notes and expected dependencies.

### 9. CI and packaging

Needed work:

- no-CUDA import test;
- CPU tiny-model API tests;
- conversion/config tests;
- optional GPU workflow for smoke benchmarks;
- dependency extras for training/quantization/dev docs.

## P2: close performance and quantization gaps

### 10. Albatross / RWKV-LM speed gap

Performance work should continue on the fast-token/native-graph route instead of
adding wrapper layers.

Current intended route:

```text
native_graph -> fused fp16 kernel -> fused W8/W4 kernel
```

Needed proof:

- prefill/decode/batch-size sweeps against the same checkpoint and same card;
- latency and peak-memory rows;
- cache hit-rate rows;
- clear ratio gates in `bench/analyze_results.py` / `bench/check_results.py`.

### 11. Quantized speed

Current status: W8/W4 loading and memory reduction work, but speed is not yet
production-complete.

Needed work:

- native packed W8/W4 weight layout;
- fused dequant + projection path;
- card-specific tuning for V100/A100/4090/H100/50-series;
- quality telemetry close to llama.cpp-style practical quantization levels;
- speed target: W8/W4 should be no slower than fp16 on common cards.

### 12. Training throughput

Needed work:

- compare HF Trainer/PEFT throughput with RWKV-LM training where possible;
- batch-size and sequence-length sweeps;
- activation/checkpointing memory rows;
- ZeRO-2/3 throughput and memory rows.

## P3: upstream and long-term compatibility

### 13. Native Transformers direction

Long-term upstream shape:

```text
src/transformers/models/rwkv7/
  configuration_rwkv7.py
  modeling_rwkv7.py
  tokenization_rwkv7.py
  convert_rwkv7_original_to_hf.py
```

Needed work:

- pure PyTorch/reference path without mandatory FLA;
- optional CUDA/Triton kernels;
- CPU and AMD compatibility story;
- Transformers model common tests;
- generation tests;
- tokenizer/model-card docs.

### 14. HF-compatible speculative decoding

Needed work:

- more draft/target size pairs;
- longer prompts and larger batches;
- acceptance-rate telemetry;
- correctness checks against target greedy output;
- documentation for when speculative decoding helps or hurts.

## PR checklist for contributors

Before opening a PR, include:

- [ ] What was changed and why.
- [ ] Exact command(s) run.
- [ ] Hardware and software versions for GPU work.
- [ ] Result rows or benchmark summary if applicable.
- [ ] Updated docs if behavior, support matrix, or TODO status changed.
- [ ] Confirmation that the change is HF-adapter scoped.

For documentation-only PRs, at minimum run:

```bash
git diff --check
```

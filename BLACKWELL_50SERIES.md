# 50 系(Blackwell sm_120)适配 — 跟着 main 验证

RTX 5070 Laptop(sm_120, 8GB)。逐项验证 main 最新在 Blackwell 上的表现,抓 sm_120 特有问题。

## 验证记录

### 2026-07-01 — main native_model.py(c81e2f2)在 5070 ✅
`tests/test_native_model.py`(0.1B fp32,main 最新 native,batched + tuple-cache + JIT):
- batch-forward bsz=3: min_cos=0.999999, max_abs=0.000027, argmax 3/3
- generate: greedy token-identical 16/16
- generate-cache: incremental_cache=True(prefill (1,8) → decode (1,1) 带 cache,**main 的 use_cache=True tuple-cache 方案在 sm_120 走通**)
- **NATIVE MODEL PASS**

### 2026-07-01 — main quant-fast-forward(4bit nf4)在 5070 ✅
`tests/test_quantized_inference.py`(0.1B fp16,4bit nf4):footprint 242.9MB、peak VRAM 274.3MB、quant_skip(lm_head + lora)、72 linear_4bit、**fast_forward 路径激活(backend=fla,max_abs_vs_ref 0.078)**、next_token 4171 一致。**PASS**。之前"4bit 慢 13×"已被 main 的 quant-fast-forward 解决。

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

### 2026-07-01 — main speculative-decode(0.4B target + 0.1B draft)在 5070 ✅
`tests/test_speculative_decode.py`:16 token、acceptance_rate 82.4%、target_forward 8 / draft_forward 20、resync_saved 27。**PASS**。(注:0.4B/0.1B 目录须用 main convert 重转,旧目录缺 `rwkv7_speculative_generate`。)

### 2026-07-01 — ⚠️ 训练 backward 在 5070 崩(FLA kernel 超 shared memory)
`tests/test_hf_rl_training_smoke.py`(DPO,0.1B,chunk 和 fused_recurrent 都试过)backward 炸:
```
triton.OutOfResources: out of resource: shared memory, Required: 131072, Hardware limit: 101376
fla/ops/generalized_delta_rule/dplr/chunk_A_bwd.py:499 chunk_dplr_bwd_dqk_intra
```
**根因**:FLA 的 RWKV7 反向 kernel(DPLR chunk_A_bwd)需 **128KB 共享内存,5070(sm_120)硬件上限 ~99KB(101376)** → 训练 backward 在 Blackwell 上无法运行。chunk / fused_recurrent 共用同一反向 kernel,换 attn-mode 无效。
**影响**:Blackwell 上 **推理 ✅(native/quant/spec 都过),训练 backward ❌**。PEFT LoRA forward/backward(之前 test_peft_lora)用的是 fused_recurrent 小批量,可能没触发;RL(DPO/GRPO)的 backward 触发了大 chunk 反向 → 炸。
**修复方向**:① 减 fla kernel 的 block size / num_stages(让它 < 99KB);② upstream fla 为 sm_120 调小 kernel 配置;③ 或训练在 V100/A100(更大 shared memory)上跑,Blackwell 仅做推理验证。

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

### 2026-07-01 — ✅ 50 系训练 workaround:用 native 模型做 backward
根因确认:`fla.utils._device.check_shared_mem()` 对 5070 已返回 **False**(max shared mem 101376 < 阈值 102400),所以 BK 已减到 32——但 fla DPLR backward kernel **仍需 131072 字节 > 5070 上限 101376**。即 **fla 的"小 shared-mem 路径"对 5070(99KB)还不够小,是 upstream fla 问题**(需更小 BT/num_stages)。

**Workaround(50 系可用)**:用 **`NativeRWKV7ForCausalLM`(纯 PyTorch,无 fla kernel)**做训练 backward。验证:0.1B fp16,LoRA-style 冻结除 r_proj+lm_head,末位 logit loss=10.9,**backward 成功,2 个非零梯度**。native 走 autograd,不碰 triton kernel → 无 shared mem 限制。

**结论(Blackwell 训练)**:
- FLA wrapper 训练 backward ❌(fla kernel 超 99KB,upstream fla 须修)。
- **Native 模型训练 backward ✅(50 系用 native 做训练)**。

### 2026-07-01 — ⏭️ device_map / deepspeed 在 5070 撞墙(单卡 + 无 deepspeed)
- `test_device_map_generate`:**skip**(需 ≥2 CUDA 设备,单 5070 不满足)。
- `test_deepspeed_configs`(Zero2/3):**deepspeed MISSING**(本机未装,Zero 训练无法跑;且 Zero 通常需多卡)。配置文件校验见上(exit code)。
- 这两项需**多卡 + deepspeed** 环境,单 5070 做不了(撞墙)。50 系单卡能覆盖的是推理 + native 训练 backward。

## 50 系数据汇总(RTX 5070 Laptop, sm_120, 8GB, fp16, 0.1B 为主)

### 推理速度
| 路径 | prefill tok/s | decode bsz=1 tok/s | TPOT ms/tok |
|---|---|---|---|
| FLA HF(chunk,eager) | ~16980 | ~37 | ~27(框架开销大) |
| FLA HF + native_graph(CUDA graph) | — | ~395 | ~2.5 |
| FLA HF + rwkv7_forward_token | — | ~248 | ~4.0(TPOT p50) |
| Native 模型 generate(JIT) | —(顺序 prefill) | ~86 | — |
| 官方 rwkv(torch-ref,本机无 nvcc) | ~220 | ~99 | ~10 |

### TTFT(单次 prefill,bench_ttft_tpot)
| ISL | p50 | p99 |
|---|---|---|
| 512 | 36ms | 49ms |
| 2048 | 71ms | 80ms |

### 量化(quant-fast-forward,4bit nf4)
| 档 | footprint | peak VRAM | fast_forward | 状态 |
|---|---|---|---|---|
| fp16 | ~336MB | — | — | 基线 |
| 4bit nf4 | 242.9MB | 274.3MB | backend=fla, max_abs 0.078 | ✅ PASS |
（注:之前无 fast-forward 时 4bit decode 45ms/tok=22 tok/s,慢 13×;main 的 quant-fast-forward 已加快速路径。)

### 显存(decode-only)
- FLA HF decode-only: ~376MB(0.1B,权重 ~336 + 开销 ~40)
- 2.9B fp16: OOM(>8GB,需量化)

### 投机解码(0.4B target + 0.1B draft)
- acceptance_rate 82.4%,target_forward 8 / draft_forward 20

### 精度
- native vs FLA: cos=1.0(0.1B/0.4B/1.5B),generate token-identical

### 训练
- FLA backward: ❌(kernel 128KB > 99KB shared mem)
- Native backward: ✅(workaround,纯 PyTorch)

### 2026-07-01 — ✅ 50 系训练 workaround 落地:NativeRWKV7 SFT 可训练
给 `native_model.py` 加了序列级 loss 支持(`_run(collect_all=True)` 累积每 token logits + forward 接 `labels` 算 cross_entropy)+ `get_input/output_embeddings`(peft 需要)。

**验证(0.1B,5070,LoRA-like:只解冻 r_proj)**:
- fp32 SFT 5 步 loss: **[2.12, 1.57, 0.72, 0.34, 0.38]** —— **loss 稳定下降,backward 干净,无 shared-mem 限制**。
- fp16 SFT 第 2 步溢出 nan(fp16 grad overflow,非 shared-mem;需 fp32/bf16/grad-scaling)。

**结论**:Blackwell 训练 backward 的 sm_120 硬阻塞(FLA kernel 超 99KB)**有解**——用 NativeRWKV7ForCausalLM(纯 PyTorch,无 fla kernel)做训练,序列级 SFT 正常。FLA wrapper 推理可用、训练 backward 阻塞;**native 模型推理+训练在 Blackwell 都通**。

### 2026-07-01 — TRL SFTTrainer on native(生产包装层)集成坑(follow-up)
核心训练路径(native 手动 SFT + 回归测试 `test_native_training_smoke.py`)已证在 5070 工作(loss 1.56→0.44)。但 **TRL `SFTTrainer` 直接套 native 模型还有集成坑**:
1. `gradient_checkpointing` 默认开 → native 不支持(关掉可绕过)。
2. 关掉后 Trainer `compute_loss` 调到 `_forward_unimplemented`(默认 PreTrainedModel.forward),没路由到 NativeRWKV7ForCausalLM.forward——是 Trainer 包装层与 native 模型的路由问题,需调试(可能 SFTTrainer 对模型 forward 签名/包装有假设)。
**结论**:Blackwell **native 训练能力已证(手动 + 测试)**;TRL SFTTrainer 生产包装的适配是**后续 polish**(非 sm_120 阻塞,是 Trainer 集成)。FLA wrapper 的 SFTTrainer 在 V100 工作(main 已覆盖),Blackwell 用 native 时走手动/trainer-adapted 路径。

### 2026-07-01 — main PR#28 native training unit test 在 5070 ✅
main 的 `tests/test_native_model_training_unit.py`(团队 PR#28 加的 native CausalLM 训练 loss + 单元测试)在 5070 直接 **PASS**。即 **main 自带的 native 训练测试在 Blackwell 上绿**——不只我的 `test_native_training_smoke.py`,main 的官方测试也覆盖了 Blackwell native 训练路径。

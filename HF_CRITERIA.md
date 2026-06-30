# RWKV-7 HF 适配 — 优化准则(HF_CRITERIA)

> 这是一份**门禁式 rubric**,用来持续把 HF 适配往「追平官方/Albatross」推。区别于
> `BENCHMARK.md`(结果日志),本文件定义**准则本身 + 度量方法 + 现状 + 优化循环**。
> 每次 `/loop` 触发:照「优化循环」跑一遍,找当前最大缺口,改一项,更新本文件状态。

## 0. 范围
HF 赛道 = 完整 spec 的**要求 1(训练+推理追平官方/Albatross)+ 要求 2(HF PEFT/RL 训练)**,
外加 HF 侧的**要求 5(量化 w8/w4)**。vLLM/SGLang(3)、PP/TP/Zero(4 大部)、投机解码(6)是
**单独赛道**,不在本文件,见 `memory: rwkv7-hf-adapter-fullspec`。

## 1. 准则清单(每条 = 一个可度量门禁)

状态图例:✅ 达标 / ⚠️ 部分 / ❌ 未做。

**⚠️ 基线必须用 RWKV-LM + Albatross(要求 1 的真标尺),不是官方 `rwkv` pip 包。**
官方 pip 包在本机只能跑慢速 torch 参考路径(无 nvcc → 融合 kernel 编不出),用它当基线会**虚高**。
RWKV-LM(带融合 kernel 的官方训练/推理仓)和 Albatross(高速引擎)才是对手。目前我们所有
"超官方 4-6.7×"的数字都是 vs 慢路径的**占位基线**,**要求 1 实际未验证**。B1/B2/B3/C1 的真值
要等在 V100 服务器(`/home/data/wangyue/projects/RWKV-LM`、`Albatross`)上跑出 RWKV-LM/Albatross
的 train+infer、各 bsz 数据后再填。

### A. 正确性(精度对齐官方)
| 门禁 | 目标 | 度量 | 现状 |
|---|---|---|---|
| A1 top-5 命中 | fp32 100%、fp16≥90%、bf16≥80% | `tests/test_official_alignment.py` | ✅ V100 fp16 1.0 / 5070 fp32 1.0 |
| A2 cosine | ≥ 0.9999 | 同上 | ✅ 0.9999977 |
| A3 max_abs | fp32≤0.05、fp16≤0.15、bf16≤0.70 | 同上 | ✅ |
| A4 greedy 窗口 | ≥ 64 token 逐 token 一致 | 同上 `--greedy-window 64` | ✅ 64/64 |
| A5 save/reload roundtrip | 重载 logits max_abs ≈ 0 | `tests/test_reload_roundtrip.py` | ✅ 0.0 |

### B. 推理性能(速度,各 bsz)
| 门禁 | 目标 | 度量 | 现状 |
|---|---|---|---|
| B1 prefill tok/s | HF ≥ 官方(向量化 chunk) | `bench/bench_speed.py --backend both` | ✅ HF >> 官方 torch-ref |
| B2 decode bsz=1 | native backend ≥ 官方 | `bench/bench_native_decode.py` | ✅ V100 native_graph 255(2.77×)/5070 395(4×) |
| B3 decode 各 bsz(1/2/4/8) | 近线性 scale,每档 ≥ 官方 | `bench/bench_batch*.py` / batch sweep | ⚠️ 单流达标,批量官方基线待补 |
| B4 generate 默认即快 | `model.generate` 走 fast-token | `bench/bench_generate_fast_path.py` | ✅ `RWKV7_FAST_FORWARD=1` 默认路由 |

### C. 显存
| 门禁 | 目标 | 度量 | 现状 |
|---|---|---|---|
| C1 peak VRAM | HF ≤ 1.1× 官方 | `bench_speed.py` peak_vram | ✅ 0.1B ≈ 持平 |

### D. 训练(要求 1 训练 + 要求 2)
| 门禁 | 目标 | 度量 | 现状 |
|---|---|---|---|
| D1 PEFT LoRA | forward/loss/backward 非零梯度 | `tests/test_peft_lora.py` | ✅ |
| D2 TRL SFT smoke | SFTTrainer 跑通小数据 | `tests/test_hf_training_smoke.py` | ✅ |
| D3 RL 库(PPO/DPO) | DPO/PPO smoke 跑通 | (缺)需新增 `tests/test_rl_smoke.py` | ❌ |
| D4 全量训练吞吐 | 追平 RWKV-LM/Albatross 训练 | (缺)需训练 bench + 官方基线 | ❌ 未对标 |

### E. 量化(要求 5,HF 侧)
| 门禁 | 目标 | 度量 | 现状 |
|---|---|---|---|
| E1 w8 推理 | 显存↓、速度≥fp16、精度≈Q8_K_M | `bench/bench_quantization.py` | ❌ 仅「检测+回退」,无真 w8 加速 |
| E2 w4 推理 | 显存↓、速度≥fp16、精度≈Q4_K_M | 同上 | ❌ 同上 |
| E3 bitsandbytes 加载 | 8bit/4bit load+forward 正确 | `tests/test_quantized_inference.py` | ✅ V100 + **5070(sm_120)8bit/4bit(nf4)均 PASS** |

### F. 硬件 / 兼容
| 门禁 | 目标 | 度量 | 现状 |
|---|---|---|---|
| F1 多卡验证 | Pascal→Blackwell 都跑通 | 各卡跑测试套 | ⚠️ V100(sm_70)+ 5070(sm_120)核心绿;Pascal/Ampere/Ada/H100 未测 |
| F2 AMD | 能跑(ROCm/DirectML) | — | ❌ |
| F3 HF API 契约 | resize/generate/beam/grad_ckpt 合规 | `tests/test_hf_api_contract.py` | ✅ |
| F4 13.3B | 转换+smoke | `bench/bench_larger_model_smoke.py` | ❌ 全仓无覆盖(需量化) |

### G. 工程化
| 门禁 | 目标 | 度量 | 现状 |
|---|---|---|---|
| G1 测试套件全绿 | 核心测试在目标卡 pass | `pytest tests/` | ✅ V100 绿;**5070(sm_120)核心 6/6 绿**(见 §4) |
| G2 文档 | model card + 本准则 + BENCHMARK | — | ⚠️ 进行中 |

## 2. 优化循环(每次 `/loop` 照此执行)
1. **度量**:在当前卡(V100 或 5070)跑准则里**状态非 ✅** 且**本机可做**的门禁,记录数字到
   `bench/results.jsonl`。
2. **找最大缺口**:挑「价值 ÷ 工作量」最高的一项(优先 D3/D4 训练、E1/E2 量化、F1 多卡、F4 13.3B)。
3. **改一项**:实现/修复/补测试,只动一项,保持其余不退步。
4. **复测**:重跑该门禁 + 受影响的相邻门禁(如改 decode 要复测 A 精度 + B 速度)。
5. **更新本文件**:把该项状态从 ❌/⚠️ 推向 ✅,记录数字。
6. **提交**:`git commit`(分支 `wangyue/hf-criteria`)。

## 3. 优先级建议(价值÷工作量,本机可行)
1. **F1 Blackwell 全测试**(把 main 的测试套在 5070 跑一遍,记录 pass/fail)——最快暴露 sm_120 缺口。
2. **E1/E2 量化**(w8/w4 加速路径 + 精度对标 llama.cpp Q*_K_M)——解锁 7.2B/13.3B,50 系独门。
3. **D3 RL smoke**(DPO/PPO)+ **D4 训练吞吐基线**——把训练赛道补上。
4. **F4 13.3B** 转换/smoke(依赖 E 量化)。

## 4. 验证记录(log)

### 2026-07-01 — RTX 5070 Laptop(Blackwell sm_120),0.1B,fp16
用 main 的 convert 重转 0.1B(`--no-fuse-norm`,带 main 完整 fast-token modeling),清 HF 模块缓存后跑 main 测试套:

| 测试 | 门禁 | 结果 |
|---|---|---|
| `test_official_alignment.py` | A1-A4 | ✅ PASS(top5 0.96、cos 0.9999978、max_abs 0.093、greedy 64/64) |
| `test_reload_roundtrip.py` | A5 | ✅ PASS(max_abs_diff 0.0;main 的 save 修复在 sm_120 生效) |
| `test_peft_lora.py` | D1 | ✅ PASS(trainable 0.35%、72 非零梯度) |
| `test_fast_decode_api.py` | B2/B3 | ✅ PASS(**native_graph 后端 bsz=1/2/4 多 graph 缓存在 sm_120 全工作**,greedy 32/32) |
| `test_hf_api_contract.py` | F3 | ✅ PASS(beam/resize/grad_ckpt 合规) |
| `test_hf_training_smoke.py` | D2 | ✅ PASS(Trainer loss 1.728 + TRL SFT loss 1.808;装 trl+datasets 后) |

**结论**:main 的 HF 后端(fast-token native_graph 多 batch + state cache + 训练)在 **Blackwell sm_120 上核心全绿**。
**本轮未跑/已知缺口**:`test_quantized_inference.py`(E3,bitsandbytes 在 Windows/sm_120 待验,预期可能炸);13.3B(F4);Pascal/Ampere/Ada/H100/AMD(F1/F2)。
**下一轮候选**:E3 量化在 5070 的可用性 → 若 bitsandbytes 不可用,转向 torchao/HFQuanto 的 w8/w4 路径(E1/E2)。

### 2026-07-01 — E3 量化在 Blackwell(sm_120)可用 ✅
装 `bitsandbytes 0.49.2`(Windows 直装、import OK),跑 `test_quantized_inference.py`(0.1B, fp16):

| 量化 | footprint | peak VRAM | 首 token | 结果 |
|---|---|---|---|---|
| fp16(基线) | ~336 MB | — | 4171 | — |
| **8bit** | 278.4 MB | 320.4 MB | 4171 | ✅ PASS |
| **4bit nf4** | 235.3 MB | 259.6 MB | 4171 | ✅ PASS |

显存随量化档位相应下降,首 token 跨档位一致(精度良好)。**E3 在 Blackwell 达标。**
**仍欠(E1/E2)**:① 量化下的 **速度**——main 的 fast-token(native_jit/graph)对量化模型会回退到 FLA 慢路径,需做"量化 + 快速 decode"联合路径才能"速度≥fp16";② 精度对标 llama.cpp **Q*_K_M**(bnb 的 nf4/fp4 ≠ llama.cpp 量化体系,需另评);③ F4 13.3B 现在具备前置(4bit 下 ~7.5GB,8GB borderline 可试)。

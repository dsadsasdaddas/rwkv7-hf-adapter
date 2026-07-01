# RWKV-7 HF 适配 — 优化准则(HF_CRITERIA)

> 这是一份**门禁式 rubric**,用来持续把 HF 适配往「追平官方/Albatross」推。区别于
> `BENCHMARK.md`(结果日志),本文件定义**准则本身 + 度量方法 + 现状 + 优化循环**。
> 每次 `/loop` 触发:照「优化循环」跑一遍,找当前最大缺口,改一项,更新本文件状态。

## 0. 范围
HF 赛道 = 完整 spec 的**要求 1(训练+推理追平官方/Albatross)+ 要求 2(HF PEFT/RL 训练)**,
外加 HF 侧的**要求 5(量化 w8/w4)**。vLLM/SGLang(3)、PP/TP/Zero(4 大部)、投机解码(6)是
**单独赛道**,不在本文件,见 `memory: rwkv7-hf-adapter-fullspec`。

**HF 的边界(明确)**:HF adapter 是"模型实现",不是服务引擎。**只对单次 `model.generate` 调用
的延迟/吞吐、batch generate、训练兼容性负责**。**多用户并发、SLO、最大并发、continuous batching
属于 vLLM/SGLang(req-3),HF 不背这些 KPI**(即使 main 里 `RWKV7StateCache` 有服务级原件,
那是为 req-3 预留,HF 不测并发)。

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

### B. 推理性能(HF 范围:单次调用 + batch generate)

> **HF adapter 是"模型实现",不是服务引擎。** 性能 KPI = **单次 `model.generate` 调用**
> 的延迟和吞吐,以及**一次调用喂 B 条序列**的 batch generate。**多用户并发/SLO/最大并发/
> continuous batching 属于 vLLM/SGLang(req-3),不在 HF 范围**(main 的 `RWKV7StateCache`
> 服务级原件是为将来 req-3 复用,但 HF 不背并发 KPI)。

**度量标准(以后 HF 性能数字都按这个)**:拆 **TTFT**(单次 prefill 首 token 延迟)+
**TPOT**(单次 decode 每 token 延迟),报 **p50/p99**(重复多次);在**固定 ISL/OSL** 下。
- 长度 profile:smoke 用 (ISL=512, OSL=128);补数时跑 ISL∈{128,512,2048}。

| 门禁 | 目标 | 度量 | 现状 |
|---|---|---|---|
| B1 TTFT(单次 prefill 延迟) | p99 ≤ 1.1× RWKV-LM/Albatross @ 同 ISL | `bench/bench_ttft_tpot.py` | ⚠️→有数:5070 p50 36-71ms / p99 49-80ms(ISL 512-2048)。vs RWKV-LM 未同卡(本地无 nvcc) |
| B2 TPOT(单次 decode/token) | p50 ≤ 1.1× RWKV-LM/Albatross | `bench/bench_ttft_tpot.py` | ⚠️→有数:5070 HF TPOT p50 **4.0ms**/p99 5.2ms(248 tok/s)。**HF forward 比 raw native_graph(2.5ms=395 tok/s)慢 ~60%**——HF 框架 per-call 开销在小模型显著,是 HF adapter 可优化点 |
| B3 batch generate 吞吐 | `model.generate(batch_size=B)` 近线性、≥ 官方 | `model.generate(batch=B)` 计时 | ⚠️→有数:0.1B 5070 B=1/4/8 = 168/736/**1144 tok/s**(近线性 scale;B=1 偏慢是 generate 框架开销,批量化摊薄) |
| B4 generate 默认即快 | `model.generate` 走 fast-token | `bench/bench_generate_fast_path.py` | ✅ `RWKV7_FAST_FORWARD=1` 默认路由 |

**关键缺口**:main 的 bench 报的是固定长度固定 batch 的聚合 tok/s + ms/tok(smoke)。HF 该补的是
**单次调用的 TTFT/TPOT(p50/p99)** 和 **batch generate** 的正经数(对齐 RWKV-LM/Albatross)。
并发/SLO 不在范围(req-3)。

### C. 显存
| 门禁 | 目标 | 度量 | 现状 |
|---|---|---|---|
| C1 peak VRAM | HF ≤ 1.1× 官方 | `bench_speed.py` peak_vram | ✅ 0.1B ≈ 持平 |

### D. 训练(要求 1 训练 + 要求 2)
| 门禁 | 目标 | 度量 | 现状 |
|---|---|---|---|
| D1 PEFT LoRA | forward/loss/backward 非零梯度 | `tests/test_peft_lora.py` | ✅ |
| D2 TRL SFT smoke | SFTTrainer 跑通小数据 | `tests/test_hf_training_smoke.py` | ✅ |
| D3 RL 库(DPO) | DPO smoke 跑通 | `tests/test_rl_smoke.py` | ✅ TRL 1.7 DPOTrainer 跑通(0.1B+LoRA,train_loss 0.66;Blackwell sm_120) |
| D4 全量训练吞吐 | 追平 RWKV-LM/Albatross 训练 | (缺)需训练 bench + 官方基线 | ❌ 未对标 |

### E. 量化(要求 5,HF 侧)
| 门禁 | 目标 | 度量 | 现状 |
|---|---|---|---|
| E1/E2 量化 decode | 显存↓、速度≥fp16(req-5) | 单 token decode TPOT | ❌ **量化比 fp16 慢 ~13×**:4bit nf4 TPOT **45ms(22 tok/s)** vs fp16 3.54ms(283 tok/s)。根因:fast-token 拒绝量化→FLA 回退(无快 decode)+ bnb 4bit 每 op 反量化。要达标需「量化版 native_graph runner」。附:1D 输入量化路径崩溃**已修**(forward 把 [batch]→[batch,1],fast/FLA 两路径都安全)|
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

### 2026-07-01 — 第一个真基线:RWKV-LM 融合 kernel(V100)
SSH 到 wzu 盒子,`rwkv7-cu118` env + `RWKV_CUDA_ON=1`(融合 WKV7 kernel,编译过 sm_70),0.1B decode bsz=1:

| 路径(V100, 0.1B, decode bsz=1) | tok/s |
|---|---|
| **RWKV-LM 融合 kernel(真基线)** | **97.9** |
| 我们 HF native_graph(main) | 255.5 = **2.6× 真基线** ✅ |

**重要修正**:main BENCHMARK 里的"official 92.1"**就是融合 kernel**(≈ 我测的 97.9),**不是慢 torch-ref**。所以 main 的 decode 领先是**真实**的,不是我先前担心的虚高。(我本地 5070 的"official 99"才是 torch-ref,数字巧合接近。)
**仍欠(要求 1 的其余)**:① **Albatross** 基线还没跑;② **训练吞吐**基线(RWKV-LM/Albatross 的 train tok/s)没量——这是我们最可能输的;③ **各 bsz(2/4/8)** + **更大模型(0.4B-7.2B)** 的真基线没跑全。decode-bsz1-0.1B 只是我们最强的 launch-bound 区间。

### 2026-07-01 — Albatross / 训练基线摸底(服务器)
- **Albatross**:高速推理引擎(多版本 faster2/3/3a/3b/4 + benchmark.py)。README 给的是 **5090**(Blackwell,**和 5070 同代**)数:decode B1T1 ≈ **144 tok/s**(单 token ~6.9ms 固定开销),prefill B1T1024 17000+ tps,batch decode B32T32 21000+ tps。**对标它要在 V100 上跑它的 benchmark.py**(用我们的 0.1B 模型)——下轮做。
  - 初步对照:我们 HF native_graph 在 **V100** decode bsz=1 = 255 tok/s;Albatross 在 5090 decode bsz=1 ≈ 144 tok/s。**跨卡不可直接比**(V100≠5090),但量级上我们在 decode bsz=1 不输甚至领先——不过 Albatross 强在 **batch + prefill**(B32T32 21000+ tps,我们没这种量级的批量快路径)。
- **RWKV-LM 训练**:仓库里只有 `train_temp/rwkv7_train_simplified.py`(自述"slow & different results",2 层 demo)——**不是全量训练,不代表真实吞吐**。真·训练基线要全套生产训练环境,本阶段难取。**训练侧 req-1 暂时无法诚实验证**(我们 vs RWKV-LM 全量训练的 fwd+bwd tok/s)。

### 2026-07-01 — Albatross 同卡基线:环境阻塞
试着在 V100 跑 Albatross `faster2_251201/benchmark.py`(指到 0.1B)三次,均卡在环境:① import 路径(已修,脚本须在 faster2 目录内跑);② `rwkv7` env 无 CUDA_HOME;③ **`rwkv7-cu118` env 的 CUDA toolkit 不完整(`cusparse.h` 缺失),torch.compile 编 Albatross 的扩展失败**。→ **Albatross 同卡基线在现服务器环境跑不通**,只能拿 README 的 5090 数(decode B1T1 ~144、batch B32T32 21k+)作量级参考。
**结论:req-1 余下的真基线(Albatross 同卡、训练吞吐)都因环境/代表性问题暂时取不到。**decode bsz=1 已用 RWKV-LM 真基线验证(2.6× 领先);batch+prefill + 训练侧的差距只能定性(Albatross 强在 batch+prefill,我们无对等快路径;训练侧未验证)。
**可推进的本机项**:E1/E2(量化+快速 decode 联合)、13.3B(但 .pth ~26GB 下载阻塞 + 4bit borderline)。

### 2026-07-01 — 服务级 state-cache 测试在 Blackwell(2 过 1 真失败)
跑 main 的 state-cache 三件套在 5070(sm_120, 0.1B, fp16, fuse_norm=false):

| 测试 | 结果 | 说明 |
|---|---|---|
| `test_batch_cache.py` | ✅ PASS | select_batch 批量 cache 操作,max_abs 0.0 |
| `test_chunked_prefill.py` | ✅ PASS | 分块 prefill,max_abs 0.046、decode 0.0 |
| `test_dynamic_batch_cache.py` | ❌ **FAIL** | **fast_token heterogeneous decode step=1 logit 差 31.69** |

**这是个真·Blackwell bug**:`RWKV7_FAST_TOKEN` 后端(native_jit/graph)在**异构批量**(同一 batch 里不同长度序列 + padding)解码时**结果错**(差 31.69,远超容差)。批量同构(batch_cache)正常 → 问题在**异构/padding 的 fast-token 路径**。
**下一步**:定位 `_RWKV7NativeGraphBatchedTokenRunner` / fast-token 异构分支在 sm_120 的数值问题(padding 未正确 mask?index 偏移?),是 `不断优化` 的具体目标。

### 2026-07-01 — dynamic_batch_cache bug 精确定位 + workaround
复跑 `test_dynamic_batch_cache.py` 强制后端:
- `RWKV7_FAST_TOKEN_BACKEND=native_jit` → **PASS**(diff 0.06-0.15)。
- 默认(auto→`native_graph` 批量)→ **FAIL**(异构 decode step=1 diff 31.69)。

**结论:bug 只在 `_RWKV7NativeGraphBatchedTokenRunner`(native_graph 批量 graph runner)**。单流 native_graph(`rwkv7_forward_one` bsz=1)、native_jit 批量(`block_step_batched`)都正确。即:**CUDA-graph 批量 runner 的 cache 绑定/捕获**(`copy_from_cache`/`bind_cache` 或 graph capture 处理 [B,H,N,N] 批量布局)在异构/动态批量下有数值 bug。
**Workaround**:`RWKV7_FAST_TOKEN_BACKEND=native_jit`(批量场景用 JIT 而非 graph,正确,略慢)。或固定 `native_graph` 仅用于单流 bsz=1。
**真·根因**(copy_from_cache/bind_cache 的批量布局 vs 单流 squeeze 差异)需运行期调试,留作下一步。

### 2026-07-01 — 批量 graph bug 深化:数学一致 → bug 在 graph 管道
静态对比 `_block_ip`(单流,工作)与 `_block_ip_batched`(批量,native_graph 用的):两者**数学逐行一致**(`vk=v@kᵀ`、`state@ab`、`state@r`、`state.copy_(new_state)` 原地更新、`xpa/xpf/v_first` 也都 `.copy_` 原地),只差一个 B leading 维。state 布局约定也相同。
→ **bug 不在数学,而在 `_RWKV7NativeGraphBatchedTokenRunner` 的 graph 捕获/绑定管道**:批量缓冲 [B,H,N,N] 的 copy_from_cache/bind_cache,或 B>1 时 warmup stream → capture 的内存池/地址处理。单流(B=1)同套管道没事 → 怀疑点多在 **B 维相关的地址/绑定**。
**下一步(运行期调试)**:同一 batched_state 上,分别用 native_graph 批量 vs native_jit 批量各跑 1 步,dump 中间 state/logits 逐层比,定位 copy_from_cache 还是 capture 哪步开始发散。Workaround 仍是 native_jit。

### 2026-07-01 — 批量 graph bug 根因 100% 确认:view 别名
运行期调试(同 batched state: native_graph 批量 vs native_jit 批量 → logits 差 0.0625、state 差 0.0,**批量 runner 本身正确**);再验证单流 runner 复用:`held lg0 changed after second decode: True`、`lg0==lg1: True`。
**根因**:`_RWKV7NativeGraph*TokenRunner.replay` 返回 `self.logits.view(...)`(共享 buffer 视图),`bind_cache` 把 cache 的 recurrent/conv/ffn state 也绑成 buffer 视图。当**同一 runner 被复用于多个独立输入**(测试的逐行解码、或任何跨调用持有结果的 caller),视图别名 → 后续 replay 覆盖前面持有的 logits/state。`test_dynamic_batch_cache` 的逐行循环正好触发(4 行复用单流 runner → 收集的 logits 全变最后一行)。native_jit 每次 eager 产生新张量,无别名 → 通过。
**修复选项**(设计权衡):
- (A) runner 返回 `.clone()` 的 logits + `bind_cache` 用 `.contiguous()` 拷贝(非视图)→ 每次 replay 后 cache 独立,任何复用都安全。代价:每步多一次 state in/out 拷贝(单流 ~MB 级,可接受),且失去 copy_from_cache 的 no-op 快路径。
- (B) 文档化"视图契约":caller 必须**立即消费** logits(argmax)再下一次调用;动态批量用**每请求独立 runner**(不要复用单流 runner 跨行)。改测试 clone。
- (C) 让 runner 缓存按 (batch_size, cache_id) 而非仅 batch_size,每 cache 独立 runner。
**当前**:workaround `RWKV7_FAST_TOKEN_BACKEND=native_jit`(批量/动态场景)。真修需要团队定 A/B/C(涉及性能权衡)。

### 2026-07-01 — dynamic_batch_cache bug 已修复(方案 A)✅
应用方案 A:`_RWKV7NativeGraph*TokenRunner.replay` 返回 `self.logits.view(...).clone()`、`bind_cache` 用 `.contiguous()`/`.clone()`(拷贝非视图)。结果(runner 跨输入复用时每次独立,无别名):

| 测试(5070 sm_120, native_graph) | 修复前 | 修复后 |
|---|---|---|
| `test_dynamic_batch_cache.py` | ❌ FAIL(31.69) | ✅ PASS(step1 0.0625,全步 0.06-0.09) |
| `test_fast_decode_api.py`(bsz=1/2/4) | ✅ | ✅(greedy 16/16,无回归) |
| `test_batch_cache.py`(select_batch) | ✅ | ✅(max_abs 0.0,无回归) |

**代价**:失去 `copy_from_cache` 的 no-op 快路径(每步多一次 state in/out 拷贝,~MB 级,decode 影响很小)。**Blackwell 上 state-cache 三件套现 3/3 全绿。**

#!/usr/bin/env python3
# coding=utf-8
"""End-to-end native int8 fused-RKV decode probe (Blackwell sm_120).

`bench_native_quant_rkv_sweep.py` proves the *isolated* int8 fused RKV kernel
beats fp16 (1.07x @0.1B, 2.36x @1.5B). That is a kernel microbench, not a full
model forward. This script answers the end-to-end question: with the 3 R/K/V
linears of every attention layer swapped for the int8 fused dequant-GEMV
kernel, how much does real per-token decode move on the 5070?

It loads NativeRWKV7ForCausalLM (pure-PyTorch path, use_jit=False, which routes
through native.attn_step_batched), quantizes each layer's r/k/v proj, monkey-
patches attn_step_batched to use int8_fused_rkv_gemv, and times a full
per-token forward over a real sequence (embedding -> all layers -> norm ->
lm_head). Correctness is checked vs the fp16 forward (cosine on final logits).

Honest scope: this is the pure-Python decode path, NOT the native_graph path.
The graph path fuses launches so the int8 launch-count advantage is smaller
there; this probe is an upper-bound-ish read on the RKV-quant end-to-end win.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("RWKV_V7_ON", "1")

import torch
import torch.nn.functional as F

import rwkv7_hf.native as _native
from rwkv7_hf import native as _nat
from rwkv7_hf.native import EXP_HALF, _init_state_batched
from rwkv7_hf.native_model import NativeRWKV7ForCausalLM
from rwkv7_hf.native_quant import (
    int8_fused_rkv_gemv,
    int4_fused_rkv_gemv,
    native_int8_fused_rkv_available,
    quantize_int8_rowwise,
    quantize_int4_rowwise,
)
from transformers import AutoTokenizer

_FALSE = {"0", "false", "False", "no", "off"}


def _quantize_layers(model, quant: str):
    packs = {"w8": quantize_int8_rowwise, "w4": quantize_int4_rowwise}
    pack = packs[quant]
    n = 0
    for layer in model.model.layers:
        attn = layer.attn
        dev = attn.r_proj.weight.device
        qr, sr = pack(attn.r_proj.weight.detach())
        qk, sk = pack(attn.k_proj.weight.detach())
        qv, sv = pack(attn.v_proj.weight.detach())
        attn._qpack_rkv = {
            "qr": qr.to(dev), "qk": qk.to(dev), "qv": qv.to(dev),
            "sr": sr.to(dev), "sk": sk.to(dev), "sv": sv.to(dev),
        }
        n += 1
    return n


def _make_patched_attn_step(kernel):
    """Copy of native.attn_step_batched using the int8/int4 fused RKV kernel."""

    def _patched(layer, layer_id, x, x_prev, v_first, state):
        B = int(x.shape[0])
        H, N = layer.num_heads, layer.head_dim
        hidden = H * N
        xx = x_prev - x
        xr = x + xx * layer.x_r.reshape(1, hidden)
        xw = x + xx * layer.x_w.reshape(1, hidden)
        xk = x + xx * layer.x_k.reshape(1, hidden)
        xv = x + xx * layer.x_v.reshape(1, hidden)
        xa = x + xx * layer.x_a.reshape(1, hidden)
        xg = x + xx * layer.x_g.reshape(1, hidden)
        qp = getattr(layer, "_qpack_rkv", None)
        if qp is not None:
            r, k, v = kernel(
                xr, xk, xv, qp["qr"], qp["qk"], qp["qv"],
                qp["sr"], qp["sk"], qp["sv"], block_m=16, block_k=32,
            )
        else:
            r = F.linear(xr, layer.r_proj.weight)
            k = F.linear(xk, layer.k_proj.weight)
            v = F.linear(xv, layer.v_proj.weight)
        w = F.linear(torch.tanh(F.linear(xw, layer.w_lora.lora[0].weight)),
                     layer.w_lora.lora[2].weight, layer.w_lora.lora[2].bias)
        a = torch.sigmoid(layer.a_lora.lora[2].bias +
                          F.linear(F.linear(xa, layer.a_lora.lora[0].weight),
                                   layer.a_lora.lora[2].weight))
        g = F.linear(torch.sigmoid(F.linear(xg, layer.g_lora.lora[0].weight)),
                     layer.g_lora.lora[2].weight)
        kk = F.normalize((k * layer.k_k.reshape(1, hidden)).view(B, H, N), dim=-1, p=2).view(B, hidden)
        k = k * (1 + (a - 1) * layer.k_a.reshape(1, hidden))
        if layer_id == 0:
            v_first = v
        else:
            v = v + (v_first - v) * torch.sigmoid(
                layer.v_lora.lora[2].bias +
                F.linear(F.linear(xv, layer.v_lora.lora[0].weight),
                         layer.v_lora.lora[2].weight))
        w = torch.exp(-EXP_HALF * torch.sigmoid(w.float()))
        vk = v.view(B, H, N, 1) @ k.view(B, H, 1, N)
        ab = (-kk).view(B, H, N, 1) @ (kk * a).view(B, H, 1, N)
        state = state * w.view(B, H, 1, N) + state @ ab.float() + vk.float()
        out = state.to(x.dtype) @ r.view(B, H, N, 1)
        out = out.view(B, hidden)
        out = F.group_norm(out, num_groups=H, weight=layer.g_norm.weight,
                           bias=layer.g_norm.bias, eps=N * 1e-5)
        sk = (r.view(B, H, N) * k.view(B, H, N) * layer.r_k.reshape(1, H, N)).sum(dim=-1, keepdim=True)
        out = out + (sk * v.view(B, H, N)).view(B, hidden)
        out = F.linear(out * g, layer.o_proj.weight)
        return out, x, state, v_first

    return _patched


def _time_forward(model, ids, *, warmup: int, iters: int) -> tuple[float, torch.Tensor]:
    """Time a full per-token forward (use_jit=False -> attn_step_batched path)."""
    base = model.model
    logits_ref = None
    with torch.inference_mode():
        for _ in range(warmup):
            state, xpa, xpf, vf = _init_state_batched(model, ids.shape[0], ids.device, base.embeddings.weight.dtype)
            lg, *_ = model._run(ids, state, xpa, xpf, vf, use_jit=False)
            logits_ref = lg[:, -1:].float().clone()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            state, xpa, xpf, vf = _init_state_batched(model, ids.shape[0], ids.device, base.embeddings.weight.dtype)
            lg, *_ = model._run(ids, state, xpa, xpf, vf, use_jit=False)
            last = lg[:, -1:]
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) / iters
    tokps = (ids.shape[1] * ids.shape[0]) / dt
    return tokps, dt, logits_ref, last.float().clone()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-dir", required=True)
    ap.add_argument("--dtype", default="fp16", choices=["fp16", "fp32"])
    ap.add_argument("--quant", default="w8", choices=["w8", "w4"])
    ap.add_argument("--seq-len", type=int, default=128)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--iters", type=int, default=8)
    ap.add_argument("--prompt", default="The quick brown fox jumps over the lazy dog. ")
    ap.add_argument("--results", default=str(Path(__file__).parent / "results.jsonl"))
    args = ap.parse_args()

    dt = torch.float16 if args.dtype == "fp16" else torch.float32
    print(f"device: {torch.cuda.get_device_name(0)} cap {torch.cuda.get_device_capability()}")
    avail = native_int8_fused_rkv_available() if args.quant == "w8" else None
    print(f"int8 fused RKV kernel available: {avail}")

    tok = AutoTokenizer.from_pretrained(args.hf_dir, trust_remote_code=True)
    model = NativeRWKV7ForCausalLM.from_pretrained(
        args.hf_dir, torch_dtype=dt, device_map="cuda").eval()

    # build input ids of desired length
    text = args.prompt * max(1, args.seq_len // 8 + 1)
    ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids[:, :args.seq_len]
    ids = ids.repeat(args.batch_size, 1).to("cuda")
    print(f"input: batch={ids.shape[0]} seq={ids.shape[1]} hidden={model.config.hidden_size}")

    # --- fp16 baseline ---
    torch.cuda.reset_peak_memory_stats()
    fp_tokps, fp_dt, ref_logits, _ = _time_forward(model, ids, warmup=args.warmup, iters=args.iters)
    print(f"[fp16 ] {fp_tokps:.1f} tok/s  ({fp_dt*1000:.2f} ms/forward)")

    # --- quantize + patch ---
    n = _quantize_layers(model, args.quant)
    kernel = int8_fused_rkv_gemv if args.quant == "w8" else int4_fused_rkv_gemv
    orig = _native.attn_step_batched
    _native.attn_step_batched = _make_patched_attn_step(kernel)
    try:
        q_tokps, q_dt, _, q_last = _time_forward(model, ids, warmup=args.warmup, iters=args.iters)
        peak = torch.cuda.max_memory_allocated() / 1024 / 1024
        print(f"[{args.quant}  ] {q_tokps:.1f} tok/s  ({q_dt*1000:.2f} ms/forward)")
        cos = torch.nn.functional.cosine_similarity(
            ref_logits.flatten().unsqueeze(0), q_last.flatten().unsqueeze(0)).item()
        max_abs = (ref_logits - q_last).abs().max().item()
        argmax_match = int(ref_logits.argmax() == q_last.argmax())
    finally:
        _native.attn_step_batched = orig

    speedup = fp_dt / q_dt
    print(f"\n=== end-to-end (pure-Python decode path) ===")
    print(f"fp16 {fp_tokps:.1f} tok/s -> {args.quant} {q_tokps:.1f} tok/s | speedup {speedup:.3f}x")
    print(f"correctness vs fp16: cos={cos:.4f} max_abs={max_abs:.4f} argmax_match={argmax_match}")

    row = {
        "axis": "native_quant_e2e_decode",
        "device": torch.cuda.get_device_name(0),
        "compute_cap": list(torch.cuda.get_device_capability()),
        "hf_dir": args.hf_dir,
        "dtype": args.dtype,
        "quant": args.quant,
        "hidden": model.config.hidden_size,
        "batch": ids.shape[0],
        "seq": ids.shape[1],
        "fp16_tokps": round(fp_tokps, 2),
        "quant_tokps": round(q_tokps, 2),
        "speedup_vs_fp16": round(speedup, 4),
        "cos_vs_fp16": round(cos, 4),
        "max_abs_vs_fp16": round(max_abs, 4),
        "argmax_match": argmax_match,
        "peak_vram_mb": round(peak, 1),
        "path": "native_pure_pytorch_attn_step_batched",
        "note": "end-to-end per-token forward; pure-Python path (not native_graph). "
                "int8 replaces 3 RKV linears with 1 fused dequant-GEMV kernel.",
    }
    print(json.dumps(row, indent=2, ensure_ascii=False))
    if args.results:
        out = Path(args.results)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"\nappended -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

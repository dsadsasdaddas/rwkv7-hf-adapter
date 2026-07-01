#!/usr/bin/env python3
# coding=utf-8
"""HF single-call TTFT/TPOT bench (HF scope — no concurrency/SLO).

TTFT = one model() prefill call latency (ms), per input length.
TPOT = one single-token decode step latency (ms/token), bsz=1, threading state.
Both report p50/p99 over repeated calls. This is the HF-relevant latency view;
multi-user concurrency/SLO is req-3 (vLLM/SGLang), out of scope here.

Usage: python bench/bench_ttft_tpot.py --model <hf_dir> [--isl 128 512 2048] [--reps 50]
"""
from __future__ import annotations

import argparse
import os
import time

os.environ.setdefault("RWKV_V7_ON", "1")
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}


def pct(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(len(xs) * p))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--dtype", default="fp16", choices=list(DTYPES))
    ap.add_argument("--isl", type=int, nargs="+", default=[128, 512, 2048])
    ap.add_argument("--osl", type=int, default=128)
    ap.add_argument("--reps", type=int, default=50)
    args = ap.parse_args()
    dt = DTYPES[args.dtype]
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True, torch_dtype=dt, device_map="cuda").eval()
    V = int(model.config.vocab_size)
    print(f"=== {args.model}\n    {args.dtype} | {torch.cuda.get_device_name(0)} | "
          f"RWKV7_FAST_FORWARD={os.environ.get('RWKV7_FAST_FORWARD','(default)')} ===")

    print("\nTTFT (single prefill call, use_cache=False):")
    for isl in args.isl:
        ids = torch.randint(0, V, (1, isl), device="cuda")
        with torch.no_grad():
            for _ in range(5):
                model(ids, use_cache=False)
            ts = []
            for _ in range(args.reps):
                torch.cuda.synchronize(); t0 = time.perf_counter()
                model(ids, use_cache=False)
                torch.cuda.synchronize(); ts.append((time.perf_counter() - t0) * 1000)
        p50, p99 = pct(ts, .5), pct(ts, .99)
        print(f"  ISL={isl:5d}  p50={p50:7.2f}ms  p99={p99:7.2f}ms  "
              f"(~{isl / p50 * 1000:.0f} tok/s prefill)")

    print("\nTPOT (single-token decode, bsz=1, thread state):")
    seed = torch.randint(0, V, (1, 64), device="cuda")
    with torch.no_grad():
        out = model(seed, use_cache=True, logits_to_keep=1)
        st, nx = out.past_key_values, out.logits[:, -1:].argmax(-1)
        for _ in range(10):
            out = model(nx, past_key_values=st, use_cache=True, logits_to_keep=1)
            st, nx = out.past_key_values, out.logits[:, -1:].argmax(-1)
        ts = []
        for _ in range(args.osl):
            torch.cuda.synchronize(); t0 = time.perf_counter()
            out = model(nx, past_key_values=st, use_cache=True, logits_to_keep=1)
            st, nx = out.past_key_values, out.logits[:, -1:].argmax(-1)
            torch.cuda.synchronize(); ts.append((time.perf_counter() - t0) * 1000)
    p50, p99 = pct(ts, .5), pct(ts, .99)
    print(f"  p50={p50:6.3f}ms/tok  p99={p99:6.3f}ms/tok  (~{1000 / p50:.0f} tok/s decode)")


if __name__ == "__main__":
    main()

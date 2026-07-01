#!/usr/bin/env python3
# coding=utf-8
"""50-series (Blackwell sm_120) training regression test.

On Blackwell the FLA-backed wrapper's backward exceeds shared memory
(`chunk_A_bwd` needs 128KB > 5070's 99KB), so training must use the
pure-PyTorch NativeRWKV7ForCausalLM. This test asserts that path trains
(loss strictly decreases) and never trips the triton shared-memory limit.

  python tests/test_native_training_smoke.py --model <hf_dir>
"""
from __future__ import annotations

import argparse

import torch
from transformers import AutoTokenizer

from rwkv7_hf.native_model import NativeRWKV7ForCausalLM

PROMPT = "User: Hello!\n\nAssistant: Hi there, how can I help you today?"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--steps", type=int, default=5)
    args = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    # fp32: fp16 grad overflows on this tiny model without loss scaling; the
    # point of this test is the backward PATH (no fla shared-mem kernel), not dtype.
    model = NativeRWKV7ForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float32, device_map="cuda").train()
    for p in model.parameters():
        p.requires_grad_(False)
    # LoRA-like: only r_proj per layer trainable.
    for layer in model.model.layers:
        for p in layer.attn.r_proj.parameters():
            p.requires_grad_(True)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=5e-4)
    ids = tok(PROMPT, return_tensors="pt", add_special_tokens=False).input_ids.to("cuda")
    labels = ids.clone()
    losses = []
    for _ in range(args.steps):
        out = model(input_ids=ids, labels=labels)
        out.loss.backward()
        opt.step()
        opt.zero_grad()
        losses.append(float(out.loss))
    print("native SFT losses:", [round(x, 4) for x in losses])
    ok = len(losses) >= 3 and losses[-1] < losses[0] and all(l == l for l in losses)  # no nan
    print("NATIVE TRAINING SMOKE PASS" if ok else "NATIVE TRAINING SMOKE FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

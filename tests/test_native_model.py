#!/usr/bin/env python3
# coding=utf-8
"""Regression test for the native (fla-free) RWKV-7 model (gate H1).

Verifies NativeRWKV7ForCausalLM (pure PyTorch, no fla) loads the converted
weights, forwards bit-exact vs the FLA wrapper, and generates token-identical
greedy output.

  python tests/test_native_model.py --model <hf_dir>
"""
from __future__ import annotations

import argparse

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from rwkv7_hf.native_model import NativeRWKV7ForCausalLM

PROMPTS = [
    "The quick brown fox jumps over the lazy dog.",
    "Once upon a time, in a faraway land,",
    "User: Hello!\n\nAssistant:",
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--gen-tokens", type=int, default=16)
    args = ap.parse_args()
    d = args.model
    tok = AutoTokenizer.from_pretrained(d, trust_remote_code=True)
    fla = AutoModelForCausalLM.from_pretrained(
        d, trust_remote_code=True, torch_dtype=torch.float32, device_map="cuda").eval()
    nat = NativeRWKV7ForCausalLM.from_pretrained(
        d, torch_dtype=torch.float32, device_map="cuda").eval()

    worst_cos, worst_abs, argmax_ok = 1.0, 0.0, 0
    for p in PROMPTS:
        ids = tok(p, return_tensors="pt", add_special_tokens=False).input_ids.to("cuda")
        with torch.no_grad():
            lf = fla(ids).logits[0, -1].float().cpu()
            ln = nat(ids).logits[0, -1].float().cpu()
        cos = F.cosine_similarity(lf.unsqueeze(0), ln.unsqueeze(0)).item()
        worst_cos = min(worst_cos, cos)
        worst_abs = max(worst_abs, (lf - ln).abs().max().item())
        argmax_ok += int(lf.argmax() == ln.argmax())
    print(f"[forward] min_cos={worst_cos:.6f} max_abs={worst_abs:.6f} "
          f"argmax {argmax_ok}/{len(PROMPTS)}")

    # greedy generate token-identical
    ids = tok(PROMPTS[2], return_tensors="pt", add_special_tokens=False).input_ids.to("cuda")
    with torch.no_grad():
        no = nat.generate(ids, max_new_tokens=args.gen_tokens, do_sample=False)
        fo = fla.generate(ids, max_new_tokens=args.gen_tokens, do_sample=False,
                          use_cache=True, pad_token_id=0)
    nt = no[0, ids.shape[1]:].tolist()
    ft = fo[0, ids.shape[1]:].tolist()
    match = sum(int(a == b) for a, b in zip(nt, ft))
    print(f"[generate] greedy token-identical {match}/{len(nt)}")

    ok = worst_cos >= 0.999 and argmax_ok == len(PROMPTS) and match == len(nt)
    print("NATIVE MODEL PASS" if ok else "NATIVE MODEL FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

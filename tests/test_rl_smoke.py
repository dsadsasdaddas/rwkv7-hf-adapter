#!/usr/bin/env python3
# coding=utf-8
"""DPO (RL) smoke test for the RWKV-7 HF adapter (req-2: HF PEFT + RL training).

Verifies TRL DPOTrainer runs forward+backward on the RWKV-7 model with LoRA on a
tiny synthetic preference dataset. This is a smoke test, not convergence.
"""
from __future__ import annotations

import argparse
import os

os.environ.setdefault("RWKV_V7_ON", "1")
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer

PREF = [
    {"prompt": "User: Hello!\n\nAssistant:",
     "chosen": " Hello! I'm glad to help. What do you need?",
     "rejected": " i dont know"},
    {"prompt": "User: What is 2+2?\n\nAssistant:",
     "chosen": " 2+2 equals 4.",
     "rejected": " maybe five"},
    {"prompt": "User: Tell me a joke.\n\nAssistant:",
     "chosen": " Why did the chicken cross the road? To get to the other side!",
     "rejected": " no"},
    {"prompt": "User: Thanks!\n\nAssistant:",
     "chosen": " You're welcome! Anything else I can help with?",
     "rejected": " bye"},
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token or "<|endoftext|>"
    model = AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True, torch_dtype=torch.float16,
        device_map=args.device if args.device.startswith("cuda") else None).eval()
    lora = LoraConfig(task_type="CAUSAL_LM", r=4, lora_alpha=8, lora_dropout=0.0,
                      target_modules=["r_proj", "k_proj", "v_proj", "o_proj", "key", "value"])
    model = get_peft_model(model, lora)
    ds = Dataset.from_list(PREF)
    cfg = DPOConfig(
        output_dir="/tmp/rwkv7_dpo_smoke", num_train_epochs=1, per_device_train_batch_size=1,
        max_length=64, learning_rate=1e-4, logging_steps=1,
        save_strategy="no", report_to="none", remove_unused_columns=False,
    )
    trainer = DPOTrainer(model=model, args=cfg, train_dataset=ds, processing_class=tok)
    out = trainer.train()
    print("DPO smoke train_runtime", round(out.metrics["train_runtime"], 2),
          "train_loss", round(float(out.metrics.get("train_loss", 0.0)), 4))
    print("DPO SMOKE PASS")


if __name__ == "__main__":
    main()

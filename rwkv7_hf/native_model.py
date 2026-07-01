# coding=utf-8
"""Native RWKV-7 model — pure PyTorch, NO `fla` dependency (gate H1).

This is the phase-3 "native transformers implementation": RWKV7 modules written
from scratch (not subclassing FLA), loading the same converted weights, and
running the official TMix_one/CMix_one math ported in ``rwkv7_hf/native.py``
(verified bit-exact vs FLA at cos=1.0). Step 1: a correct, FLA-free forward.
Prefill is sequential (slow); fast decode/prefill come later (H2/H3).

The module attribute names match ``scripts/convert_rwkv7_to_hf.py`` output, so
``NativeRWKV7ForCausalLM.from_pretrained(<hf_dir>)`` loads the converted weights.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PretrainedConfig
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel

from .native import _init_state, _step_token


class NativeRWKV7Config(PretrainedConfig):
    """Standalone RWKV-7 config (no fla import). Carries the converted fields."""

    model_type = "rwkv7_native"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.hidden_size = kwargs.get("hidden_size", 768)
        self.num_hidden_layers = kwargs.get("num_hidden_layers", 12)
        self.head_dim = kwargs.get("head_dim", 64)
        self.num_heads = kwargs.get("num_heads", None) or self.hidden_size // self.head_dim
        self.intermediate_size = kwargs.get("intermediate_size", self.hidden_size * 4)
        self.decay_low_rank_dim = kwargs.get("decay_low_rank_dim", 64)
        self.gate_low_rank_dim = kwargs.get("gate_low_rank_dim", 128)
        self.a_low_rank_dim = kwargs.get("a_low_rank_dim", 64)
        self.v_low_rank_dim = kwargs.get("v_low_rank_dim", 32)
        self.layer_types = kwargs.get("layer_types", None)


class _LoRA(nn.Module):
    """Matches convert keys w_lora.lora.{0,2}.weight / lora.2.bias."""

    def __init__(self, hidden: int, low_rank: int, bias: bool):
        super().__init__()
        self.lora = nn.Sequential(
            nn.Linear(hidden, low_rank, bias=False),
            nn.Identity(),
            nn.Linear(low_rank, hidden, bias=bias),
        )

    def forward(self, x):
        return self.lora(x)


class NativeRWKV7Attention(nn.Module):
    """TMix (RWKV_x070_TMix_one). Attributes match native.attn_step access."""

    def __init__(self, config: NativeRWKV7Config, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim
        self.hidden_size = config.hidden_size
        hidden = config.hidden_size
        for p in ("x_r", "x_w", "x_k", "x_v", "x_a", "x_g"):
            setattr(self, p, nn.Parameter(torch.zeros(1, 1, hidden)))
        self.k_k = nn.Parameter(torch.zeros(hidden))
        self.k_a = nn.Parameter(torch.zeros(hidden))
        self.r_k = nn.Parameter(torch.zeros(self.num_heads, self.head_dim))
        self.r_proj = nn.Linear(hidden, hidden, bias=False)
        self.k_proj = nn.Linear(hidden, hidden, bias=False)
        self.v_proj = nn.Linear(hidden, hidden, bias=False)
        self.o_proj = nn.Linear(hidden, hidden, bias=False)
        self.w_lora = _LoRA(hidden, config.decay_low_rank_dim, bias=True)
        self.a_lora = _LoRA(hidden, config.a_low_rank_dim, bias=True)
        self.g_lora = _LoRA(hidden, config.gate_low_rank_dim, bias=False)
        if layer_idx != 0:
            self.v_lora = _LoRA(hidden, config.v_low_rank_dim, bias=True)
        self.g_norm = nn.GroupNorm(self.num_heads, hidden, eps=self.head_dim * 1e-5)


class NativeRWKV7FFN(nn.Module):
    """CMix (RWKV_x070_CMix_one)."""

    def __init__(self, config: NativeRWKV7Config):
        super().__init__()
        self.x_k = nn.Parameter(torch.zeros(config.hidden_size))
        self.key = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.value = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)


class NativeRWKV7Layer(nn.Module):
    def __init__(self, config: NativeRWKV7Config, layer_idx: int):
        super().__init__()
        self.attn = NativeRWKV7Attention(config, layer_idx)
        self.ffn = NativeRWKV7FFN(config)
        self.attn_norm = nn.LayerNorm(config.hidden_size)
        self.ffn_norm = nn.LayerNorm(config.hidden_size)
        if layer_idx == 0:
            self.pre_norm = nn.LayerNorm(config.hidden_size)


class NativeRWKV7Model(PreTrainedModel):
    config_class = NativeRWKV7Config
    base_model_prefix = "model"

    def __init__(self, config: NativeRWKV7Config):
        super().__init__(config)
        self.embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [NativeRWKV7Layer(config, i) for i in range(config.num_hidden_layers)]
        )
        self.norm = nn.LayerNorm(config.hidden_size)


class NativeRWKV7ForCausalLM(PreTrainedModel):
    config_class = NativeRWKV7Config
    base_model_prefix = "model"
    _no_split_modules = ["NativeRWKV7Layer"]
    # transformers>=5 expects dict-like _tied_weights_keys (we tie nothing).
    _tied_weights_keys = {}

    @property
    def all_tied_weights_keys(self):
        return {}

    def __init__(self, config: NativeRWKV7Config):
        super().__init__(config)
        self.model = NativeRWKV7Model(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, input_ids, **kwargs):
        base = self.model
        state, xpa, xpf, v_first = _init_state(
            self, input_ids.device, base.embeddings.weight.dtype)
        x = None
        for t in range(input_ids.shape[1]):
            x = F.embedding(input_ids[0, t:t + 1], base.embeddings.weight).reshape(-1)
            x, state, xpa, xpf, v_first = _step_token(self, x, state, xpa, xpf, v_first)
        x = base.norm(x)
        logits = F.linear(x, self.lm_head.weight).view(1, 1, -1)
        return CausalLMOutputWithPast(logits=logits)

# -*- coding: utf-8 -*-
"""可配置 Qwen2 同构复刻（轻量北冥）：0.5B / 1.5B 共用一套代码。

只实现推理所需的部分：
  embed → 28×Block(pre-RMSNorm, GQA, RoPE, SwiGLU) → ln_f → lm_head
权重从 HF safetensors 逐位加载，forward 自己写，
方便后续加层循环、动态残差、莫比乌斯扭转。
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class QwenConfig:
    d: int
    layers: int
    nq: int
    nkv: int
    hd: int
    ffn: int
    vocab: int
    eps: float = 1e-6
    theta: float = 10000.0
    maxlen: int = 131072
    tie: bool = False

    @classmethod
    def from_file(cls, path: str) -> "QwenConfig":
        with open(path, encoding="utf-8") as f:
            c = json.load(f)
        d = int(c["hidden_size"])
        nq = int(c["num_attention_heads"])
        return cls(
            d=d,
            layers=int(c["num_hidden_layers"]),
            nq=nq,
            nkv=int(c.get("num_key_value_heads", nq)),
            hd=int(c.get("head_dim") or (d // nq)),
            ffn=int(c["intermediate_size"]),
            vocab=int(c["vocab_size"]),
            eps=float(c.get("rms_norm_eps", 1e-6)),
            theta=float(c.get("rope_theta", 10000.0)),
            maxlen=int(c.get("max_position_embeddings", 32768)),
            tie=bool(c.get("tie_word_embeddings", False)),
        )


class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-6):
        super().__init__()
        self.w = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        v = x.float().pow(2).mean(-1, keepdim=True)
        return (x * torch.rsqrt(v + self.eps)).to(x.dtype) * self.w


def rope_cache(hd: int, seq_len: int, theta: float, device="cpu"):
    inv = 1.0 / (theta ** (torch.arange(0, hd, 2, device=device, dtype=torch.float32) / hd))
    pos = torch.arange(seq_len, device=device, dtype=torch.float32)
    ang = pos[:, None] * inv[None, :]
    emb = torch.cat([ang, ang], -1)
    return emb.cos(), emb.sin()


def apply_rope(x, cos, sin):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    rot = torch.cat([-x2, x1], -1)
    T = x.shape[-2]
    return x * cos[:T].view(1, 1, T, -1).to(x.dtype) + rot * sin[:T].view(1, 1, T, -1).to(x.dtype)


class Block(nn.Module):
    def __init__(self, cfg: QwenConfig):
        super().__init__()
        self.cfg = cfg
        self.ln1 = RMSNorm(cfg.d, cfg.eps)
        self.q = nn.Linear(cfg.d, cfg.nq * cfg.hd, bias=True)
        self.k = nn.Linear(cfg.d, cfg.nkv * cfg.hd, bias=True)
        self.v = nn.Linear(cfg.d, cfg.nkv * cfg.hd, bias=True)
        self.o = nn.Linear(cfg.nq * cfg.hd, cfg.d, bias=False)
        self.ln2 = RMSNorm(cfg.d, cfg.eps)
        self.gate = nn.Linear(cfg.d, cfg.ffn, bias=False)
        self.up = nn.Linear(cfg.d, cfg.ffn, bias=False)
        self.down = nn.Linear(cfg.ffn, cfg.d, bias=False)

    def forward(self, x, cos, sin):
        B, T, _ = x.shape
        c = self.cfg
        h = self.ln1(x)
        q = self.q(h).view(B, T, c.nq, c.hd).transpose(1, 2)
        k = self.k(h).view(B, T, c.nkv, c.hd).transpose(1, 2)
        v = self.v(h).view(B, T, c.nkv, c.hd).transpose(1, 2)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        k = k.repeat_interleave(c.nq // c.nkv, 1)
        v = v.repeat_interleave(c.nq // c.nkv, 1)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        a = a.transpose(1, 2).contiguous().view(B, T, c.nq * c.hd)
        x = x + self.o(a)
        h2 = self.ln2(x)
        y = self.down(F.silu(self.gate(h2)) * self.up(h2))
        return x + y


class QwenModel(nn.Module):
    def __init__(self, cfg: QwenConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab, cfg.d)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.layers)])
        self.ln_f = RMSNorm(cfg.d, cfg.eps)
        self.head = nn.Linear(cfg.d, cfg.vocab, bias=False)

    def forward(self, ids):
        T = ids.shape[1]
        cos, sin = rope_cache(self.cfg.hd, T, self.cfg.theta, ids.device)
        x = self.embed(ids)
        for blk in self.blocks:
            x = blk(x, cos, sin)
        return self.head(self.ln_f(x))


def load_native(model: QwenModel, path: str) -> QwenModel:
    """加载我们自己保存的 state_dict（key 是本模型的命名）。"""
    from safetensors.torch import load_file
    state = load_file(os.path.join(path, "model.safetensors"), device="cpu")
    model.load_state_dict(state, strict=True)
    return model


def load_hf(model: QwenModel, path: str) -> QwenModel:
    """把 HF Qwen2 safetensors 逐位映射到本模型。"""
    from safetensors.torch import load_file
    state = load_file(os.path.join(path, "model.safetensors"), device="cpu")
    cfg = model.cfg
    out = {"embed.weight": state["model.embed_tokens.weight"],
           "ln_f.w": state["model.norm.weight"]}
    if not cfg.tie and "lm_head.weight" in state:
        out["head.weight"] = state["lm_head.weight"]
    for i in range(cfg.layers):
        p = f"model.layers.{i}."
        b = f"blocks.{i}."
        out[b + "ln1.w"] = state[p + "input_layernorm.weight"]
        out[b + "q.weight"] = state[p + "self_attn.q_proj.weight"]
        out[b + "q.bias"] = state[p + "self_attn.q_proj.bias"]
        out[b + "k.weight"] = state[p + "self_attn.k_proj.weight"]
        out[b + "k.bias"] = state[p + "self_attn.k_proj.bias"]
        out[b + "v.weight"] = state[p + "self_attn.v_proj.weight"]
        out[b + "v.bias"] = state[p + "self_attn.v_proj.bias"]
        out[b + "o.weight"] = state[p + "self_attn.o_proj.weight"]
        out[b + "ln2.w"] = state[p + "post_attention_layernorm.weight"]
        out[b + "gate.weight"] = state[p + "mlp.gate_proj.weight"]
        out[b + "up.weight"] = state[p + "mlp.up_proj.weight"]
        out[b + "down.weight"] = state[p + "mlp.down_proj.weight"]
    model.load_state_dict(out, strict=True)
    if cfg.tie:
        model.head.weight = model.embed.weight
    return model

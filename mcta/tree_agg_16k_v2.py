# -*- coding: utf-8 -*-
"""16K 聚合树：8K 子块 → 16K 父块，共享一层注意力。

  token embed + depthwise conv
  → 8K 块表示（kernel=S, stride=S）
  → 16K 父表示（kernel=2, stride=2）
  → 共享注意力在 2 个 8K 子块上二选一
  → 选中 8K token + 问题 → 同一套注意力
  → 生成
"""
from __future__ import annotations

import argparse
import os
import random
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = os.environ.get("MCTA_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from mcta.sentence_memory_m1 import CharTok, read_books, build_vocab, chunked_ce  # noqa: E402


class TreeAgg16V2(nn.Module):
    def __init__(self, vocab, d=256, S=8192, nq=4, nkv=4, hd=64, ffn=1024):
        super().__init__()
        self.d, self.S = d, S
        self.embed = nn.Embedding(vocab, d)
        self.tok_dw = nn.Conv1d(d, d, 3, padding=1, groups=d)
        self.tok_pw = nn.Conv1d(d, d, 1)
        self.ctx = nn.ModuleList([nn.Conv1d(d, d, 3, padding=di, dilation=di, groups=d) for di in (1, 2, 4, 8)])
        self.ctx_pw = nn.Conv1d(d, d, 1)
        self.pool_w = nn.Linear(d, 1, bias=False)
        self.pos = nn.Embedding(9000, d)
        self.rq = nn.Linear(d, d, bias=False)
        self.rs = nn.Linear(d, d, bias=False)
        self.blk_dw = nn.Conv1d(d, d, S, stride=S, groups=d)
        self.blk_pw = nn.Conv1d(d, d, 1)
        self.par_dw = nn.Conv1d(d, d, 2, stride=2, groups=d)
        self.par_pw = nn.Conv1d(d, d, 1)
        self.qproj = nn.Linear(d, d, bias=False)
        self.Wq = nn.Linear(d, d, bias=False)
        self.Wk = nn.Linear(d, d, bias=False)
        self.Wv = nn.Linear(d, d, bias=False)
        self.Wo = nn.Linear(d, d, bias=False)
        self.ln1 = nn.LayerNorm(d)
        self.ln2 = nn.LayerNorm(d)
        self.gate = nn.Linear(d, ffn, bias=False)
        self.up = nn.Linear(d, ffn, bias=False)
        self.down = nn.Linear(ffn, d, bias=False)
        self.ln_f = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)
        self.head.weight = self.embed.weight
        self.nq, self.nkv, self.hd = nq, nkv, hd

    def encode(self, ids):
        x = self.embed(ids)
        xt = x.transpose(1, 2)
        xt = xt + F.silu(self.tok_pw(self.tok_dw(xt)))
        for c in self.ctx:
            xt = xt + F.silu(c(xt))
        xt = xt + F.silu(self.ctx_pw(xt))
        z = xt.reshape(1, self.d, 2, self.S).permute(0, 2, 3, 1)   # (1,2,S,d)
        score = self.pool_w(z).squeeze(-1)                         # (1,2,S)
        attn = torch.softmax(score, -1)
        blocks = (attn.unsqueeze(-1) * z).sum(2).transpose(1, 2)   # (1,d,2)
        parent = self.par_pw(self.par_dw(F.silu(blocks)))          # (1,d,1)
        return xt.transpose(1, 2), blocks, parent

    def route(self, q, H):
        qv = F.normalize(self.rq(q), dim=-1)
        kv = F.normalize(self.rs(H.transpose(1, 2))[0], dim=-1)
        return (kv @ qv) * 10.0

    def _attn(self, feat):
        T0 = feat.shape[1]
        feat = feat + self.pos(torch.arange(T0, device=feat.device))[None]
        h = self.ln1(feat)
        T = h.shape[1]
        q = self.Wq(h).view(1, T, self.nq, self.hd).transpose(1, 2)
        k = self.Wk(h).view(1, T, self.nkv, self.hd).transpose(1, 2)
        v = self.Wv(h).view(1, T, self.nkv, self.hd).transpose(1, 2)
        k = k.repeat_interleave(self.nq // self.nkv, 1)
        v = v.repeat_interleave(self.nq // self.nkv, 1)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        a = a.transpose(1, 2).contiguous().view(1, T, self.nq * self.hd)
        x = feat + self.Wo(a)
        x = x + self.down(F.silu(self.gate(self.ln2(x))) * self.up(self.ln2(x)))
        return x

    def token_hidden(self, feat):
        return self._attn(feat)

    def token_forward(self, feat):
        return self.head(self.ln_f(self._attn(feat)))


def make_sample(corpus, tok, S, T, rng):
    start = rng.randrange(0, max(1, len(corpus) - T - 100))
    text = corpus[start:start + T]
    target = rng.randrange(2)
    off = rng.randrange(200, S - 200)
    pos = target * S + off
    dg = "".join(str(rng.randrange(10)) for _ in range(4))
    fact = f"密钥是{dg}。"
    text = text[:pos] + fact + text[pos:]
    text = text[:T]
    ids = tok.encode(text)
    qtext = "问题：密钥是什么？\n答："
    qids = tok.encode(qtext)
    ans = tok.encode(dg + "\n")
    return ids, qids, ans, pos // S, dg


def train(args):
    corpus = read_books()
    tok = CharTok(build_vocab(corpus))
    model = TreeAgg16V2(len(tok.itos), d=args.d, S=args.S, ffn=args.ffn).to("cuda")
    if args.init:
        ck = torch.load(args.init, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["state"])
        print("init", args.init, flush=True)
    print("params", sum(p.numel() for p in model.parameters()) / 1e6, "M", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95))
    rng = random.Random(1)
    for step in range(1, args.steps + 1):
        opt.zero_grad(set_to_none=True)
        tot = 0.0
        hit = 0
        for _a in range(4):
            ids, qids, ans, gold, dg = make_sample(corpus, tok, args.S, args.T, rng)
            if gold >= 2:
                continue
            x = torch.tensor([ids], device="cuda")
            q = torch.tensor([qids], device="cuda")
            feats, blocks, parent = model.encode(x)
            qvec = model.rq(model.token_hidden(model.embed(q))[0].mean(0, keepdim=True))[0]
            logits = model.route(qvec, blocks)
            loss = F.cross_entropy(logits[None], torch.tensor([gold], device="cuda"))
            hit += int(int(logits.argmax()) == gold)
            if not args.route_only:
                s0 = gold * args.S
                blk = feats[0, s0:s0 + args.S]
                qe = model.embed(q)[0]
                ae = model.embed(torch.tensor([ans], device="cuda"))[0]
                seq = torch.cat([blk, qe, ae], 0).unsqueeze(0)
                lg = model.token_forward(seq)
                y = torch.full((1, lg.shape[1]), -100, device="cuda")
                y[0, blk.shape[0] + qe.shape[0]:] = torch.tensor([ans], device="cuda")
                loss = loss + chunked_ce(lg, y)
            (loss / 4).backward()
            tot += float(loss)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 100 == 0:
            print(f"step {step} loss={tot/4:.3f} route_acc={hit/4:.2f}", flush=True)
    name = "treeagg16v2_route.pt" if args.route_only else "treeagg16v2_full.pt"
    torch.save({"state": model.state_dict(), "vocab": tok.itos, "S": args.S,
                "T": args.T, "d": args.d, "ffn": args.ffn}, os.path.join(ROOT, "state", name))


@torch.no_grad()
def evaluate(args):
    ck = torch.load(os.path.join(ROOT, "state", args.ckpt), map_location="cpu", weights_only=False)
    tok = CharTok(ck["vocab"])
    model = TreeAgg16V2(len(tok.itos), d=ck["d"], S=ck["S"], ffn=ck["ffn"]).to("cuda")
    model.load_state_dict(ck["state"]); model.eval()
    corpus = read_books()
    rng = random.Random(999)
    ok = rec = 0
    n = 16
    for _ in range(n):
        ids, qids, ans, gold, dg = make_sample(corpus, tok, ck["S"], ck["T"], rng)
        x = torch.tensor([ids], device="cuda")
        q = torch.tensor([qids], device="cuda")
        feats, blocks, parent = model.encode(x)
        qvec = model.rq(model.token_hidden(model.embed(q))[0].mean(0, keepdim=True))[0]
        top = model.route(qvec, blocks).argmax().item()
        rec += int(top == gold)
        s0 = top * ck["S"]
        seq = torch.cat([feats[0, s0:s0 + ck["S"]], model.embed(q)[0]], 0).unsqueeze(0)
        out = []
        for _g in range(4):
            lg = model.token_forward(seq)
            t = int(lg[0, -1].argmax())
            out.append(t)
            seq = torch.cat([seq, model.embed(torch.tensor([[t]], device="cuda"))], 1)
        ok += int(tok.decode(out) == dg)
    print(f"acc={ok/n:.3f} recall={rec/n:.3f}", flush=True)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["route", "train", "eval"], required=True)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--T", type=int, default=16384)
    ap.add_argument("--S", type=int, default=8192)
    ap.add_argument("--d", type=int, default=256)
    ap.add_argument("--ffn", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--init", default="")
    ap.add_argument("--ckpt", default="treeagg16v2_full.pt")
    ap.add_argument("--route-only", action="store_true")
    args = ap.parse_args()
    if args.mode == "route":
        args.route_only = True
        train(args)
    elif args.mode == "train":
        train(args)
    else:
        evaluate(args)


if __name__ == "__main__":
    main()

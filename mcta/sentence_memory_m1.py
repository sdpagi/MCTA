# -*- coding: utf-8 -*-
"""M2：句子级块 + 段落/句子路由 + 长上下文 needle。

  · 以句子为最小块（句号/问号/感叹号/分号/换行）
  · 超长句再按逗号切
  · 路由：用问题位置的 hidden 对句子摘要打分，选 top-k 句
  · 注意力只对"选中句子的 token + 局部窗口"做（gather，不建 T×T 矩阵）
  · 训练：LM loss（答案）+ 路由监督（事实句）
"""
from __future__ import annotations

import argparse
import glob
import math
import os
import random
import re
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from mcta.expert_qwen import rope_cache, apply_rope

ROOT = os.environ.get("MCTA_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BOOKS = os.path.join(ROOT, "books")
VOCAB_SIZE = 3000
MAX_SENT = 120


def read_books(cap_mb=60):
    """递归读取 books/，按 cap_mb 截断，避免 400MB 全量进内存。"""
    files = glob.glob(os.path.join(BOOKS, "**", "*.txt"), recursive=True)
    files.sort()
    parts = []
    total = 0
    cap = cap_mb * 1024 * 1024
    for p in files:
        if total >= cap:
            break
        try:
            raw = open(p, "rb").read()
        except Exception:
            continue
        s = None
        for enc in ("utf-8", "gb18030", "utf-16"):
            try:
                s = raw.decode(enc)
                break
            except Exception:
                pass
        if not s:
            continue
        s = s.replace("\r", "\n")
        parts.append(s)
        total += len(s)
    return "\n".join(parts)


def split_sentences(text):
    parts = re.split(r"(?<=[。！？；\n])", text)
    out = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        if len(p) > MAX_SENT:
            sub = re.split(r"(?<=[，、])", p)
            cur = ""
            for s in sub:
                if len(cur) + len(s) > MAX_SENT:
                    if cur:
                        out.append(cur)
                    cur = s
                else:
                    cur += s
            if cur:
                out.append(cur)
        else:
            out.append(p)
    return out


class CharTok:
    def __init__(self, chars):
        self.itos = chars
        self.stoi = {c: i for i, c in enumerate(chars)}
        self.unk = len(chars) - 1

    def encode(self, s):
        return [self.stoi.get(c, self.unk) for c in s]

    def decode(self, ids):
        return "".join(self.itos[i] if 0 <= i < len(self.itos) else "?" for i in ids)


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.w = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        return self.w * x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)


class Block(nn.Module):
    def __init__(self, d, nq, nkv, hd, ffn):
        super().__init__()
        self.nq, self.nkv, self.hd = nq, nkv, hd
        self.ln1, self.ln2 = RMSNorm(d), RMSNorm(d)
        self.q = nn.Linear(d, nq * hd, bias=False)
        self.k = nn.Linear(d, nkv * hd, bias=False)
        self.v = nn.Linear(d, nkv * hd, bias=False)
        self.o = nn.Linear(nq * hd, d, bias=False)
        self.g = nn.Linear(d, ffn, bias=False)
        self.u = nn.Linear(d, ffn, bias=False)
        self.dn = nn.Linear(ffn, d, bias=False)


class SmallLM(nn.Module):
    def __init__(self, vocab, d=384, layers=8, nq=6, nkv=2, hd=64, ffn=1536, theta=10000.0):
        super().__init__()
        self.d = d
        self.hd = hd
        self.attn_impl = "einsum"
        self.theta = theta
        self.embed = nn.Embedding(vocab, d)
        self.blocks = nn.ModuleList([Block(d, nq, nkv, hd, ffn) for _ in range(layers)])
        self.ln_f = RMSNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)
        self.head.weight = self.embed.weight
        self.rq = nn.Linear(d, d, bias=False)
        self.sum = nn.Linear(d, d, bias=False)

    def sentence_summary(self, h, sent_ids):
        # sent_ids: (T,) long，句子编号 0..n-1
        n = int(sent_ids.max().item()) + 1
        sums = torch.zeros(n, h.shape[-1], device=h.device, dtype=h.dtype)
        cnt = torch.zeros(n, 1, device=h.device, dtype=h.dtype)
        sums.index_add_(0, sent_ids, h[0])
        cnt.index_add_(0, sent_ids, torch.ones(h.shape[1], 1, device=h.device, dtype=h.dtype))
        return self.sum(sums / cnt.clamp(min=1))

    def route(self, x, sent_ids, k, pos=-1):
        # 用指定位置（问题位置）的 query 选句
        q = self.rq(x[:, pos:pos + 1] if pos >= 0 else x[:, -1:])   # (1,1,d)
        s = self.sentence_summary(x, sent_ids)         # (n,d)
        logits = (q[0, 0] @ s.T) / math.sqrt(self.d)   # (n,)
        cur = int(sent_ids[pos if pos >= 0 else -1].item())
        logits[cur + 1:] = -1e9                        # 只能选当前句之前
        top = logits.topk(min(k, logits.numel())).indices
        return logits, top

    def block_full(self, blk, x, cos, sin):
        T = x.shape[1]
        q = blk.q(blk.ln1(x)).view(1, T, blk.nq, blk.hd).transpose(1, 2)
        kk = blk.k(blk.ln1(x)).view(1, T, blk.nkv, blk.hd).transpose(1, 2)
        vv = blk.v(blk.ln1(x)).view(1, T, blk.nkv, blk.hd).transpose(1, 2)
        q = apply_rope(q, cos, sin)
        kk = apply_rope(kk, cos, sin)
        kk = kk.repeat_interleave(blk.nq // blk.nkv, 1)
        vv = vv.repeat_interleave(blk.nq // blk.nkv, 1)
        a = F.scaled_dot_product_attention(q, kk, vv, is_causal=True)
        a = a.transpose(1, 2).contiguous().view(1, T, blk.nq * blk.hd)
        x = x + blk.o(a)
        x = x + blk.dn(F.silu(blk.g(blk.ln2(x))) * blk.u(blk.ln2(x)))
        return x

    def block_attn(self, blk, x, cand, cos, sin):
        T = x.shape[1]
        q = blk.q(blk.ln1(x)).view(1, T, blk.nq, blk.hd).transpose(1, 2)
        kk = blk.k(blk.ln1(x)).view(1, T, blk.nkv, blk.hd).transpose(1, 2)
        vv = blk.v(blk.ln1(x)).view(1, T, blk.nkv, blk.hd).transpose(1, 2)
        q = apply_rope(q, cos, sin)
        kk = apply_rope(kk, cos, sin)
        kk = kk.repeat_interleave(blk.nq // blk.nkv, 1)
        vv = vv.repeat_interleave(blk.nq // blk.nkv, 1)
        idx = torch.arange(T, device=x.device)
        Kc = kk[:, :, cand, :]
        Vc = vv[:, :, cand, :]
        mask = cand[None, :] <= idx[:, None]
        if self.attn_impl == "sdpa":
            a = F.scaled_dot_product_attention(
                q, Kc, Vc, attn_mask=mask[None, None, :, :].to(torch.bool))
        else:
            score = torch.einsum("bhqd,bhnd->bhqn", q, Kc) / math.sqrt(blk.hd)
            score = score.masked_fill(~mask[None, None, :, :], -1e9)
            a = torch.softmax(score, -1) @ Vc
        a = a.transpose(1, 2).contiguous().view(1, T, blk.nq * blk.hd)
        x = x + blk.o(a)
        x = x + blk.dn(F.silu(blk.g(blk.ln2(x))) * blk.u(blk.ln2(x)))
        return x

    def forward(self, ids, sent_ids, k=8, local=64, route_pos=-1, mode="route"):
        x = self.embed(ids)
        cache = {}
        T = ids.shape[1]
        cos, sin = rope_cache(self.hd, T, self.theta, ids.device)
        use_ckpt = self.training and torch.is_grad_enabled()
        for blk in self.blocks:
            if mode == "full":
                if use_ckpt:
                    x = checkpoint(self.block_full, blk, x, cos, sin, use_reentrant=False)
                else:
                    x = self.block_full(blk, x, cos, sin)
                continue
            # 路由在 checkpoint 外算一次，候选集固定，避免重算时 top-k 变化
            logits, top = self.route(x, sent_ids, k, route_pos)
            sel = torch.zeros_like(sent_ids, dtype=torch.bool)
            for s in top.tolist():
                sel |= (sent_ids == s)
            idx = torch.arange(T, device=ids.device)
            sel |= (idx >= T - local)
            sel[0] = True
            cand = idx[sel]
            if use_ckpt:
                x = checkpoint(self.block_attn, blk, x, cand, cos, sin, use_reentrant=False)
            else:
                x = self.block_attn(blk, x, cand, cos, sin)
            cache.setdefault("logits", []).append(logits)
            cache.setdefault("top", []).append(top)
        return self.head(self.ln_f(x)), cache


def build_vocab(corpus):
    from collections import Counter
    cnt = Counter(corpus)
    chars = [c for c, _ in cnt.most_common(VOCAB_SIZE)]
    for c in "0123456789密钥是问题答：。，\n ":
        if c not in chars:
            chars.append(c)
    return chars



def chunked_ce(lg, y, chunk=2048):
    V = lg.shape[-1]
    total = 0.0
    cnt = 0
    T = lg.shape[1]
    for i in range(0, T - 1, chunk):
        end = min(i + chunk, T - 1)
        l = lg[:, i:end].float().reshape(-1, V)
        t = y[:, i + 1:end + 1].reshape(-1)
        total = total + F.cross_entropy(l, t, ignore_index=-100, reduction="sum")
        cnt += int((t != -100).sum())
    return total / max(cnt, 1)

def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--chars", type=int, default=4096)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--d", type=int, default=384)
    ap.add_argument("--ffn", type=int, default=1536)
    ap.add_argument("--nq", type=int, default=6)
    ap.add_argument("--nkv", type=int, default=2)
    ap.add_argument("--hd", type=int, default=64)
    ap.add_argument("--tag", default="")
    ap.add_argument("--eval-every", type=int, default=200)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup", type=int, default=0)
    ap.add_argument("--route-weight", type=float, default=0.1)
    ap.add_argument("--slow-step", type=int, default=0)
    ap.add_argument("--slow-val", type=float, default=1e9)
    ap.add_argument("--mode", default="route", choices=["route", "full"])
    ap.add_argument("--theta", type=float, default=10000.0)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    corpus = read_books()
    sents = split_sentences(corpus)
    chars = build_vocab(corpus)
    tok = CharTok(chars)
    print("sentences", len(sents), "vocab", len(chars), flush=True)

    def sample(rng):
        # 选一段句子，拼到约 chars 长度
        start = rng.randrange(0, max(1, len(sents) - 50))
        ctx = []
        total = 0
        i = start
        while i < len(sents) and total < args.chars:
            ctx.append(sents[i])
            total += len(sents[i])
            i += 1
        # 插入事实句
        dg = "".join(str(rng.randrange(10)) for _ in range(4))
        fact = f"密钥是{dg}。"
        fpos = rng.randrange(1, max(2, len(ctx) - 1))
        ctx = ctx[:fpos] + [fact] + ctx[fpos:]
        prompt_sents = ctx + ["问题：密钥是什么？", "答："]
        text = "".join(prompt_sents)
        target = dg + "\n"
        ids = tok.encode(text) + tok.encode(target)
        labels = [-100] * len(tok.encode(text)) + tok.encode(target)
        # 句子编号
        sent_ids = []
        for si, s in enumerate(prompt_sents):
            sent_ids += [si] * len(tok.encode(s))
        sent_ids += [len(prompt_sents)] * len(tok.encode(target))
        return ids, labels, sent_ids, fpos, dg

    model = SmallLM(len(chars), d=args.d, layers=args.layers, nq=args.nq, nkv=args.nkv, hd=args.hd, ffn=args.ffn, theta=args.theta).to("cuda")
    print("params", sum(p.numel() for p in model.parameters())/1e6, "M", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95))
    def _lr_lambda(st):
        if args.warmup > 0 and st < args.warmup:
            return max(st, 1) / args.warmup
        prog = (st - args.warmup) / max(1, args.steps - args.warmup)
        prog = min(1.0, max(0.0, prog))
        return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * prog))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, _lr_lambda)
    rng = random.Random(args.seed + 1)
    ckpt_path = os.path.join(ROOT, "state",
                             f"sentence_memory_m2_{args.chars}{args.tag}.pt")

    def eval_val(n=8):
        model.eval()
        r2 = random.Random(123)
        tot = 0.0
        with torch.no_grad():
            for _ in range(n):
                ids, labels, sent_ids, fpos, dg = sample(r2)
                plen = len(ids) - len(tok.encode(dg + "\n"))
                x = torch.tensor([ids], device="cuda")
                y = torch.tensor([labels], device="cuda")
                sid = torch.tensor(sent_ids, device="cuda")
                if args.mode == "full":
                    lg, _ = model(x, sid, mode="full")
                else:
                    lg, _ = model(x, sid, k=args.k, route_pos=plen - 1)
                tot += float(chunked_ce(lg, y))
        model.train()
        return tot / n

    best = None
    bad = 0
    for step in range(1, args.steps + 1):
        opt.zero_grad(set_to_none=True)
        tot = 0.0
        for _a in range(4):                      # 梯度累积 4 条，等效 batch=4
            ids, labels, sent_ids, fpos, dg = sample(rng)
            x = torch.tensor([ids], device="cuda")
            y = torch.tensor([labels], device="cuda")
            sid = torch.tensor(sent_ids, device="cuda")
            plen = len(ids) - len(tok.encode(dg + "\n"))
            if args.mode == "full":
                lg, cache = model(x, sid, mode="full")
                loss = chunked_ce(lg, y)
            else:
                lg, cache = model(x, sid, k=args.k, route_pos=plen - 1)
                loss = chunked_ce(lg, y)
                tgt = torch.tensor([fpos], device="cuda")
                for lg_l in cache["logits"]:
                    loss = loss + args.route_weight * F.cross_entropy(lg_l[None], tgt.clamp(max=lg_l.numel() - 1))
            (loss / 4).backward()
            tot += float(loss)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % args.eval_every == 0:
            v = eval_val()
            mark = ""
            if best is None or v < best:
                best, bad = v, 0
                torch.save({"state": model.state_dict(), "chars": chars}, ckpt_path)
                mark = " *best"
            else:
                bad += 1
            print(f"step {step} loss={tot/4:.3f} val={v:.3f}{mark}", flush=True)
            if bad >= args.patience:
                print("早停", flush=True)
                break
            if args.slow_step and step == args.slow_step and v > args.slow_val:
                print(f"收敛过慢（step {step} val {v:.3f} > {args.slow_val}），停", flush=True)
                break

    if os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["state"])
    # 评估
    model.eval()
    ok = 0
    rec = 0
    n = 16
    with torch.no_grad():
        for _ in range(n):
            ids, labels, sent_ids, fpos, dg = sample(random.Random(999 + _))
            plen = len(ids) - len(tok.encode(dg + "\n"))
            x = torch.tensor([ids[:plen]], device="cuda")
            sid = torch.tensor(sent_ids[:plen], device="cuda")
            out = []
            hit = False
            for _g in range(4):
                if args.mode == "full":
                    lg, cache = model(x, sid, mode="full")
                else:
                    lg, cache = model(x, sid, k=args.k, route_pos=plen - 1)
                if args.mode != "full" and any(fpos in t.tolist() for t in cache["top"]):
                    hit = True
                t = int(lg[0, -1].argmax())
                out.append(t)
                x = torch.cat([x, torch.tensor([[t]], device="cuda")], 1)
                sid = torch.cat([sid, torch.tensor([sid[-1].item()], device="cuda")])
            if tok.decode(out) == dg:
                ok += 1
            rec += int(hit)
            if _ < 4:
                print("  pred", repr(tok.decode(out)), "gold", repr(dg), "fpos", fpos, flush=True)
    print("FINAL acc", round(ok / n, 3), "recall", round(rec / n, 3), flush=True)
    if not os.path.exists(ckpt_path):
        torch.save({"state": model.state_dict(), "chars": chars}, ckpt_path)


if __name__ == "__main__":
    main()

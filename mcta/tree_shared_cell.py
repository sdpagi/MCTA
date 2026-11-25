# -*- coding: utf-8 -*-
"""共享 LevelCell：训练 16K（一层），零样本堆层到 32K/64K。

  token embed + 上下文卷积
  → 8K 块表示（attention pooling）
  → LevelCell：2 个子块 → 父块 + 路由 logits
  → 同一套 LevelCell 递归堆层
  → 选中 8K token + 问题 → token 读出
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


class LevelCell(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.ln = nn.LayerNorm(d)
        self.agg = nn.Linear(2 * d, d, bias=False)
        self.out_ln = nn.LayerNorm(d)
        self.use_residual = True
        self.res_mode = "mean"
        self.rq = nn.Linear(d, d, bias=False)
        self.rs = nn.Linear(d, d, bias=False)

    def forward(self, children, q):
        # children: (G,2,d), q: (d,)
        c = self.ln(children)
        agg = self.agg(c.reshape(c.shape[0], 2 * c.shape[2]))
        qv = F.normalize(self.rq(q), dim=-1)
        kv = F.normalize(self.rs(c), dim=-1)
        logits = torch.einsum("d,gcd->gc", qv, kv) * 10.0
        if not self.use_residual:
            parent = self.out_ln(agg)
        elif self.res_mode == "attn":
            w = torch.softmax(logits, -1)
            ctx = (w.unsqueeze(-1) * c).sum(1)
            parent = self.out_ln(agg + ctx)
        elif self.res_mode == "both":
            w = torch.softmax(logits, -1)
            ctx = (w.unsqueeze(-1) * c).sum(1)
            parent = self.out_ln(agg + c.mean(1) + ctx)
        else:
            parent = self.out_ln(agg + c.mean(1))
        return parent, logits


class SharedTree(nn.Module):
    def __init__(self, vocab, d=256, S=8192, nq=4, nkv=4, hd=64, ffn=1024):
        super().__init__()
        self.d, self.S = d, S
        self.embed = nn.Embedding(vocab, d)
        self.tok_dw = nn.Conv1d(d, d, 3, padding=1, groups=d)
        self.tok_pw = nn.Conv1d(d, d, 1)
        self.ctx = nn.ModuleList([nn.Conv1d(d, d, 3, padding=di, dilation=di, groups=d) for di in (1, 2, 4, 8)])
        self.ctx_pw = nn.Conv1d(d, d, 1)
        self.pool_w = nn.Linear(d, 1, bias=False)
        self.cell = LevelCell(d)
        self.qenc = nn.Linear(d, d, bias=False)
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
        self.readout_head = nn.Linear(d, vocab, bias=False)
        self.readout_head.weight.data.copy_(self.embed.weight.data)
        self.pos = nn.Embedding(9000, d)
        self.nq, self.nkv, self.hd = nq, nkv, hd

    def blocks(self, ids):
        x = self.embed(ids)
        xt = x.transpose(1, 2)
        xt = xt + F.silu(self.tok_pw(self.tok_dw(xt)))
        for c in self.ctx:
            xt = xt + F.silu(c(xt))
        xt = xt + F.silu(self.ctx_pw(xt))
        n = xt.shape[2] // self.S
        z = xt.reshape(1, self.d, n, self.S).permute(0, 2, 3, 1)   # (1,n,S,d)
        score = self.pool_w(z).squeeze(-1)
        attn = torch.softmax(score, -1)
        b = (attn.unsqueeze(-1) * z).sum(2)                         # (1,n,d)
        return xt.transpose(1, 2), b[0]

    def build_tree(self, blocks, q):
        """自底向上，返回每层 (parent_reps, logits)。非 2 的幂时 pad 到 2 的幂。"""
        n = blocks.shape[0]
        p = 1 << (n - 1).bit_length()
        if p > n:
            blocks = torch.cat([blocks, blocks.new_zeros(p - n, self.d)], 0)
        levels = []
        cur = blocks
        while cur.shape[0] >= 2:
            g = cur.shape[0] // 2
            children = cur[:g * 2].reshape(g, 2, self.d)
            parent, logits = self.cell(children, q)
            levels.append((parent, logits))
            cur = parent
        return levels

    def _attn(self, feat):
        T0 = feat.shape[1]
        feat = feat + self.pos(torch.arange(T0, device=feat.device))[None]
        h = self.ln1(feat)
        q = self.Wq(h).view(1, T0, self.nq, self.hd).transpose(1, 2)
        k = self.Wk(h).view(1, T0, self.nkv, self.hd).transpose(1, 2)
        v = self.Wv(h).view(1, T0, self.nkv, self.hd).transpose(1, 2)
        k = k.repeat_interleave(self.nq // self.nkv, 1)
        v = v.repeat_interleave(self.nq // self.nkv, 1)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        a = a.transpose(1, 2).contiguous().view(1, T0, self.nq * self.hd)
        x = feat + self.Wo(a)
        x = x + self.down(F.silu(self.gate(self.ln2(x))) * self.up(self.ln2(x)))
        return x

    def token_q(self, qids):
        e = self.embed(torch.tensor([qids], device="cuda")).mean(1)
        return self.qenc(e)[0]

    def token_forward(self, feat):
        return self.head(self.ln_f(self._attn(feat)))


def make_sample(corpus, tok, S, T, rng, nblocks):
    start = rng.randrange(0, max(1, len(corpus) - T - 100))
    text = corpus[start:start + T]
    target = rng.randrange(nblocks)
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
    return ids, qids, ans, target, dg


def make_sample_at(corpus, tok, S, T, rng, target):
    start = rng.randrange(0, max(1, len(corpus) - T - 100))
    text = corpus[start:start + T]
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
    return ids, qids, ans, target, dg


def train_rdepth(args):
    corpus = read_books()
    tok = CharTok(build_vocab(corpus))
    model = SharedTree(len(tok.itos), d=args.d, S=args.S, ffn=args.ffn).to("cuda")
    model.cell.res_mode = args.res_mode
    if args.init:
        ck = torch.load(args.init, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["state"], strict=False)
        print("init", args.init, flush=True)
    print("params", sum(p.numel() for p in model.parameters()) / 1e6, "M", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95))
    rng = random.Random(1)
    for step in range(1, args.steps + 1):
        opt.zero_grad(set_to_none=True)
        tot = 0.0
        hit = 0
        for _a in range(4):
            nb = args.T // args.S
            if nb < 4:
                depth = 1
                target = rng.randrange(nb)
            else:
                depth = 2 if rng.random() < 0.5 else 1
                if depth == 1:
                    pair = rng.randrange(nb // 2)
                    target = pair * 2 + rng.randrange(2)
                else:
                    target = rng.randrange(nb)
            ids, qids, ans, gold, dg = make_sample_at(corpus, tok, args.S, args.T, rng, target)
            x = torch.tensor([ids], device="cuda")
            feats, blocks = model.blocks(x)
            rb = blocks if args.route_only else blocks.detach()
            q = model.token_q(qids)
            if depth == 2 and rb.shape[0] < 4:
                depth = 1
            if depth == 1:
                pair = target // 2
                children = rb[pair * 2:(pair + 1) * 2]
                parent, logits = model.cell(children.reshape(1, 2, model.d), q)
                loss = F.cross_entropy(logits[0][None], torch.tensor([target % 2], device="cuda"))
                hit += int(int(logits[0].argmax()) == target % 2)
            else:
                levels = model.build_tree(rb, q)
                gold2, gold8 = target // 2, target % 2
                l2 = levels[1][1][0]
                l8 = levels[0][1][gold2]
                loss = F.cross_entropy(l2[None], torch.tensor([gold2], device="cuda"))
                loss = loss + F.cross_entropy(l8[None], torch.tensor([gold8], device="cuda"))
                hit += int(int(l2.argmax()) == gold2) * int(int(l8.argmax()) == gold8)
            if args.joint:
                s0 = target * args.S
                blk = feats[0, s0:s0 + args.S]
                qe = model.embed(torch.tensor([qids], device="cuda"))[0]
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
    torch.save({"state": model.state_dict(), "vocab": tok.itos, "S": args.S,
                "T": args.T, "d": args.d, "ffn": args.ffn}, os.path.join(ROOT, "state", "sharedtree_rd.pt"))


def train_rdepth3(args):
    corpus = read_books(); tok = CharTok(build_vocab(corpus))
    model = SharedTree(len(tok.itos), d=args.d, S=args.S, ffn=args.ffn).to("cuda")
    model.cell.res_mode = args.res_mode
    if args.init:
        ck = torch.load(args.init, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["state"], strict=False)
        print("init", args.init, flush=True)
    print("params", sum(p.numel() for p in model.parameters()) / 1e6, "M", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95))
    rng = random.Random(1)
    maxd = 3
    for step in range(1, args.steps + 1):
        opt.zero_grad(set_to_none=True); tot = 0.0; hit = 0
        for _a in range(4):
            nb = args.T // args.S
            target = rng.randrange(nb)
            ids, qids, ans, gold, dg = make_sample_at(corpus, tok, args.S, args.T, rng, target)
            x = torch.tensor([ids], device="cuda")
            feats, blocks = model.blocks(x)
            q = model.token_q(qids)
            levels = model.build_tree(blocks if args.route_only else blocks.detach(), q)
            depth = rng.randint(1, min(maxd, len(levels)))
            g = target; loss = 0.0; correct = True
            for li in range(depth):
                logits = levels[li][1][g // 2]
                bit = g % 2
                loss = loss + F.cross_entropy(logits[None], torch.tensor([bit], device="cuda"))
                correct = correct and (int(logits.argmax()) == bit)
                g = g // 2
            hit += int(correct)
            (loss / 4).backward(); tot += float(loss)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        if step % 200 == 0:
            print(f"step {step} loss={tot/4:.3f} route_acc={hit/4:.2f}", flush=True)
    torch.save({"state": model.state_dict(), "vocab": tok.itos, "S": args.S,
                "T": args.T, "d": args.d, "ffn": args.ffn, "res_mode": args.res_mode}, os.path.join(ROOT, "state", args.save))


def train(args):
    corpus = read_books()
    tok = CharTok(build_vocab(corpus))
    model = SharedTree(len(tok.itos), d=args.d, S=args.S, ffn=args.ffn).to("cuda")
    if args.init:
        ck = torch.load(args.init, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["state"], strict=False)
        print("init", args.init, flush=True)
    print("params", sum(p.numel() for p in model.parameters()) / 1e6, "M", flush=True)
    if args.freeze_route:
        frozen = list(model.cell.parameters()) + list(model.qenc.parameters()) + list(model.tok_dw.parameters()) + list(model.tok_pw.parameters()) + list(model.ctx.parameters()) + list(model.ctx_pw.parameters()) + list(model.pool_w.parameters())
        for p in frozen:
            p.requires_grad = False
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95))
    rng = random.Random(1)
    for step in range(1, args.steps + 1):
        opt.zero_grad(set_to_none=True)
        tot = 0.0
        hit = 0
        for _a in range(4):
            ids, qids, ans, gold, dg = make_sample(corpus, tok, args.S, args.T, rng, 2)
            x = torch.tensor([ids], device="cuda")
            q = torch.tensor([qids], device="cuda")
            feats, blocks = model.blocks(x)
            qv = model.token_q(qids)
            levels = model.build_tree(blocks if args.route_only else blocks.detach(), qv)
            parent, logits = levels[0]
            loss = F.cross_entropy(logits[0][None], torch.tensor([gold], device="cuda"))
            hit += int(int(logits[0].argmax()) == gold)
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
    name = "sharedtree16_route.pt" if args.route_only else "sharedtree16_full.pt"
    torch.save({"state": model.state_dict(), "vocab": tok.itos, "S": args.S,
                "T": args.T, "d": args.d, "ffn": args.ffn}, os.path.join(ROOT, "state", name))


@torch.no_grad()
def evaluate(args):
    ck = torch.load(os.path.join(ROOT, "state", args.ckpt), map_location="cpu", weights_only=False)
    tok = CharTok(ck["vocab"])
    model = SharedTree(len(tok.itos), d=ck["d"], S=ck["S"], ffn=ck["ffn"]).to("cuda")
    model.load_state_dict(ck["state"], strict=False); model.eval()
    corpus = read_books()
    rng = random.Random(999)
    ok = rec = 0
    n = 16
    for _ in range(n):
        ids, qids, ans, gold, dg = make_sample(corpus, tok, ck["S"], ck["T"], rng, 2)
        x = torch.tensor([ids], device="cuda")
        feats, blocks = model.blocks(x)
        q = model.token_q(qids)
        rb = blocks if args.route_only else blocks.detach()
        levels = model.build_tree(rb, q)
        logits = levels[0][1][0]
        top = int(logits.argmax())
        rec += int(top == gold)
        s0 = top * ck["S"]
        seq = torch.cat([feats[0, s0:s0 + ck["S"]], model.embed(torch.tensor([qids], device="cuda"))[0]], 0).unsqueeze(0)
        out = []
        for _g in range(4):
            lg = model.token_forward(seq)
            t = int(lg[0, -1].argmax())
            out.append(t)
            seq = torch.cat([seq, model.embed(torch.tensor([[t]], device="cuda"))], 1)
        ok += int(tok.decode(out) == dg)
    print(f"acc={ok/n:.3f} recall={rec/n:.3f}", flush=True)


@torch.no_grad()
def extrapolate(args):
    ck = torch.load(os.path.join(ROOT, "state", args.ckpt), map_location="cpu", weights_only=False)
    tok = CharTok(ck["vocab"])
    model = SharedTree(len(tok.itos), d=ck["d"], S=ck["S"], ffn=ck["ffn"]).to("cuda")
    model.load_state_dict(ck["state"], strict=False); model.eval()
    if args.no_residual:
        model.cell.use_residual = False
    model.cell.res_mode = ck.get("res_mode", "mean")
    corpus = read_books(); rng = random.Random(999)
    nblocks = args.T // ck["S"]
    rec = 0; n = 8
    for _ in range(n):
        ids, qids, ans, gold, dg = make_sample(corpus, tok, ck["S"], args.T, rng, nblocks)
        x = torch.tensor([ids], device="cuda")
        feats, blocks = model.blocks(x)
        q = model.token_q(qids)
        rb = blocks if args.route_only else blocks.detach()
        levels = model.build_tree(rb, q)
        idx = 0
        for li in range(len(levels) - 1, -1, -1):
            logits = levels[li][1][idx]
            c = int(logits.argmax())
            idx = idx * 2 + c
        rec += int(idx == gold)
    print(f"extrap T={args.T} route_recall={rec/n:.3f}", flush=True)


def train_readout(args):
    corpus = read_books()
    tok = CharTok(build_vocab(corpus))
    ck = torch.load(os.path.join(ROOT, "state", args.init), map_location="cpu", weights_only=False)
    model = SharedTree(len(tok.itos), d=ck["d"], S=ck["S"], ffn=ck["ffn"]).to("cuda")
    model.load_state_dict(ck["state"], strict=False)
    for p in model.parameters():
        p.requires_grad = False
    for name, p in model.named_parameters():
        if name.startswith(("Wq", "Wk", "Wv", "Wo", "ln1", "ln2", "gate", "up", "down", "ln_f", "pos", "readout_head")):
            p.requires_grad = True
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, betas=(0.9, 0.95))
    rng = random.Random(1)
    model.train()
    for step in range(1, args.steps + 1):
        opt.zero_grad(set_to_none=True)
        tot = 0.0
        for _a in range(4):
            T = rng.choice([16384, 32768, 65536])
            n = T // args.S
            target = rng.randrange(n)
            ids, qids, ans, gold, dg = make_sample_at(corpus, tok, args.S, T, rng, target)
            x = torch.tensor([ids], device="cuda")
            with torch.no_grad():
                feats, blocks = model.blocks(x)
            s0 = gold * args.S
            blk = feats[0, s0:s0 + args.S]
            qe = model.embed(torch.tensor([qids], device="cuda"))[0]
            ae = model.embed(torch.tensor([ans], device="cuda"))[0]
            seq = torch.cat([blk, qe, ae], 0).unsqueeze(0)
            lg = model.token_forward(seq)
            y = torch.full((1, lg.shape[1]), -100, device="cuda")
            y[0, blk.shape[0] + qe.shape[0]:] = torch.tensor([ans], device="cuda")
            loss = chunked_ce(lg, y)
            (loss / 4).backward()
            tot += float(loss)
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        opt.step()
        if step % 100 == 0:
            print(f"step {step} readout_loss={tot/4:.3f}", flush=True)
    torch.save({"state": model.state_dict(), "vocab": tok.itos, "S": args.S,
                "T": args.T, "d": ck["d"], "ffn": ck["ffn"]}, os.path.join(ROOT, "state", "sharedtree_rd_readout.pt"))


@torch.no_grad()
def eval_copy(args):
    ck = torch.load(os.path.join(ROOT, "state", args.ckpt), map_location="cpu", weights_only=False)
    tok = CharTok(ck["vocab"])
    model = SharedTree(len(tok.itos), d=ck["d"], S=ck["S"], ffn=ck["ffn"]).to("cuda")
    model.load_state_dict(ck["state"], strict=False); model.eval()
    corpus = read_books(); rng = random.Random(999)
    n = 16
    for T in args.Ts:
        nb = T // ck["S"]
        ok = rec = 0
        for _ in range(n):
            target = rng.randrange(nb)
            ids, qids, ans, gold, dg = make_sample_at(corpus, tok, ck["S"], T, rng, target)
            x = torch.tensor([ids], device="cuda")
            feats, blocks = model.blocks(x)
            q = model.token_q(qids)
            levels = model.build_tree(rb, q)
            idx = 0
            for li in range(len(levels) - 1, -1, -1):
                logits = levels[li][1][idx]
                c = int(logits.argmax())
                idx = idx * 2 + c
            rec += int(idx == gold)
            s0 = idx * ck["S"]
            seq = torch.cat([feats[0, s0:s0 + ck["S"]], model.embed(torch.tensor([qids], device="cuda"))[0]], 0).unsqueeze(0)
            out = []
            for _g in range(4):
                lg = model.token_forward(seq)
                t = int(lg[0, -1].argmax())
                out.append(t)
                seq = torch.cat([seq, model.embed(torch.tensor([[t]], device="cuda"))], 1)
            ok += int(tok.decode(out) == dg)
        print(f"T={T} acc={ok/n:.3f} recall={rec/n:.3f}", flush=True)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["route", "train", "eval", "extrap", "rdepth", "rdepth3", "readout", "evalcopy"], required=True)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--T", type=int, default=16384)
    ap.add_argument("--S", type=int, default=8192)
    ap.add_argument("--d", type=int, default=256)
    ap.add_argument("--ffn", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--ckpt", default="sharedtree16_full.pt")
    ap.add_argument("--route-only", action="store_true")
    ap.add_argument("--freeze-route", action="store_true")
    ap.add_argument("--init", default="")
    ap.add_argument("--joint", action="store_true")
    ap.add_argument("--Ts", type=int, nargs="+", default=[16384, 32768, 65536])
    ap.add_argument("--save", default="sharedtree_rd3_res.pt")
    ap.add_argument("--no-residual", action="store_true")
    ap.add_argument("--res-mode", default="mean")
    args = ap.parse_args()
    if args.mode == "route":
        args.route_only = True
        train(args)
    elif args.mode == "train":
        train(args)
    elif args.mode == "rdepth":
        train_rdepth(args)
    elif args.mode == "rdepth3":
        args.route_only = True
        train_rdepth3(args)
    elif args.mode == "readout":
        train_readout(args)
    elif args.mode == "evalcopy":
        eval_copy(args)
    elif args.mode == "extrap":
        extrapolate(args)
    else:
        evaluate(args)


if __name__ == "__main__":
    main()

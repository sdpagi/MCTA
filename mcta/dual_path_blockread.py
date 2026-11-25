# -*- coding: utf-8 -*-
"""双路径：共享 rdepth 路由 + v2 读出，只编码选中的 8K 块。"""
import os, random, sys, torch
import torch.nn.functional as F
ROOT=r"D:\NewComputerAI\ZERO"; sys.path.insert(0,ROOT)
from mcta.sentence_memory_m1 import CharTok, read_books, build_vocab
from mcta.tree_shared_cell import SharedTree, make_sample_at
from mcta.tree_agg_16k_v2 import TreeAgg16V2

def encode_block(v2, block_ids):
    x = v2.embed(torch.tensor([block_ids], device="cuda"))
    xt = x.transpose(1, 2)
    xt = xt + F.silu(v2.tok_pw(v2.tok_dw(xt)))
    for c in v2.ctx:
        xt = xt + F.silu(c(xt))
    xt = xt + F.silu(v2.ctx_pw(xt))
    return xt.transpose(1, 2)

def main():
    tok=CharTok(torch.load(os.path.join(ROOT,"state","sharedtree_rd3.pt"),map_location="cpu",weights_only=False)["vocab"]); corpus=read_books()
    rck=torch.load(os.path.join(ROOT,"state","sharedtree_rd3.pt"),map_location="cpu",weights_only=False)
    route=SharedTree(len(tok.itos),d=rck["d"],S=rck["S"],ffn=rck["ffn"]).to("cuda"); route.load_state_dict(rck["state"],strict=False); route.eval()
    vck=torch.load(os.path.join(ROOT,"state","treeagg16v2_full.pt"),map_location="cpu",weights_only=False)
    read=TreeAgg16V2(len(tok.itos),d=vck["d"],S=vck["S"],ffn=vck["ffn"]).to("cuda"); read.load_state_dict(vck["state"],strict=False); read.eval()
    rng=random.Random(999)
    with torch.no_grad():
      for T in [1048576,2097152,4194304]:
        nb=T//8192; ok=rec=0; n=4
        for _ in range(n):
            target=rng.randrange(nb)
            ids,qids,ans,gold,dg=make_sample_at(corpus,tok,8192,T,rng,target)
            x=torch.tensor([ids],device="cuda")
            rfeats,rblocks=route.blocks(x); q=route.token_q(qids)
            levels=route.build_tree(rblocks.detach(),q); idx=0
            for li in range(len(levels)-1,-1,-1):
                c=int(levels[li][1][idx].argmax()); idx=idx*2+c
            rec+=int(idx==gold)
            blk_ids=ids[idx*8192:(idx+1)*8192]
            bfeat=encode_block(read,blk_ids)
            seq=torch.cat([bfeat[0],read.embed(torch.tensor([qids],device="cuda"))[0]],0).unsqueeze(0)
            out=[]
            for _g in range(4):
                lg=read.token_forward(seq); t=int(lg[0,-1].argmax()); out.append(t)
                seq=torch.cat([seq,read.embed(torch.tensor([[t]],device="cuda"))],1)
            ok+=int(tok.decode(out)==dg)
        print("T",T,"acc",ok/n,"recall",rec/n,flush=True)

if __name__=="__main__": main()

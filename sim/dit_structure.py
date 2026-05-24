#!/usr/bin/env python3
"""
dit_structure.py - apply the Ditto performance model to DiT-XL/2 (workload swap test).

The paper also evaluates DiT. Our whole Phase-2 analysis (MAC enumeration, attention
difference, memory ratio, speedup roofline, energy six-segments) is meant to be
workload-agnostic; this script swaps SDM UNet -> DiT-XL/2 and re-runs the structural
analysis to (1) check the model really generalizes and (2) cross-check against the
paper's DiT numbers. This needs only a model load + one forward for layer shapes --
NO 19GB trace (bit-width distribution, which needs activations, is left for a later
DiT trace-collection pass).

DiT-XL/2: 28 blocks, hidden 1152, 16 heads, head_dim 72, patch 2, latent 32x32
-> sequence length 256 tokens. Pure self-attention (ada_norm_zero injects the
timestep/class condition; there is NO cross-attention, unlike SDM).

    cd ~/Ditto && python3 sim/dit_structure.py
"""
from __future__ import annotations

import torch
import torch.nn as nn
from diffusers import DiTTransformer2DModel

# reuse the analytical models already validated on SDM
import fig13_roofline as RF      # comp_*, mem_bytes, run_bw
import energy_model as EN        # total_energy, compute_flip, etotal

N_PE_ITC = 27648
N_PE_DITTO = 39398


def enumerate_dit():
    """Load DiT-XL/2, run one forward to capture layer shapes, return the same
    layer-dict format used by the SDM analysis (cat, macs, A, B, C)."""
    m = DiTTransformer2DModel.from_pretrained(
        "facebook/DiT-XL-2-256", subfolder="transformer").eval()
    shapes, attn = {}, []

    def lin_hook(nm):
        def h(mod, inp, out):
            y = out[0] if isinstance(out, (tuple, list)) else out
            shapes[nm] = (tuple(inp[0].shape), tuple(y.shape))
        return h

    def attn_hook(mod):
        def h(a_mod, args, kwargs, out):
            hs = args[0] if args else kwargs.get("hidden_states")
            ctx = kwargs.get("encoder_hidden_states")
            is_cross = ctx is not None
            sq = hs.shape[1]
            sk = ctx.shape[1] if is_cross else hs.shape[1]
            heads = getattr(a_mod, "heads", 16)
            hd = a_mod.to_q.out_features // heads
            attn.append((is_cross, hs.shape[0], heads, sq, sk, hd))
        return h

    lh = [mm.register_forward_hook(lin_hook(n))
          for n, mm in m.named_modules() if isinstance(mm, nn.Linear)]
    ah = [mm.register_forward_hook(attn_hook(mm), with_kwargs=True)
          for _, mm in m.named_modules() if type(mm).__name__ == "Attention"]

    # DiT forward: latent (B,4,32,32), timestep, class_labels
    with torch.no_grad():
        m(torch.randn(1, 4, 32, 32), timestep=torch.tensor([1]),
          class_labels=torch.tensor([0]))
    for h in lh + ah:
        h.remove()

    layers = []
    for n, (in_s, out_s) in shapes.items():
        # all DiT Linears: M = product of leading dims, K = in_features, N = out
        M = 1
        for d in in_s[:-1]:
            M *= d
        K, N = in_s[-1], out_s[-1]
        layers.append({"cat": "linear", "macs": float(M * K * N),
                       "A": float(M * K), "B": float(K * N), "C": float(M * N)})
    for is_cross, Bsz, heads, sq, sk, hd in attn:
        cat = "attn_cross" if is_cross else "attn_self"
        for (M, K, N) in [(sq, hd, sk), (sq, sk, hd)]:
            layers.append({"cat": cat, "macs": float(Bsz * heads * M * K * N),
                           "A": float(Bsz * heads * M * K),
                           "B": float(Bsz * heads * K * N),
                           "C": float(Bsz * heads * M * N)})
    return layers


def main():
    print("Loading DiT-XL/2 and enumerating layers (one forward) ...", flush=True)
    layers = enumerate_dit()

    lin = [L for L in layers if L["cat"] == "linear"]
    aself = [L for L in layers if L["cat"] == "attn_self"]
    across = [L for L in layers if L["cat"] == "attn_cross"]
    mac_lin = sum(L["macs"] for L in lin) / 1e9
    mac_self = sum(L["macs"] for L in aself) / 1e9
    mac_cross = sum(L["macs"] for L in across) / 1e9
    total = mac_lin + mac_self + mac_cross
    print(f"\n=== DiT-XL/2 structure (vs SDM UNet) ===")
    print(f"  linear/proj MACs : {mac_lin:7.2f} G")
    print(f"  self-attn QK/PV  : {mac_self:7.2f} G")
    print(f"  cross-attn QK/PV : {mac_cross:7.2f} G  (DiT has none -> conditioning via ada_norm)")
    print(f"  TOTAL            : {total:7.2f} G")
    attn_frac = (mac_self + mac_cross) / total * 100
    print(f"  attention fraction: {attn_frac:.1f}%  (SDM was ~15.7%)")

    # speedup roofline (reuse validated model)
    print("\n=== speedup roofline (reusing fig13_roofline model) ===")
    print(f"  {'BW(B/cyc)':>10} {'diff-only':>10} {'+Defo':>8} {'flip%':>7}")
    for bw in [64, 256, 1024, 4096]:
        r = RF.run_bw(layers, bw)
        print(f"  {bw:>10} {r['diff_only']:>9.2f}x {r['defo']:>7.2f}x {r['flip']*100:>6.1f}%")
    ceil = RF.run_bw(layers, 1e12)
    print(f"  compute ceiling (BW->inf): {ceil['defo']:.2f}x")

    # energy six-segment (reuse validated model)
    print("\n=== energy six-segment (reusing energy_model) ===")
    itc = EN.total_energy(layers, "itc")
    camd = EN.total_energy(layers, "camd")
    flip = EN.compute_flip(layers)
    defo = EN.total_energy(layers, "ditto", defo_flip=flip)
    it = EN.etotal(itc)
    segs = ["core", "sram", "dram", "eu", "vpu", "defo"]
    print("  " + " ".join(f"{s:>7}" for s in segs) + f" {'TOTAL':>8}")
    for name, e in [("ITC ", itc), ("Cam-D", camd), ("Ditto", defo)]:
        print("  " + " ".join(f"{e[s]/it:>7.3f}" for s in segs)
              + f" {EN.etotal(e)/it:>8.3f}   {name}")
    print(f"  Ditto saving: {(1-EN.etotal(defo)/it)*100:.1f}%   "
          f"Cam-D: {EN.etotal(camd)/it:.2f}x   flip {len(flip)}/{len(layers)}")

    print("\nNote: structural analysis only (model load + one forward). DiT bit-width")
    print("distribution (Fig 5) needs a DiT activation trace -> later Phase-1 pass.")
    print("attention fraction and ceiling are the key DiT-vs-SDM contrasts to report.")


if __name__ == "__main__":
    main()

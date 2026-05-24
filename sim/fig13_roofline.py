#!/usr/bin/env python3
"""
fig13_roofline.py - speedup as an explicit bandwidth sweep (honest about unknown DRAM).

The paper does NOT publish Ditto's DRAM, and every cycle-accurate attempt
(ReadWriteTrace serial, multi-channel, HBM, SimpleO3 bubble) either degenerated
or introduced artifacts -- because faithfully simulating this accelerator's
PARALLEL memory needs a config the paper never gave. So instead of hiding an
assumed bandwidth inside a simulator, we make bandwidth the EXPLICIT sweep axis:

    layer_cycle(mode) = max( compute_cycle(mode), mem_bytes(mode) / BW )

where BW (bytes/cycle) is swept. This is a roofline: concurrency is the bandwidth
ceiling itself, so no serial/bubble artifacts. We then show speedup vs BW and
mark where Ditto reaches the paper's 1.5x.

Compute models (validated): itc=macs/27648, act=macs/(39398*2),
diff=macs*NONZERO*BITF/(39398*4) (x2 sub-ops for self-attention).
Memory: act reads A+B+C; diff adds prev; self-attn diff carries full Q/K prev.
Attention included (total 401.6 G), so this is consistent with the 9.03x ceiling
and 2.46x memory ratio.

    cd ~/Ditto && python3 sim/fig13_roofline.py
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn
from diffusers import UNet2DConditionModel

N_PE_ITC = 27648
N_PE_DITTO = 39398
LANES = 4
BW_ZERO, BW_LE4, BW_GT4 = 0.459, 0.534, 0.007
NONZERO = BW_LE4 + BW_GT4
BITF = (BW_LE4 + 2 * BW_GT4) / NONZERO
RESULT_DIR = Path.home() / "Ditto" / "results"

# --- EU / VPU cycle handling (kept consistent with energy_model.py) ---
# EU (Encoding Unit) is the pipeline stage BEFORE the PE array (Fig 11): it encodes
# the next tile while the PEs compute the current one, so in steady state EU is
# OVERLAPPED by PE compute and adds NO cycles. (Documented assumption.)
# VPU runs nonlinear ops on a separate unit AFTER the GEMM (nonlinearity is not in
# the difference domain), so its cycles are SERIAL (added, not overlapped). diff/
# camd additionally pay a restore (accumulate-to-full-activation) before nonlinear.
VPU_THRU = N_PE_DITTO            # VPU lanes ~ PE width (assume same throughput)
VPU_FRAC = 0.5                   # fraction of outputs feeding a nonlinearity
def vpu_cycles(L, mode):
    nonlin = L["C"] * VPU_FRAC / VPU_THRU
    restore = L["C"] * VPU_FRAC / VPU_THRU if mode in ("diff", "camd") else 0.0
    return nonlin + restore


def comp_itc(m):
    return m / N_PE_ITC


def comp_act(m):
    return m / (N_PE_DITTO * 2)


def comp_diff(m, n_sub=1):
    return n_sub * m * NONZERO * BITF / (N_PE_DITTO * LANES)


def mem_bytes(L, mode):
    A, B, C = L["A"], L["B"], L["C"]
    if L["cat"] == "attn_self" and mode == "diff":
        return A + B + C + A + B          # deltas + full prev Q,K operands
    if mode == "diff":
        return B + A + C + A + C          # W + in + out + prev_in + prev_out
    return B + A + C                      # act / itc


def enumerate_with_attention():
    unet = UNet2DConditionModel.from_pretrained(
        "CompVis/stable-diffusion-v1-4", subfolder="unet").eval()
    shapes, attn = {}, []

    def lin_hook(nm):
        def h(m, inp, out):
            y = out[0] if isinstance(out, (tuple, list)) else out
            shapes[nm] = (tuple(inp[0].shape), tuple(y.shape))
        return h

    def attn_hook(mod):
        def h(m, args, kwargs, out):
            hs = args[0] if args else kwargs.get("hidden_states")
            ctx = args[1] if (len(args) > 1 and torch.is_tensor(args[1])) else None
            if kwargs.get("encoder_hidden_states") is not None:
                ctx = kwargs["encoder_hidden_states"]
            is_cross = ctx is not None
            sq = hs.shape[1]
            sk = ctx.shape[1] if is_cross else hs.shape[1]
            heads = getattr(m, "heads", 8)
            dim = m.to_q.out_features
            hd = max(1, dim // heads)
            attn.append((is_cross, hs.shape[0], heads, sq, sk, hd))
        return h

    lh = [m.register_forward_hook(lin_hook(n)) for n, m in unet.named_modules()
          if isinstance(m, (nn.Conv2d, nn.Linear))]
    ah = [m.register_forward_hook(attn_hook(m), with_kwargs=True)
          for _, m in unet.named_modules() if type(m).__name__ == "Attention"]
    with torch.no_grad():
        unet(torch.randn(1, 4, 64, 64), torch.tensor(1), torch.randn(1, 77, 768))
    for h in lh + ah:
        h.remove()

    mods = dict(unet.named_modules())
    layers = []
    for n, (in_s, out_s) in shapes.items():
        mod = mods[n]
        if isinstance(mod, nn.Conv2d):
            M = in_s[0] * out_s[2] * out_s[3]
            K = (in_s[1] // mod.groups) * mod.kernel_size[0] * mod.kernel_size[1]
            N = out_s[1]
        else:
            M = 1
            for d in in_s[:-1]:
                M *= d
            K, N = mod.in_features, mod.out_features
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


def run_bw(layers, bw):
    ti = td = tdefo = 0.0
    nflip = 0
    for L in layers:
        n_sub = 2 if L["cat"] == "attn_self" else 1
        ma = mem_bytes(L, "act")
        md = mem_bytes(L, "diff")
        # roofline core: max(compute, memory); EU overlaps (no term); VPU serial (added)
        ci = max(comp_itc(L["macs"]), ma / bw) + vpu_cycles(L, "itc")
        cd = max(comp_diff(L["macs"], n_sub), md / bw) + vpu_cycles(L, "diff")
        ca = max(comp_act(L["macs"]), ma / bw) + vpu_cycles(L, "act")
        ti += ci
        td += cd
        if ca < cd:
            tdefo += ca
            nflip += 1
        else:
            tdefo += cd
    return {"bw": bw, "diff_only": ti / td if td else 0,
            "defo": ti / tdefo if tdefo else 0,
            "flip": nflip / len(layers) if layers else 0}


def main():
    print("Enumerating UNet (with attention) ...", flush=True)
    layers = enumerate_with_attention()
    total = sum(L["macs"] for L in layers) / 1e9
    print(f"{len(layers)} layers/matmuls, {total:.1f} G MACs\n")

    # bytes/cycle at 1 GHz: 1 B/cyc = 1 GB/s. Sweep 16 .. 4096 B/cyc (~16GB/s..4TB/s)
    bws = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
    print(f"{'BW(B/cyc)':>10} {'~GB/s':>8} {'diff-only':>11} {'+Defo':>9} {'flip%':>8}")
    print("-" * 50)
    rows = []
    for bw in bws:
        r = run_bw(layers, bw)
        rows.append(r)
        print(f"{bw:>10} {bw:>8} {r['diff_only']:>10.2f}x {r['defo']:>8.2f}x "
              f"{r['flip']*100:>7.1f}%")

    # find BW where Defo speedup first reaches paper's 1.5x
    reach = next((r for r in rows if r["defo"] >= 1.5), None)
    ceiling = run_bw(layers, 1e12)
    print(f"\nCompute-bound ceiling (BW->inf): diff-only {ceiling['diff_only']:.2f}x, "
          f"+Defo {ceiling['defo']:.2f}x")
    if reach:
        print(f"Ditto reaches paper's 1.5x speedup at BW >= {reach['bw']} B/cyc "
              f"(~{reach['bw']} GB/s at 1GHz), flip {reach['flip']*100:.0f}%")
    else:
        print("Defo speedup stays below 1.5x across the swept range "
              "(memory-bound region) -- higher BW needed.")
    print("\nHonest note: the paper does not publish Ditto's DRAM. Bandwidth is the")
    print("explicit sweep axis here, not a fitted value. The curve shows WHAT")
    print("bandwidth Ditto needs to hit 1.5x, rather than asserting a single number.")

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULT_DIR / "fig13_roofline.json", "w") as f:
        json.dump({"total_macs_giga": total, "rows": rows,
                   "ceiling_defo": ceiling["defo"]}, f, indent=2)
    print(f"\nSaved: {RESULT_DIR / 'fig13_roofline.json'}")


if __name__ == "__main__":
    main()

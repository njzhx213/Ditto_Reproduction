#!/usr/bin/env python3
"""
fig13_attn.py - Stage 1 of bringing attention into the MAIN line (analytical mem).

Extends the cycle model so the speedup / Defo / memory numbers INCLUDE the
attention QK/PV matmuls that the Linear-only enumeration missed (probe: 63 G =
15.7% of true total, 61 G self-attention). Uses the analytical tiling memory
model (not Ramulator yet -- that is Stage 3). Supersedes the linear-only
fig13_speedup.py as the attention-complete analytical main line.

Per-layer compute (paper):
  linear/conv/attn_cross : 1 difference sub-op  -> comp_diff
  attn_self (Q*K, P*V)   : 2 difference sub-ops -> 2 * comp_diff
ITC baseline always one full 8-bit GEMM.

Per-layer memory (analytical tiling, bytes):
  We work in element counts (int8). For a GEMM M x K x N:
    A = M*K (input act), B = K*N, C = M*N (output)
  Linear/Conv  act : W(=B) + in(=A) + out(=C)
               diff: + prev_in + prev_out
  attn_self    : here BOTH operands are step-varying activations. The two sub-ops
    Q_t.dK + dQ.K_{t+1} must read Q_t and K_{t+1} (previous-step activations acting
    as 'weights') in addition to the deltas. So diff working set carries the full
    Q and K matrices as prev, not just small prev_in/out. Modeled below.
  attn_cross   : K,V constant across steps -> like linear diff (one prev set).

Honesty: the attn_self memory term treats Q_t/K_{t+1} as full re-read prev
operands (mechanism-based; paper gives no per-op attention byte counts). It is a
first-order analytical model; Stage 3 will feed the same access pattern to
Ramulator.

    cd ~/Ditto && python3 sim/fig13_attn.py
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


def comp_itc(macs):
    return macs / N_PE_ITC


def comp_diff(macs, n_subop=1):
    return n_subop * macs * NONZERO * BITF / (N_PE_DITTO * LANES)


def comp_act_ditto(macs):
    return macs / (N_PE_DITTO * 2)


def reload_factor(ws, buf):
    return max(1.0, ws / buf)


def layer_bytes(L, mode):
    """Element counts for act vs diff working set, by category."""
    A, B, C = L["A"], L["B"], L["C"]   # in, weight/operand, out
    if L["cat"] == "attn_self":
        # both operands vary; diff reads dQ,dK (~A,B sized) + prev Q_t,K_{t+1} (~A,B)
        if mode == "diff":
            return A + B + C + A + B          # deltas + full prev operands
        return A + B + C
    else:
        # linear/conv/attn_cross: weight fixed (or context const); diff adds prev in/out
        if mode == "diff":
            return B + A + C + A + C          # W + in + out + prev_in + prev_out
        return B + A + C


def layer_cycles(L, bpc, buf):
    n_sub = 2 if L["cat"] == "attn_self" else 1
    eff_act = layer_bytes(L, "act") * reload_factor(layer_bytes(L, "act"), buf)
    eff_diff = layer_bytes(L, "diff") * reload_factor(layer_bytes(L, "diff"), buf)
    cyc_itc = max(comp_itc(L["macs"]), eff_act / bpc)
    cyc_diff = max(comp_diff(L["macs"], n_sub), eff_diff / bpc)
    cyc_act = max(comp_act_ditto(L["macs"]), eff_act / bpc)
    return cyc_itc, cyc_diff, cyc_act, eff_act, eff_diff


def run_at(layers, bpc, buf):
    ti = td = tdefo = 0.0
    ea = ed = edefo = 0.0
    nflip = 0
    for L in layers:
        ci, cd, ca, e_a, e_d = layer_cycles(L, bpc, buf)
        ti += ci
        td += cd
        ea += e_a
        ed += e_d
        if ca < cd:
            tdefo += ca
            edefo += e_a
            nflip += 1
        else:
            tdefo += cd
            edefo += e_d
    return {
        "bpc": bpc, "speedup_diff_only": ti / td if td else 0,
        "speedup_defo": ti / tdefo if tdefo else 0,
        "flip_frac": nflip / len(layers) if layers else 0,
        "mem_ratio_bare": ed / ea if ea else 0,
        "mem_ratio_defo": edefo / ea if ea else 0,
    }


def enumerate_with_attention():
    unet = UNet2DConditionModel.from_pretrained(
        "CompVis/stable-diffusion-v1-4", subfolder="unet").eval()
    shapes = {}
    attn = []

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
            Bsz = hs.shape[0]
            attn.append((is_cross, Bsz, heads, sq, sk, hd))
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
        # QK: M=sq, K=hd, N=sk ; PV: M=sq, K=sk, N=hd  (per head x heads x batch)
        for (M, K, N) in [(sq, hd, sk), (sq, sk, hd)]:
            macs = Bsz * heads * M * K * N
            layers.append({"cat": cat, "macs": float(macs),
                           "A": float(Bsz * heads * M * K),
                           "B": float(Bsz * heads * K * N),
                           "C": float(Bsz * heads * M * N)})
    return layers


def main():
    print("Enumerating UNet WITH attention QK/PV ...", flush=True)
    layers = enumerate_with_attention()
    total = sum(L["macs"] for L in layers) / 1e9
    n_self = sum(1 for L in layers if L["cat"] == "attn_self")
    n_cross = sum(1 for L in layers if L["cat"] == "attn_cross")
    n_lin = sum(1 for L in layers if L["cat"] == "linear")
    print(f"Layers: {n_lin} linear/conv + {n_self} self-attn + {n_cross} cross-attn "
          f"matmuls = {len(layers)}")
    print(f"Total MACs: {total:.1f} G (linear-only was 338.6 G)\n")

    # buffer fixed at a representative on-chip size; sweep bandwidth
    buf = int(0.25 * 1024 * 1024)
    bpcs = [8, 32, 128, 512, 2048, 8192]
    print(f"{'BYTES/cyc':>10} {'diff-only':>11} {'+Defo':>9} {'flip%':>8} "
          f"{'bare-mem':>9} {'defo-mem':>9}")
    print("-" * 62)
    rows = []
    for b in bpcs:
        r = run_at(layers, b, buf)
        rows.append(r)
        print(f"{b:>10} {r['speedup_diff_only']:>10.2f}x {r['speedup_defo']:>8.2f}x "
              f"{r['flip_frac']*100:>7.1f}% {r['mem_ratio_bare']:>8.2f}x "
              f"{r['mem_ratio_defo']:>8.2f}x")

    # attention-aware bare compute ceiling (huge bandwidth -> compute bound)
    r_hi = run_at(layers, 1e15, buf)
    print(f"\nAttention-aware bare compute ceiling: {r_hi['speedup_diff_only']:.2f}x "
          f"(linear-only was 10.40x; attention_compute.py got 9.03x)")
    print(f"Bare memory ratio incl. attention: {rows[0]['mem_ratio_bare']:.2f}x")

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULT_DIR / "fig13_attn.json", "w") as f:
        json.dump({"total_macs_giga": total, "rows": rows,
                   "ceiling": r_hi["speedup_diff_only"]}, f, indent=2)
    print(f"\nSaved: {RESULT_DIR / 'fig13_attn.json'}")


if __name__ == "__main__":
    main()

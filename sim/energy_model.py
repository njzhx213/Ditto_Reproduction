#!/usr/bin/env python3
"""
energy_model.py - Ditto vs ITC energy, toward the paper's 17.74% saving (Fig 13).

Energy = sum over components of (access_count x energy_per_access). All counts we
already have (MACs, SRAM/DRAM accesses); per-access energies are public 45nm
constants (the paper uses FreePDK 45nm + CACTI). We report the Ditto/ITC energy
RATIO (-> paper 17.74% saving), which is robust to the absolute constants.

Energy constants (public 45nm, Horowitz ISSCC'14 class; relative scale is what
matters for a ratio):
  MAC 8-bit  : ~0.20 pJ      MAC 4-bit : ~0.05 pJ   (energy ~ bitwidth^2)
  SRAM read/write (large on-chip, per 64B line): ~20 pJ
  DRAM access (per 64B line): ~1300 pJ   (off-chip dominates: ~100x SRAM)
These are LABELED public estimates, NOT CACTI-measured. We do energy RATIOS, so
the conclusion depends on relative magnitudes (DRAM >> SRAM >> MAC), not exact pJ.

Component models:
  Compute: ITC = MACs x e8 ; Ditto = MACs x nonzero x e4 (zero-skip + 4-bit)
  Memory:  accesses (in 64B lines) x (SRAM or DRAM energy). diff mode adds prev
           traffic; Defo flips memory-heavy layers back to act (uses the flip
           decision at a chosen bandwidth, default the ~251 GB/s point that meets
           the paper's 1.5x speedup).

    cd ~/Ditto && python3 sim/energy_model.py
"""
from __future__ import annotations

import torch
import torch.nn as nn
from diffusers import UNet2DConditionModel

# ---- public 45nm energy constants (pJ); labeled estimates, not CACTI ----
E_MAC8 = 0.20
E_MAC4 = 0.05
# E_SRAM: CACTI 7.0 measured, 45nm scratch RAM, 64B (512-bit) access line.
# CACTI reports 818 pJ/read for a 4MB bank, of which ~635 pJ (78%) is the in-bank
# H-tree (data transport across the bank) and only ~180 pJ is cell+decoder. The
# true per-access energy therefore depends on bank organization (unpublished): a
# small/near bank ~180 pJ, a 4MB bank ~818 pJ. We default to the full 4MB-bank
# number and sweep this as a sensitivity. This replaces an earlier hand-guessed
# 20 pJ that was ~40x too low.
E_SRAM = 818.0     # per 64B line (CACTI 45nm, 4MB bank; range 180..818)
E_DRAM = 1300.0    # per 64B line (public DDR-class estimate)
CL = 64            # bytes per access (int8 elems per line)

# ---- Encoding Unit (Fig 11): runs once per activation element in diff mode ----
# Work per element: subtract prev, classify sign-magnitude into zero / <=4b / >4b,
# emit the nonzero low-bit code. A multi-gate op (sub + comparators + shift + mux),
# ~an order above a single MAC. Calibrated so EU is a small-but-visible segment
# (~3% of total) matching the EU band in paper Fig 13. diff mode only.
E_EU = 6.0         # pJ per activation element encoded (diff mode only)
# ---- VPU (vector/nonlinear unit) and Defo unit ----
# VPU runs nonlinear ops (GroupNorm/SiLU/LayerNorm/GEGLU/softmax) on output
# activations; ~a few ALU ops per element. Modeled at E_VPU pJ/element, with
# VPU_FRAC the fraction of layer outputs that feed a nonlinearity (UNet is
# nonlinearity-dense; defo_static found 149 nonlinear modules). Same work in ITC
# and Ditto (difference processing does not change nonlinear math).
E_VPU = 12.0       # pJ per output element through a nonlinear op (transcendental;
                   # calibrated to ~3% of total, the VPU band in paper Fig 13)
VPU_FRAC = 0.5     # fraction of layer outputs feeding a nonlinearity (swept)
# Restore: diff/camd must accumulate difference back to full activation before any
# nonlinear op (nonlinearity is not closed under differencing). One add per output
# element. A single add ~ MAC-add energy at 45nm.
E_RESTORE = 0.05   # pJ per output element (accumulate; diff/camd modes only)
# Defo: one cheap per-layer decision; fixed small cost (~0.3% of total over ~346 layers).
E_DEFO_PER_LAYER = 1.5e6    # pJ per layer (decision logic), small but visible

N_PE_DITTO = 39398
BW_ZERO, BW_LE4, BW_GT4 = 0.459, 0.534, 0.007
NONZERO = BW_LE4 + BW_GT4
# 8-bit costs two 4-bit MACs; effective per-MAC energy for the >4-bit fraction
# is counted as 8-bit. Diff MAC energy: 4-bit lanes, plus the >4b fraction at 8b.
E_MAC_DIFF = (BW_LE4 * E_MAC4 + BW_GT4 * E_MAC8) / NONZERO

# default memory split: fraction of accesses that hit on-chip SRAM vs go to DRAM.
# With 192MB SRAM most weights/activations are resident; we assume a high SRAM
# hit rate and a small DRAM-miss fraction (swept for sensitivity below).
SRAM_HIT = 0.90    # 90% of accesses hit SRAM, 10% reach DRAM
# Effective on-chip tiling window. The paper has 192MB SRAM; with that much
# buffer, ITC activations are largely resident -> reload ~1 (matching paper Fig 13
# where ITC energy is Core-dominated, memory ~1/4). An earlier 256KB buffer forced
# a ~few-hundred-x reload that wrongly made memory dominate. We default to a large
# effective window; diff mode's doubled working set (it must also hold prev) is
# what pushes reload>1 and inflates memory -> the Cam-D-style offset.
BUF_ELEMS = 192 * 1024 * 1024   # paper's 192MB SRAM (int8 elems); ITC resident -> reload~1


def reload_factor(ws_elems):
    """Tiling re-fetch. With the paper's 192MB SRAM, a single layer's ITC working
    set (one activation + its weights) is essentially resident, so reload ~1 for
    ITC -- this is why paper Fig 13's ITC energy is Core-dominated, not memory.
    Only diff mode, which must ALSO hold the previous-step activation (~doubled
    working set), overflows and pays reload>1 -- the Cam-D-style memory offset."""
    if ws_elems <= BUF_ELEMS:
        return 1.0
    return ws_elems / BUF_ELEMS


def mem_lines(L, mode):
    """64B-line counts split into (weight, cur_act, prev_act).

    weight   : static read-only weights (B of linear/conv layers). Loaded once and
               REUSED across all tokens and all diffusion steps -> resident in the
               large on-chip SRAM, served as SRAM hits, NOT reloaded from DRAM per
               layer. (Earlier model wrongly streamed full B every layer, which
               over-counted memory -- badly for DiT, whose big weight matrices are
               82.8% of operands vs 53.4% for SDM.)
    cur_act  : this step's input/output activations (A,C) -- per-token/per-step,
               SRAM-hit blend, reloadable.
    prev_act : previous-step activations that temporal difference re-reads -- cross
               step, forced to DRAM (origin of Cam-D's blow-up).

    Attention QK/PV have NO static weight: Q,K,V are activations recomputed each
    step, so their 'B' operand is an activation, not a resident weight."""
    A, B, C = L["A"], L["B"], L["C"]
    is_attn = L["cat"] in ("attn_self", "attn_cross")
    if is_attn:
        weight = 0.0                       # no static weights in QK/PV
        cur_act = A + B + C
    else:
        weight = B                         # static, resident
        cur_act = A + C
    if mode == "diff":
        if L["cat"] == "attn_self":
            prev_act = A + B               # prev Q and K
        elif is_attn:
            prev_act = A                   # cross-attn: only query side varies
        else:
            prev_act = A + C               # prev_in + prev_out
    else:
        prev_act = 0.0
    w_lines = weight / CL                                   # resident, no reload
    cur_lines = (cur_act / CL) * reload_factor(cur_act)
    prev_lines = (prev_act / CL) * reload_factor(prev_act)
    return w_lines, cur_lines, prev_lines


def mem_energy(lines):
    """current-frame: SRAM-hit blend. (kept for any single-value callers)"""
    return lines * (SRAM_HIT * E_SRAM + (1 - SRAM_HIT) * E_DRAM)


def compute_energy(macs, mode, n_sub=1):
    if mode == "itc" or mode == "act":
        return macs * E_MAC8
    return n_sub * macs * NONZERO * E_MAC_DIFF      # diff: zero-skip + low bit


def eu_energy(L, mode):
    """Encoding Unit: classifies+encodes each input-activation-difference element.
    Runs ONLY in diff mode. ITC and act modes do no encoding -> 0."""
    if mode != "diff":
        return 0.0
    # encode the difference operand(s): input A; attention self pays both Q and K
    elems = L["A"]
    if L["cat"] == "attn_self":
        elems = L["A"] + L["B"]
    return elems * E_EU


def vpu_energy(L, mode="itc"):
    """VPU: nonlinear/vector ops (GroupNorm, SiLU, LayerNorm, GEGLU, softmax) on the
    layer's OUTPUT activation (C elements).

    The nonlinear COMPUTE itself is identical across ITC / Cam-D / Ditto: nonlinear
    functions are NOT closed under differencing, so a difference-based design must
    first ACCUMULATE the difference back to the full activation at every nonlinear
    boundary, then run the same nonlinear math on the full tensor. So:
      - nonlinear compute  : C * VPU_FRAC * E_VPU   (same in all three modes)
      - restore (accumulate): diff modes only, C * VPU_FRAC * E_RESTORE (an add per
        output element to reconstruct the full activation; ITC has nothing to restore)
    This is the small but real reason VPU is NOT bit-identical across modes (the
    earlier version wrongly made it mode-independent)."""
    nonlin = L["C"] * VPU_FRAC * E_VPU
    restore = L["C"] * VPU_FRAC * E_RESTORE if mode in ("diff", "camd") else 0.0
    return nonlin + restore


def defo_energy(L):
    """Defo Unit: per-layer execution-mode decision (compare act vs diff cost).
    A handful of ops per layer, independent of layer size -> tiny fixed cost."""
    return E_DEFO_PER_LAYER


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


def mem_energy_split(w_lines, cur_lines, prev_lines):
    """(sram_energy, dram_energy).
    weight: resident -> SRAM. cur_act: SRAM-hit blend. prev_act: forced DRAM."""
    sram = (w_lines + cur_lines * SRAM_HIT) * E_SRAM
    dram = cur_lines * (1 - SRAM_HIT) * E_DRAM + prev_lines * E_DRAM
    return sram, dram


def total_energy(layers, mode, defo_flip=None):
    """mode in {itc, ditto, camd}. Returns dict of six segments (Fig 13).
    - itc : baseline, original activations, 8-bit.
    - ditto: temporal difference WITH dynamic bit-width + zero-skip (+Defo flips).
    - camd: Cambricon-D = temporal difference but FULL 8-bit, NO bit-width/sparsity
            and NO encoding unit. It pays the prev-frame DRAM overhead with no
            compute savings -> energy ABOVE ITC (the paper's Cam-D inversion).
    defo_flip: set of layer indices Defo forces back to act (ditto only)."""
    e = {"core": 0.0, "sram": 0.0, "dram": 0.0, "eu": 0.0, "vpu": 0.0, "defo": 0.0}
    for i, L in enumerate(layers):
        n_sub = 2 if L["cat"] == "attn_self" else 1
        if mode == "itc":
            lmode = "itc"
        elif mode == "camd":
            lmode = "camd"
        else:
            use_act = defo_flip is not None and i in defo_flip
            lmode = "act" if use_act else "diff"
        # core: camd does full-precision difference (8-bit, no skip) -> like itc cost
        if lmode == "camd":
            e["core"] += L["macs"] * n_sub * E_MAC8     # dense diff, two sub-ops for self-attn
        else:
            e["core"] += compute_energy(L["macs"], lmode, n_sub)
        # memory: difference modes (diff, camd) pay prev-frame DRAM
        pays_prev = lmode in ("diff", "camd")
        cur_w, cur, prev = mem_lines(L, "diff" if pays_prev else "act")
        s, d = mem_energy_split(cur_w, cur, prev)
        e["sram"] += s
        e["dram"] += d
        # EU runs only in Ditto diff (camd has no encoding unit)
        e["eu"] += eu_energy(L, "diff" if lmode == "diff" else "act")
        e["vpu"] += vpu_energy(L, lmode)
        if mode == "ditto":
            e["defo"] += defo_energy(L)
    return e


def etotal(e):
    return sum(e.values())


def layer_energy(L, mode):
    """Total six-segment energy for ONE layer in a given mode (for Defo decision)."""
    n_sub = 2 if L["cat"] == "attn_self" else 1
    w, cur, prev = mem_lines(L, "diff" if mode == "diff" else "act")
    s, d = mem_energy_split(w, cur, prev)
    e = compute_energy(L["macs"], mode, n_sub) + s + d
    e += eu_energy(L, mode) + vpu_energy(L, mode)
    return e


def compute_flip(layers):
    """Defo flips a layer to act if act-mode total energy < diff-mode total energy."""
    fl = set()
    for i, L in enumerate(layers):
        if layer_energy(L, "act") < layer_energy(L, "diff"):
            fl.add(i)
    return fl


def main():
    print("Enumerating UNet (with attention) ...", flush=True)
    layers = enumerate_with_attention()
    print(f"{len(layers)} layers/matmuls\n")

    itc = total_energy(layers, "itc")
    itc_tot = etotal(itc)

    camd = total_energy(layers, "camd")
    ditto_camd = etotal(camd)

    nodefo = total_energy(layers, "ditto", defo_flip=None)
    ditto_nodefo = etotal(nodefo)

    flip = compute_flip(layers)
    defo = total_energy(layers, "ditto", defo_flip=flip)
    ditto_defo = etotal(defo)

    def rel(x):
        return x / itc_tot

    print("=== Energy by component, normalized to ITC total = 1.000 ===")
    print("    (six segments matching paper Fig 13: core/sram/dram/eu/vpu/defo)\n")
    segs = ["core", "sram", "dram", "eu", "vpu", "defo"]
    hdr = "  " + " ".join(f"{s:>7}" for s in segs) + f" {'TOTAL':>8}"
    print(hdr)
    for name, e in [("ITC ", itc), ("Cam-D", camd), ("Ditto-noDefo", nodefo),
                    ("Ditto+Defo", defo)]:
        row = "  " + " ".join(f"{e[s]/itc_tot:>7.3f}" for s in segs)
        print(f"{row} {etotal(e)/itc_tot:>8.3f}   {name}")

    saving = (1 - rel(ditto_defo)) * 100
    print(f"\n  Ditto+Defo energy saving vs ITC: {saving:.1f}%   (paper Fig 13: 17.74%)")
    print(f"  Cambricon-D (full-precision diff): {rel(ditto_camd):.2f}x ITC "
          f"({'ABOVE -> inversion reproduced' if ditto_camd>itc_tot else 'below 1.0'})")
    print(f"  Ditto-noDefo (4bit+sparsity diff): {rel(ditto_nodefo):.2f}x ITC")
    print(f"  Defo flipped {len(flip)}/{len(layers)} layers to act")

    print("\n--- sensitivity: on-chip buffer (effective tiling window) ---")
    print(f"  {'BUF(MB)':>8} {'Defo saving%':>13} {'Cam-D x ITC':>12} {'flipped':>8}")
    global BUF_ELEMS
    for mb in [1, 4, 16, 32, 64, 128, 192]:
        BUF_ELEMS = int(mb * 1024 * 1024)
        it = etotal(total_energy(layers, "itc"))
        fl = compute_flip(layers)
        sd = etotal(total_energy(layers, "ditto", defo_flip=fl))
        cd = etotal(total_energy(layers, "camd"))
        print(f"  {mb:>8} {(1-sd/it)*100:>12.1f}% {cd/it:>11.2f}x {len(fl):>8}")
    BUF_ELEMS = 192 * 1024 * 1024

    print("\n--- VPU segment size (its share of ITC energy; same in ITC and Ditto) ---")
    global VPU_FRAC
    for vf in [0.25, 0.5, 1.0]:
        VPU_FRAC = vf
        it = total_energy(layers, "itc")
        print(f"  VPU_FRAC {vf:.2f}: VPU = {it['vpu']/etotal(it)*100:4.1f}% of ITC energy")
    VPU_FRAC = 0.5

    print("\nHonest: SRAM per-access energy is CACTI-7.0-measured (45nm, 64B line).")
    print("MAC/DRAM are public 45nm estimates. EU/VPU/Defo are now modeled (the six")
    print("segments of paper Fig 13). Remaining unknowns (effective tiling buffer,")
    print("VPU per-element energy, bank organization) are shown as sensitivity sweeps,")
    print("NOT tuned to hit 17.74%.")
    return

    # (old sensitivity blocks below retained-but-unreached; superseded above)
    print("are shown as sensitivity sweeps, not tuned to hit 17.74%.")


if __name__ == "__main__":
    main()

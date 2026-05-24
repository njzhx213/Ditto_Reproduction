#!/usr/bin/env python3
"""
enumerate_unet_layers.py — A': full-UNet speedup via STRUCTURAL ENUMERATION.

Why this file exists
--------------------
Our Step A run (run_fig13.py) gave ~2.72x, not the paper's ~1.5x. That is NOT a
bug: the collected trace only contains 6 layers, all compute-heavy attention
blocks. The real SDM UNet has ~700 modules, including many small ResNet convs,
norms, and projection layers that are MEMORY-bound. Those are exactly the layers
that (a) drag the average speedup down and (b) make Defo actually fire (it falls
back to original-activation execution when temporal-diff would cost more memory).

Re-collecting a FULL trace would balloon 19 GB -> hundreds of GB. Instead we
enumerate the UNet STRUCTURE directly from diffusers (no trace, seconds), read
each Conv2d / Linear's real dimensions, compute per-layer N_macs and mem_bytes,
and feed them into the SAME cycle model + Defo as Step A. This yields a
full-network speedup that should sit well below 2.72x, near the paper's 1.5x.

Honesty notes (read these)
--------------------------
1. BIT-WIDTH is EXTRAPOLATED. We measured the temporal-diff bit-width
   distribution on 6 attention layers only (Fig 5: zero 45.9 / 4-bit 53.4 /
   >4-bit 0.7). Conv/ResNet layers were NOT measured; we reuse the same
   distribution for them. This is an INFERENCE, not a paper-stated fact and not
   a measured result. It is the single biggest assumption in this file.
2. MAC model counts Conv2d + Linear only. GroupNorm / SiLU / Softmax etc. carry
   negligible MACs (we treat their compute as ~0) but DO carry memory traffic;
   that memory shows up via the activation bytes of the neighbouring linear layer.
3. Memory model is first-order (weights + in_act + out_act, int8, 1 B/elem),
   identical in spirit to Step A. The 2.75x temporal-diff memory penalty is the
   paper's Fig 8 average.

Cross-check for free
--------------------
Defo's "fraction of layers flipped back to original-activation execution" is
reported alongside the speedup. Paper Fig 17 says this averages ~14.4%.

Usage
-----
    cd ~/Ditto
    python3 sim/enumerate_unet_layers.py
    # quicker (skip figure):  python3 sim/enumerate_unet_layers.py --no-fig

Author: njzhx213
Project: Ditto Reproduction (HPCA 2025)
Phase: Week 2, Step A' (full-UNet structural speedup)
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

# ----------------------------------------------------------------------------
# Hardware configuration (paper Table III) + Step A constants
# ----------------------------------------------------------------------------
N_PE_ITC = 27648          # ITC: A8W8, 1 effective 8-bit MAC / PE / cycle
N_PE_DITTO = 39398        # Ditto: A4W8, 4 4-bit MACs / PE / cycle
DITTO_LANES = 4
DRAM_BW = 1500.0          # bytes / cycle  (~1.5 TB/s @ 1 GHz)
MEM_PENALTY_DIFF = 2.75   # temporal-diff memory blow-up (paper Fig 8 avg)

# Bit-width distribution — EXTRAPOLATED from 6 attention layers (Fig 5 dynamic).
# See honesty note #1.
BW_ZERO = 0.459
BW_LE4 = 0.534
BW_GT4 = 0.007

RESULT_DIR = Path.home() / "Ditto" / "results"
FIG_DIR = Path.home() / "Ditto" / "figs"


# ----------------------------------------------------------------------------
# PURE cost functions (no torch) — these are unit-testable on their own
# ----------------------------------------------------------------------------
def conv_macs_mem(C_in, C_out, kH, kW, groups, B, H_in, W_in, H_out, W_out):
    """MACs and int8 byte traffic for one Conv2d invocation."""
    macs = B * C_out * (C_in // groups) * kH * kW * H_out * W_out
    w_bytes = C_out * (C_in // groups) * kH * kW
    in_bytes = B * C_in * H_in * W_in
    out_bytes = B * C_out * H_out * W_out
    return float(macs), float(w_bytes + in_bytes + out_bytes)


def linear_macs_mem(F_in, F_out, n_rows):
    """MACs and int8 byte traffic for one Linear invocation (n_rows = B*tokens)."""
    macs = n_rows * F_out * F_in
    w_bytes = F_out * F_in
    in_bytes = n_rows * F_in
    out_bytes = n_rows * F_out
    return float(macs), float(w_bytes + in_bytes + out_bytes)


# ----------------------------------------------------------------------------
# Cycle model — three execution modes, then Defo picks the cheaper of two
# ----------------------------------------------------------------------------
_NONZERO = BW_LE4 + BW_GT4
_BIT_FACTOR = (BW_LE4 * 1 + BW_GT4 * 2) / _NONZERO if _NONZERO > 0 else 1.0


def itc_cycle(macs, mem_bytes):
    """ITC baseline: full 8-bit, no diff. Roofline max(compute, memory)."""
    comp = macs / N_PE_ITC
    mem = mem_bytes / DRAM_BW
    return max(comp, mem)


def ditto_diff_cycle(macs, mem_bytes):
    """Ditto temporal-diff mode: zero-skip + 4-bit, but 2.75x memory penalty."""
    comp = macs * _NONZERO * _BIT_FACTOR / (N_PE_DITTO * DITTO_LANES)
    mem = MEM_PENALTY_DIFF * mem_bytes / DRAM_BW
    return max(comp, mem)


def ditto_act_cycle(macs, mem_bytes):
    """Ditto original-activation mode (Defo fallback): full 8-bit on Ditto PEs.
    An 8-bit op uses 2 of the 4 4-bit multipliers -> 2 effective 8-bit MAC/PE/cyc.
    No temporal-diff memory penalty (1x)."""
    comp = macs / (N_PE_DITTO * 2)
    mem = mem_bytes / DRAM_BW
    return max(comp, mem)


def run_cycle_model(layers):
    """layers: list of dicts {name, kind, macs, mem_bytes}.
    Returns aggregate cycles + per-layer Defo decisions."""
    tot_itc = tot_diff_only = tot_defo = 0.0
    n_flipped = 0
    per_layer = []

    for L in layers:
        c_itc = itc_cycle(L["macs"], L["mem_bytes"])
        c_diff = ditto_diff_cycle(L["macs"], L["mem_bytes"])
        c_act = ditto_act_cycle(L["macs"], L["mem_bytes"])

        # Defo: pick the cheaper of (temporal-diff) vs (original activation)
        if c_act < c_diff:
            c_defo = c_act
            mode = "act"      # Defo flipped this layer back to original
            n_flipped += 1
        else:
            c_defo = c_diff
            mode = "diff"

        tot_itc += c_itc
        tot_diff_only += c_diff
        tot_defo += c_defo
        per_layer.append({**L, "mode": mode,
                          "c_itc": c_itc, "c_diff": c_diff,
                          "c_act": c_act, "c_defo": c_defo})

    return {
        "total_itc": tot_itc,
        "total_diff_only": tot_diff_only,
        "total_defo": tot_defo,
        "speedup_diff_only": tot_itc / tot_diff_only if tot_diff_only else 0.0,
        "speedup_defo": tot_itc / tot_defo if tot_defo else 0.0,
        "n_layers": len(layers),
        "n_flipped": n_flipped,
        "flip_frac": n_flipped / len(layers) if layers else 0.0,
        "per_layer": per_layer,
    }


# ----------------------------------------------------------------------------
# Enumeration via diffusers (this is the part that needs torch + the model)
# ----------------------------------------------------------------------------
def enumerate_sdm_layers():
    import torch
    import torch.nn as nn
    from diffusers import UNet2DConditionModel

    print("Loading SDM v1.4 UNet from local HF cache ...", flush=True)
    unet = UNet2DConditionModel.from_pretrained(
        "CompVis/stable-diffusion-v1-4", subfolder="unet"
    ).eval()

    records = []  # filled by hooks

    def make_hook(name):
        def hook(mod, inp, out):
            x = inp[0]
            y = out[0] if isinstance(out, (tuple, list)) else out
            records.append((name, mod, tuple(x.shape), tuple(y.shape)))
        return hook

    handles = []
    for name, mod in unet.named_modules():
        if isinstance(mod, (nn.Conv2d, nn.Linear)):
            handles.append(mod.register_forward_hook(make_hook(name)))

    # One dummy forward (SD v1.4: 4-ch 64x64 latent, CLIP-L 77x768 context)
    sample = torch.randn(1, 4, 64, 64)
    timestep = torch.tensor(1, dtype=torch.long)
    context = torch.randn(1, 77, 768)
    with torch.no_grad():
        unet(sample, timestep, context)

    for h in handles:
        h.remove()

    layers = []
    for name, mod, in_shape, out_shape in records:
        if isinstance(mod, nn.Conv2d):
            B = in_shape[0]
            C_in, H_in, W_in = in_shape[1], in_shape[2], in_shape[3]
            C_out, H_out, W_out = out_shape[1], out_shape[2], out_shape[3]
            kH, kW = mod.kernel_size
            macs, mem = conv_macs_mem(C_in, C_out, kH, kW, mod.groups,
                                      B, H_in, W_in, H_out, W_out)
            kind = "conv"
        else:  # nn.Linear
            n_rows = 1
            for d in in_shape[:-1]:
                n_rows *= d
            macs, mem = linear_macs_mem(mod.in_features, mod.out_features, n_rows)
            kind = "linear"
        layers.append({"name": name, "kind": kind, "macs": macs, "mem_bytes": mem})

    return layers


# ----------------------------------------------------------------------------
# Reporting
# ----------------------------------------------------------------------------
def report(res, layers, make_fig=True):
    # MAC-weighted breakdown by layer kind, for sanity
    by_kind = {}
    for L in layers:
        k = L["kind"]
        by_kind.setdefault(k, [0, 0.0])
        by_kind[k][0] += 1
        by_kind[k][1] += L["macs"]
    total_macs = sum(L["macs"] for L in layers)

    print("\n=== Full-UNet structural enumeration (A') ===")
    print(f"Layers (Conv2d + Linear): {res['n_layers']}")
    for k, (n, m) in sorted(by_kind.items()):
        print(f"  {k:<8} count={n:<4} MACs={m/1e9:8.2f} G  ({m/total_macs*100:5.1f}%)")
    print(f"  TOTAL MACs (1 step): {total_macs/1e9:.2f} G")
    print()
    print(f"{'mode':<26} {'cycles':>14} {'speedup':>9}")
    print("-" * 52)
    print(f"{'ITC baseline':<26} {res['total_itc']:>14.0f} {1.0:>8.2f}x")
    print(f"{'Ditto (diff-only)':<26} {res['total_diff_only']:>14.0f} "
          f"{res['speedup_diff_only']:>8.2f}x")
    print(f"{'Ditto + Defo':<26} {res['total_defo']:>14.0f} "
          f"{res['speedup_defo']:>8.2f}x")
    print()
    print(f"Defo flipped {res['n_flipped']}/{res['n_layers']} layers "
          f"({res['flip_frac']*100:.1f}%) back to original-activation execution.")
    print(f"  (paper Fig 17 reports ~14.4% on average across models)")
    print()
    print(f"Paper Fig 13 SDM target: ~1.5x over ITC.")

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    out = {
        "n_layers": res["n_layers"],
        "total_macs_giga": total_macs / 1e9,
        "by_kind": {k: {"count": n, "macs_giga": m / 1e9}
                    for k, (n, m) in by_kind.items()},
        "speedup_diff_only": res["speedup_diff_only"],
        "speedup_defo": res["speedup_defo"],
        "defo_flip_frac": res["flip_frac"],
        "bitwidth_used": {"zero": BW_ZERO, "le4": BW_LE4, "gt4": BW_GT4,
                          "NOTE": "extrapolated from 6 attention layers"},
    }
    with open(RESULT_DIR / "fig13_full_unet_stats.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"Stats saved: {RESULT_DIR / 'fig13_full_unet_stats.json'}")

    if make_fig:
        _make_fig(res)


def _make_fig(res):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    labels = ["ITC\n(baseline)", "Ditto\n(diff-only)", "Ditto\n+ Defo", "Paper\n(~1.5x)"]
    vals = [1.0, res["speedup_diff_only"], res["speedup_defo"], 1.5]
    colors = ["#999999", "#C44E52", "#4C72B0", "#55A868"]

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(labels, vals, color=colors, width=0.6)
    ax.axhline(1.0, color="k", lw=0.8, ls="--", alpha=0.5)
    ax.set_ylabel("Speedup over ITC")
    ax.set_title("Fig 13 (A'): full-UNet speedup with Defo", pad=14)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.03, f"{v:.2f}x",
                ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_ylim(0, max(vals) * 1.2)
    fig.text(0.5, -0.02,
             f"Defo flipped {res['flip_frac']*100:.1f}% of layers to "
             f"original-activation execution (paper ~14.4%)",
             ha="center", fontsize=8.5, color="#555")
    plt.tight_layout()
    out = FIG_DIR / "fig13_full_unet.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Figure saved: {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-fig", action="store_true", help="skip matplotlib figure")
    args = ap.parse_args()

    layers = enumerate_sdm_layers()
    res = run_cycle_model(layers)
    report(res, layers, make_fig=not args.no_fig)


if __name__ == "__main__":
    main()

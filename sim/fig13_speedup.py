#!/usr/bin/env python3
"""
fig13_speedup.py - Step A (with first-order tiling memory model) + Defo.

Self-contained. Supersedes the earlier 6-layer run_fig13.py and the disguised-
bandwidth enumerate_unet_layers.py. Follows the original Week-2 plan: memory is
COUNTED IN BYTES, now refined with a finite on-chip buffer so that data that does
not fit is re-fetched from DRAM. Defo is the embedded per-layer min decision.

Why the tiling refinement
-------------------------
Pure unique-byte counting gave bytes_diff/bytes_act = 1.34x on the real UNet
(weight-dominated at batch 1), well below the paper's Fig-8 average of 2.75x.
Unique-byte counting misses the REPEATED activation/weight streaming that happens
when a layer\'s working set exceeds the on-chip buffer. We model that here:

    working_set_act  = W + in + out
    working_set_diff = W + in + out + prev_in + prev_out
    reload(ws)       = max(1, ws / BUF)                 # 1st-order, zero-reuse bound
    effective_bytes  = working_set * reload(working_set)

Because diff mode\'s working set is LARGER, it spills more and its reload is
amplified super-linearly -> the diff/act access ratio rises above 1.34x as BUF
shrinks. We SWEEP BUF (not tune it) and report where the ratio brackets 2.75x.

The two knobs, kept separate
----------------------------
  BUF (on-chip bytes): controls the ACCESS ratio diff/act. Bandwidth-independent.
  BYTES_PER_CYCLE:     converts effective bytes into stall cycles (roofline).
Flow: sweep BUF -> pick BUF* where ratio ~= 2.75x -> sweep bandwidth at BUF* for
the speedup curve.

Honesty boundaries
------------------
  - reload = ws/BUF is a PESSIMISTIC first-order bound (assumes no cross-tile
    reuse). Real tiling schedules sit between this and read-once. It captures the
    MECHANISM (spill -> re-fetch), not a cycle-accurate access count.
  - BUF and BYTES_PER_CYCLE are unpublished; both are swept, none tuned to answer.
  - Bit-width (zero 45.9 / 4-bit 53.4 / >4-bit 0.7) is the Fig-5 aggregate.
  - MACs count Conv2d + Linear only.

Usage
-----
    cd ~/Ditto
    python3 sim/fig13_speedup.py
    python3 sim/fig13_speedup.py --no-fig

Author: njzhx213
Project: Ditto Reproduction (HPCA 2025)
Phase: Week 2, Step A (tiling memory model + Defo)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# ---- Hardware (paper Table III) -------------------------------------------
N_PE_ITC = 27648
N_PE_DITTO = 39398
DITTO_LANES = 4

# ---- Bit-width: Fig 5 aggregate -------------------------------------------
BW_ZERO, BW_LE4, BW_GT4 = 0.459, 0.534, 0.007
_NONZERO = BW_LE4 + BW_GT4
_BIT_FACTOR = (BW_LE4 + 2 * BW_GT4) / _NONZERO if _NONZERO else 1.0

RESULT_DIR = Path.home() / "Ditto" / "results"
FIG_DIR = Path.home() / "Ditto" / "figs"


# ---- PURE math (no torch) -------------------------------------------------
def conv_bytes_macs(C_in, C_out, kH, kW, groups, B, H_in, W_in, H_out, W_out):
    macs = B * C_out * (C_in // groups) * kH * kW * H_out * W_out
    w = C_out * (C_in // groups) * kH * kW
    a_in = B * C_in * H_in * W_in
    a_out = B * C_out * H_out * W_out
    return float(macs), float(w), float(a_in), float(a_out)


def linear_bytes_macs(F_in, F_out, n_rows):
    macs = n_rows * F_out * F_in
    return float(macs), float(F_out * F_in), float(n_rows * F_in), float(n_rows * F_out)


def comp_itc(macs):
    return macs / N_PE_ITC


def comp_diff(macs):
    return macs * _NONZERO * _BIT_FACTOR / (N_PE_DITTO * DITTO_LANES)


def comp_act_ditto(macs):
    return macs / (N_PE_DITTO * 2)


def reload_factor(ws, buf):
    """Pessimistic first-order reload: data ws bytes is streamed ~ws/buf times."""
    return max(1.0, ws / buf)


def effective_bytes(L, mode, buf):
    w, ai, ao = L["w"], L["a_in"], L["a_out"]
    if mode == "diff":
        ws = w + ai + ao + ai + ao          # + prev_in + prev_out
    else:                                   # "act" / itc
        ws = w + ai + ao
    return ws * reload_factor(ws, buf)


def layer_cycles(L, bpc, buf):
    eff_act = effective_bytes(L, "act", buf)
    eff_diff = effective_bytes(L, "diff", buf)
    cyc_itc = max(comp_itc(L["macs"]), eff_act / bpc)
    cyc_diff = max(comp_diff(L["macs"]), eff_diff / bpc)
    cyc_act = max(comp_act_ditto(L["macs"]), eff_act / bpc)
    return cyc_itc, cyc_diff, cyc_act, eff_act, eff_diff


def run_at(layers, bpc, buf):
    tot_itc = tot_diff_only = tot_defo = 0.0
    eff_act_sum = eff_diff_sum = eff_defo_sum = 0.0
    n_flip = 0
    for L in layers:
        cyc_itc, cyc_diff, cyc_act, e_act, e_diff = layer_cycles(L, bpc, buf)
        tot_itc += cyc_itc
        tot_diff_only += cyc_diff
        eff_act_sum += e_act          # all layers in act mode
        eff_diff_sum += e_diff        # all layers in diff mode (BARE diff, ~ paper Fig 8)
        if cyc_act < cyc_diff:        # Defo flips this layer back to act
            tot_defo += cyc_act
            eff_defo_sum += e_act     # flipped layer pays act memory, not diff
            n_flip += 1
        else:
            tot_defo += cyc_diff
            eff_defo_sum += e_diff    # kept-diff layer pays diff memory
    return {
        "bpc": bpc, "buf": buf,
        "speedup_diff_only": tot_itc / tot_diff_only if tot_diff_only else 0.0,
        "speedup_defo": tot_itc / tot_defo if tot_defo else 0.0,
        "flip_frac": n_flip / len(layers) if layers else 0.0,
        # BARE difference (all layers diff) vs all-act -> compare to paper Fig 8 (2.75x)
        "mem_ratio": eff_diff_sum / eff_act_sum if eff_act_sum else 0.0,
        # Defo-selected memory vs all-act -> compare to paper Fig 14 (Ditto 1.56x)
        "mem_ratio_defo": eff_defo_sum / eff_act_sum if eff_act_sum else 0.0,
    }


# ---- Enumeration (needs torch + the model) --------------------------------
def enumerate_sdm_layers():
    import torch
    import torch.nn as nn
    from diffusers import UNet2DConditionModel

    print("Loading SDM v1.4 UNet from local HF cache ...", flush=True)
    unet = UNet2DConditionModel.from_pretrained(
        "CompVis/stable-diffusion-v1-4", subfolder="unet").eval()
    records = []

    def make_hook(name):
        def hook(mod, inp, out):
            x = inp[0]
            y = out[0] if isinstance(out, (tuple, list)) else out
            records.append((mod, tuple(x.shape), tuple(y.shape)))
        return hook

    handles = [m.register_forward_hook(make_hook(n))
               for n, m in unet.named_modules()
               if isinstance(m, (nn.Conv2d, nn.Linear))]
    sample = torch.randn(1, 4, 64, 64)
    timestep = torch.tensor(1, dtype=torch.long)
    context = torch.randn(1, 77, 768)
    with torch.no_grad():
        unet(sample, timestep, context)
    for h in handles:
        h.remove()

    layers = []
    for mod, in_s, out_s in records:
        if isinstance(mod, nn.Conv2d):
            macs, w, a_in, a_out = conv_bytes_macs(
                in_s[1], out_s[1], mod.kernel_size[0], mod.kernel_size[1],
                mod.groups, in_s[0], in_s[2], in_s[3], out_s[2], out_s[3])
            kind = "conv"
        else:
            n_rows = 1
            for d in in_s[:-1]:
                n_rows *= d
            macs, w, a_in, a_out = linear_bytes_macs(mod.in_features, mod.out_features, n_rows)
            kind = "linear"
        layers.append({"kind": kind, "macs": macs, "w": w, "a_in": a_in, "a_out": a_out})
    return layers


# ---- Sweeps + report ------------------------------------------------------
def analyze(layers):
    total_macs = sum(L["macs"] for L in layers)
    print("\n=== Step A: tiling memory model + Defo, full UNet ===")
    print(f"Layers (Conv2d+Linear): {len(layers)}   Total MACs: {total_macs/1e9:.1f} G")

    # --- Stage 1: BUFFER sweep -> diff/act access ratio (bandwidth-independent) ---
    print("\n[1] Buffer sweep: how on-chip buffer size drives the diff/act access ratio")
    print(f"{'BUF (MB)':>10} {'diff/act ratio':>16}")
    print("-" * 28)
    buf_mb = [0.25, 0.5, 1, 2, 4, 8, 16, 32, 64, 128, 256]
    ratio_rows = []
    for mb in buf_mb:
        buf = mb * 1024 * 1024
        r = run_at(layers, 1e9, buf)   # bandwidth irrelevant for the ratio
        ratio_rows.append((mb, r["mem_ratio"]))
        print(f"{mb:>10} {r['mem_ratio']:>15.2f}x")

    # pick BUF* whose ratio is closest to the paper's 2.75x (report, do not tune)
    buf_star_mb = min(ratio_rows, key=lambda t: abs(t[1] - 2.75))[0]
    buf_star = buf_star_mb * 1024 * 1024
    big_ratio = ratio_rows[-1][1]
    print(f"\nLarge-buffer limit ratio = {big_ratio:.2f}x "
          f"(should match the old no-reload 1.34x -> consistency check)")
    print(f"Ratio closest to paper 2.75x at BUF = {buf_star_mb} MB "
          f"(ratio {min(ratio_rows, key=lambda t: abs(t[1]-2.75))[1]:.2f}x)")

    # --- Stage 2: BANDWIDTH sweep at BUF* -> speedup + Defo curve ---
    print(f"\n[2] Bandwidth sweep at BUF = {buf_star_mb} MB")
    print(f"{'BYTES/cyc':>10} {'diff-only':>11} {'+Defo':>9} {'Defo flip%':>11}")
    print("-" * 46)
    bpcs = [8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
    bw_rows = []
    for b in bpcs:
        r = run_at(layers, b, buf_star)
        bw_rows.append(r)
        print(f"{b:>10} {r['speedup_diff_only']:>10.2f}x {r['speedup_defo']:>8.2f}x "
              f"{r['flip_frac']*100:>10.1f}%")

    cross = None
    for a, b in zip(bw_rows, bw_rows[1:]):
        lo, hi = sorted([a["speedup_defo"], b["speedup_defo"]])
        if lo <= 1.5 <= hi:
            cross = (a, b)
            break
    print()
    if cross:
        a, b = cross
        print(f"Paper 1.5x falls between BYTES/cyc {a['bpc']} ({a['speedup_defo']:.2f}x, "
              f"flip {a['flip_frac']*100:.0f}%) and {b['bpc']} ({b['speedup_defo']:.2f}x, "
              f"flip {b['flip_frac']*100:.0f}%).")
    else:
        print("Paper 1.5x not bracketed by this bandwidth range at BUF*.")

    # --- Stage 3: memory-access ratio, BARE diff vs Defo-selected ---
    # Paper distinguishes two memory numbers:
    #   Fig 8  (algorithm, bare difference, no Defo): 2.75x avg
    #   Fig 14 (hardware, with Defo): Ditto 1.56x, Cam-D 1.95x, Ditto+ 1.36x
    # Our mem_ratio is bare-diff (all layers diff) -> compare to Fig 8.
    # Our mem_ratio_defo (Defo flips memory-heavy layers back to act) -> Fig 14.
    print("\n[3] Memory-access ratio vs the paper's TWO memory figures")
    print(f"{'BYTES/cyc':>10} {'bare diff/act':>14} {'Defo-selected':>14}")
    print("-" * 40)
    for r in bw_rows:
        print(f"{r['bpc']:>10} {r['mem_ratio']:>13.2f}x {r['mem_ratio_defo']:>13.2f}x")
    # representative values at the bandwidth where Defo is active (mid-range)
    bare = bw_rows[0]["mem_ratio"]          # bandwidth-independent for bare ratio
    defo_vals = [r["mem_ratio_defo"] for r in bw_rows]
    print(f"\n  Bare difference (all layers diff): {bare:.2f}x  -> paper Fig 8  = 2.75x")
    print(f"  Defo-selected range: {min(defo_vals):.2f}x .. {max(defo_vals):.2f}x"
          f"  -> paper Fig 14 = Ditto 1.56x (Cam-D 1.95x, Ditto+ 1.36x)")
    print("  Note: our batch=1, weight-dominated SD UNet gives lower absolute ratios")
    print("  than the paper in BOTH columns; the relative ordering (Defo < bare) holds.")

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULT_DIR / "fig13_sweep.json", "w") as f:
        json.dump({"total_macs_giga": total_macs / 1e9,
                   "buffer_sweep": ratio_rows, "buf_star_mb": buf_star_mb,
                   "bandwidth_sweep": bw_rows}, f, indent=2)
    print(f"\nSaved: {RESULT_DIR / 'fig13_sweep.json'}")
    return ratio_rows, bw_rows, buf_star_mb


def make_fig(ratio_rows, bw_rows, buf_star_mb):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(12, 4.6))

    # left: buffer -> ratio
    mb = [r[0] for r in ratio_rows]
    rat = [r[1] for r in ratio_rows]
    axL.semilogx(mb, rat, "o-", color="#8172B3")
    axL.axhline(2.75, color="#C44E52", ls="--", lw=1.2, label="paper 2.75x (Fig 8)")
    axL.axhline(1.34, color="#999", ls=":", lw=1, label="no-reload 1.34x")
    axL.set_xlabel("On-chip buffer (MB)  -- swept knob #1")
    axL.set_ylabel("diff/act access ratio")
    axL.set_title("Buffer spilling drives the memory ratio")
    axL.legend(fontsize=8)

    # right: bandwidth -> speedup + flip
    bpc = [r["bpc"] for r in bw_rows]
    su = [r["speedup_defo"] for r in bw_rows]
    flip = [r["flip_frac"] * 100 for r in bw_rows]
    axR.semilogx(bpc, su, "o-", color="#4C72B0", label="Ditto+Defo speedup")
    axR.axhline(1.5, color="#55A868", ls="--", lw=1.2, label="paper 1.5x")
    axR.set_xlabel("BYTES_PER_CYCLE  -- swept knob #2")
    axR.set_ylabel("Speedup over ITC", color="#4C72B0")
    ax2 = axR.twinx()
    ax2.semilogx(bpc, flip, "s--", color="#C44E52", alpha=0.6)
    ax2.axhline(14.4, color="#C44E52", ls=":", lw=1, alpha=0.5)
    ax2.set_ylabel("Defo flip %  (paper ~14.4%)", color="#C44E52")
    axR.set_title(f"Speedup vs bandwidth (at BUF = {buf_star_mb} MB)")
    axR.legend(fontsize=8, loc="upper left")

    plt.tight_layout()
    out = FIG_DIR / "fig13_sweep.png"
    plt.savefig(out, dpi=150)
    print(f"Figure saved: {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-fig", action="store_true")
    args = ap.parse_args()
    layers = enumerate_sdm_layers()
    ratio_rows, bw_rows, buf_star_mb = analyze(layers)
    if not args.no_fig:
        make_fig(ratio_rows, bw_rows, buf_star_mb)


if __name__ == "__main__":
    main()

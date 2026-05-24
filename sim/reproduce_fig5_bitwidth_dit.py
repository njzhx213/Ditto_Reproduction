#!/usr/bin/env python3
"""
reproduce_fig5_bitwidth_dit.py - DiT-XL/2 temporal-difference bit-width distribution.

The DiT analog of reproduce_fig5_bitwidth.py. The quantization and bit-width
classification are REUSED VERBATIM from the SDM script (same ruler), so DiT and
SDM are directly comparable -- and the trace was collected with ALIGNED sampling
(50 steps, CFG 7.5) for the same reason. We report the DYNAMIC (true-absmax)
result as the honest, unmodified number; the calib percentile is the SAME value
used for SDM (NOT re-tuned for DiT), so any DiT-vs-SDM difference is the model,
not a refitted scale.

DiT trace layout (flat): traces/dit/c<class>_b<block>_<layer>_t<step>.npz
Temporal difference = adjacent denoising steps of the SAME (class, block, layer).

    cd ~/Ditto && python3 sim/reproduce_fig5_bitwidth_dit.py
    cd ~/Ditto && python3 sim/reproduce_fig5_bitwidth_dit.py --calib-pct 99.9
"""
from __future__ import annotations

import argparse
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

TRACE_ROOT = Path.home() / "Ditto" / "traces" / "dit"
RESULT_DIR = Path.home() / "Ditto" / "results"
PAPER_SDM = {"zero": 0.44, "le4_nonzero": 0.52, "gt4": 0.04}
QMAX = 127

# default calib percentile: SAME as the SDM run (SDM uses 99.0), not re-tuned for DiT
DEFAULT_CALIB_PCT = 99.0


# --- quant + classify: VERBATIM from reproduce_fig5_bitwidth.py (same ruler) ---
def classify_signed_range(diff):
    zero = int((diff == 0).sum())
    le4_nz = int(((diff >= -8) & (diff <= 7) & (diff != 0)).sum())
    gt4 = int(((diff < -8) | (diff > 7)).sum())
    return zero, le4_nz, gt4


def process_pair(prev, curr, calib_pct, do_calib=True):
    absmax_dyn = float(max(np.abs(prev).max(), np.abs(curr).max()))
    scale_dyn = absmax_dyn / QMAX if absmax_dyn > 1e-12 else 1.0
    p_dyn = np.clip(np.round(prev / scale_dyn), -QMAX, QMAX).astype(np.int32)
    c_dyn = np.clip(np.round(curr / scale_dyn), -QMAX, QMAX).astype(np.int32)
    dyn_counts = classify_signed_range(c_dyn - p_dyn)

    if not do_calib:
        return dyn_counts, None
    # percentile over a random subsample to avoid sorting millions of elems per pair
    flat = np.concatenate([np.abs(prev).ravel(), np.abs(curr).ravel()])
    if flat.size > 200_000:
        idx = np.random.default_rng(0).integers(0, flat.size, 200_000)
        flat = flat[idx]
    absmax_cal = float(np.percentile(flat, calib_pct))
    scale_cal = absmax_cal / QMAX if absmax_cal > 1e-12 else 1.0
    p_cal = np.clip(np.round(prev / scale_cal), -QMAX, QMAX).astype(np.int32)
    c_cal = np.clip(np.round(curr / scale_cal), -QMAX, QMAX).astype(np.int32)
    cal_counts = classify_signed_range(c_cal - p_cal)
    return dyn_counts, cal_counts
# -------------------------------------------------------------------------------


def parse_name(p):
    """c207_b9_attn1-to-q_t03.npz -> (class, block, layer, step)."""
    m = re.match(r"c(\d+)_b(\d+)_(.+)_t(\d+)\.npz", p.name)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2)), m.group(3), int(m.group(4))


def run(calib_pct, tensor_key, step_stride, do_calib=True):
    t0 = time.time()
    files = sorted(TRACE_ROOT.glob("*.npz"))
    if not files:
        print(f"No npz in {TRACE_ROOT}")
        sys.exit(1)

    groups = defaultdict(dict)
    for p in files:
        info = parse_name(p)
        if info:
            cls, blk, lay, step = info
            groups[(cls, blk, lay)][step] = p

    dyn = {"zero": 0, "le4": 0, "gt4": 0}
    cal = {"zero": 0, "le4": 0, "gt4": 0}
    n_pairs = 0
    for key, step_map in groups.items():
        steps = sorted(step_map)
        for i in range(0, len(steps) - 1, step_stride):
            s0, s1 = steps[i], steps[i + 1]
            d0 = np.load(step_map[s0], allow_pickle=True)
            d1 = np.load(step_map[s1], allow_pickle=True)
            prev = d0[tensor_key].astype(np.float32)
            curr = d1[tensor_key].astype(np.float32)
            if prev.shape != curr.shape:
                continue
            dcounts, ccounts = process_pair(prev, curr, calib_pct, do_calib)
            dyn["zero"] += dcounts[0]; dyn["le4"] += dcounts[1]; dyn["gt4"] += dcounts[2]
            if ccounts is not None:
                cal["zero"] += ccounts[0]; cal["le4"] += ccounts[1]; cal["gt4"] += ccounts[2]
            n_pairs += 1
        if n_pairs and n_pairs % 100 == 0:
            print(f"  ... {n_pairs} pairs, {time.time()-t0:.0f}s", flush=True)

    def frac(d):
        t = d["zero"] + d["le4"] + d["gt4"]
        return (d["zero"] / t, d["le4"] / t, d["gt4"] / t) if t else (0, 0, 0)

    dz, dl, dg = frac(dyn)
    cz, cl, cg = frac(cal)
    print(f"\n=== DiT-XL/2 temporal-difference bit-width ({tensor_key}, "
          f"{n_pairs} adjacent-step pairs, {time.time()-t0:.0f}s) ===\n")
    print(f"{'scheme':>10} {'zero':>8} {'<=4-bit':>9} {'>4-bit':>8}")
    print("-" * 38)
    print(f"{'dynamic':>10} {dz*100:>7.1f}% {dl*100:>8.1f}% {dg*100:>7.1f}%")
    print(f"{'calib':>10} {cz*100:>7.1f}% {cl*100:>8.1f}% {cg*100:>7.1f}%  "
          f"(pct={calib_pct}, same as SDM)")
    print(f"\n{'paper SDM':>10} {PAPER_SDM['zero']*100:>7.1f}% "
          f"{PAPER_SDM['le4_nonzero']*100:>8.1f}% {PAPER_SDM['gt4']*100:>7.1f}%")
    print(f"{'our SDM':>10}  ~45.9%    ~53.4%    ~0.7%  (dynamic, from SDM trace)")
    print("\nComparison is valid: same quant/classify ruler + aligned sampling")
    print("(50 steps, CFG 7.5). DiT uses class conditioning (ada_norm); SDM uses")
    print("text (cross-attn) -- a genuine model difference, not a collection mismatch.")

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    import json
    with open(RESULT_DIR / "fig5_bitwidth_dit.json", "w") as f:
        json.dump({"dynamic": {"zero": dz, "le4": dl, "gt4": dg},
                   "calib": {"zero": cz, "le4": cl, "gt4": cg},
                   "n_pairs": n_pairs, "calib_pct": calib_pct}, f, indent=2)
    print(f"\nSaved: {RESULT_DIR / 'fig5_bitwidth_dit.json'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calib-pct", type=float, default=DEFAULT_CALIB_PCT)
    ap.add_argument("--tensor", default="input", choices=["input", "output"])
    ap.add_argument("--step-stride", type=int, default=1)
    ap.add_argument("--dynamic-only", action="store_true",
                    help="skip the slower calib percentile; report only the honest dynamic result")
    a = ap.parse_args()
    run(a.calib_pct, a.tensor, a.step_stride, do_calib=not a.dynamic_only)


if __name__ == "__main__":
    main()

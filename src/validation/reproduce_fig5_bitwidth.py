"""
reproduce_fig5_bitwidth.py — Reproduce paper Fig 5 (temporal-difference bit-width)
on the full SDM trace.

Paper Fig 5 SDM column (temporal differences, A8W8): zero ~44%, 4-bit ~52%,
>4-bit ~4% (i.e. ≤4-bit ~96%).

This script:
  1. Iterates the FULL collected trace: 20 images × 50 steps × 6 layers × all rows.
  2. For each adjacent-step activation pair, quantizes with TWO schemes:
       (a) dynamic  — per-tensor true-absmax (our honest baseline)
       (b) calib    — per-tensor PERCENTILE-clip absmax, approximating the
                      outlier-clipping behaviour of Q-Diffusion calibration.
     The calibration percentile is chosen so the aggregate >4-bit fraction lands
     near the paper's ~4%.
  3. Classifies each int8 temporal difference via SIGNED-RANGE ([-8,7] = 4-bit),
     matching paper Fig 5's minimum-signed-bit-width definition.
  4. Aggregates zero / 4-bit / >4-bit fractions and emits a grouped bar chart
     comparing (our dynamic) vs (our calib) vs (paper SDM).

IMPORTANT — honesty about calibration:
  We do NOT run the actual Q-Diffusion calibration repository. We approximate its
  effect (outlier clipping → tighter scale → larger int diffs → more >4-bit) with
  a percentile-clip scale. The dynamic-quant bar is our true, unmodified result;
  the calib bar shows that a tighter scale reproduces the paper's >4-bit trend.

Usage:
    cd ~/Ditto
    python src/validation/reproduce_fig5_bitwidth.py                 # full run
    python src/validation/reproduce_fig5_bitwidth.py --max-images 3   # quick test
    python src/validation/reproduce_fig5_bitwidth.py --calib-pct 99.0 # set percentile

Author: njzhx213
Project: Ditto Reproduction (HPCA 2025)
Phase: Week 1, Day 3-5 (Fig 5)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

TRACE_ROOT = Path.home() / "Ditto" / "traces" / "sdm"
FIG_DIR = Path.home() / "Ditto" / "figs"
RESULT_DIR = Path.home() / "Ditto" / "results"

LAYER_FILES = [
    "conv_in",
    "down_blocks_1_attentions_0",
    "down_blocks_2_attentions_0",
    "mid_block_attentions_0",
    "up_blocks_1_attentions_0",
    "up_blocks_2_attentions_0",
]

# Paper Fig 5 SDM reference (temporal differences)
PAPER_SDM = {"zero": 0.44, "le4_nonzero": 0.52, "gt4": 0.04}

QMAX = 127  # int8


def quantize_dynamic(x: np.ndarray) -> np.ndarray:
    """Per-tensor true-absmax symmetric int8."""
    absmax = float(np.abs(x).max())
    if absmax < 1e-12:
        return np.zeros_like(x, dtype=np.int32)
    scale = absmax / QMAX
    return np.clip(np.round(x / scale), -QMAX, QMAX).astype(np.int32)


def quantize_calib(x: np.ndarray, pct: float, shared_absmax: float) -> np.ndarray:
    """
    Per-tensor PERCENTILE-clip symmetric int8 (approximates Q-Diffusion calibration).
    Uses a precomputed shared_absmax (percentile over the pair) so prev/curr share scale.
    """
    if shared_absmax < 1e-12:
        return np.zeros_like(x, dtype=np.int32)
    scale = shared_absmax / QMAX
    return np.clip(np.round(x / scale), -QMAX, QMAX).astype(np.int32)


def classify_signed_range(diff: np.ndarray) -> tuple[int, int, int]:
    """Count zero / 4-bit / >4-bit using signed-range [-8,7] = 4-bit."""
    zero = int((diff == 0).sum())
    le4_nz = int(((diff >= -8) & (diff <= 7) & (diff != 0)).sum())
    gt4 = int(((diff < -8) | (diff > 7)).sum())
    return zero, le4_nz, gt4


def process_pair(prev: np.ndarray, curr: np.ndarray, calib_pct: float):
    """Return (zero, le4nz, gt4) counts for both dynamic and calib quant."""
    # Dynamic: shared true-absmax over the pair
    absmax_dyn = float(max(np.abs(prev).max(), np.abs(curr).max()))
    scale_dyn = absmax_dyn / QMAX if absmax_dyn > 1e-12 else 1.0
    p_dyn = np.clip(np.round(prev / scale_dyn), -QMAX, QMAX).astype(np.int32)
    c_dyn = np.clip(np.round(curr / scale_dyn), -QMAX, QMAX).astype(np.int32)
    diff_dyn = c_dyn - p_dyn
    dyn_counts = classify_signed_range(diff_dyn)

    # Calib: shared percentile-absmax over the pair
    stacked = np.concatenate([np.abs(prev).ravel(), np.abs(curr).ravel()])
    absmax_calib = float(np.percentile(stacked, calib_pct))
    scale_calib = absmax_calib / QMAX if absmax_calib > 1e-12 else 1.0
    p_cal = np.clip(np.round(prev / scale_calib), -QMAX, QMAX).astype(np.int32)
    c_cal = np.clip(np.round(curr / scale_calib), -QMAX, QMAX).astype(np.int32)
    diff_cal = c_cal - p_cal
    cal_counts = classify_signed_range(diff_cal)

    return dyn_counts, cal_counts


def run(max_images: int, calib_pct: float, step_stride: int):
    t0 = time.time()

    # Aggregate counts
    dyn = {"zero": 0, "le4": 0, "gt4": 0}
    cal = {"zero": 0, "le4": 0, "gt4": 0}
    n_pairs = 0

    image_dirs = sorted(TRACE_ROOT.glob("image_*"))[:max_images]
    if not image_dirs:
        print(f"✗ No image dirs in {TRACE_ROOT}")
        sys.exit(1)

    for img_dir in image_dirs:
        step_dirs = sorted(img_dir.glob("step_*"))
        # adjacent step pairs with stride
        for i in range(0, len(step_dirs) - 1, step_stride):
            prev_dir = step_dirs[i]
            curr_dir = step_dirs[i + 1]
            for layer in LAYER_FILES:
                p_path = prev_dir / f"{layer}.npz"
                c_path = curr_dir / f"{layer}.npz"
                if not (p_path.exists() and c_path.exists()):
                    continue
                prev = np.load(p_path)["input"].astype(np.float32)
                curr = np.load(c_path)["input"].astype(np.float32)
                if prev.shape != curr.shape:
                    continue

                d_counts, c_counts = process_pair(prev, curr, calib_pct)
                dyn["zero"] += d_counts[0]; dyn["le4"] += d_counts[1]; dyn["gt4"] += d_counts[2]
                cal["zero"] += c_counts[0]; cal["le4"] += c_counts[1]; cal["gt4"] += c_counts[2]
                n_pairs += 1

        print(f"  {img_dir.name}: {n_pairs} layer-pairs processed "
              f"({time.time()-t0:.1f}s elapsed)", flush=True)

    def frac(d):
        tot = d["zero"] + d["le4"] + d["gt4"]
        return (d["zero"]/tot, d["le4"]/tot, d["gt4"]/tot) if tot else (0, 0, 0)

    dz, dl, dg = frac(dyn)
    cz, cl, cg = frac(cal)

    print(f"\n=== Fig 5 reproduction (full SDM trace) ===")
    print(f"Layer-pairs processed: {n_pairs}")
    print(f"Calibration percentile: {calib_pct}")
    print()
    print(f"{'scheme':<22} {'zero':>8} {'4-bit':>8} {'>4-bit':>8} {'≤4-bit':>8}")
    print("-" * 60)
    print(f"{'Ours (dynamic)':<22} {dz*100:>7.1f} {dl*100:>7.1f} {dg*100:>7.1f} {(dz+dl)*100:>7.1f}")
    print(f"{'Ours (calib ' + str(calib_pct) + ')':<22} {cz*100:>7.1f} {cl*100:>7.1f} {cg*100:>7.1f} {(cz+cl)*100:>7.1f}")
    print(f"{'Paper SDM':<22} {PAPER_SDM['zero']*100:>7.1f} {PAPER_SDM['le4_nonzero']*100:>7.1f} {PAPER_SDM['gt4']*100:>7.1f} {(PAPER_SDM['zero']+PAPER_SDM['le4_nonzero'])*100:>7.1f}")
    print()

    # Save stats
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    stats = {
        "n_layer_pairs": n_pairs,
        "calib_percentile": calib_pct,
        "dynamic": {"zero": dz, "le4_nonzero": dl, "gt4": dg},
        "calib": {"zero": cz, "le4_nonzero": cl, "gt4": cg},
        "paper_sdm": PAPER_SDM,
    }
    with open(RESULT_DIR / "fig5_stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Stats saved: {RESULT_DIR / 'fig5_stats.json'}")

    # Plot
    make_plot(dz, dl, dg, cz, cl, cg, calib_pct)

    print(f"\nTotal time: {time.time()-t0:.1f}s")


def make_plot(dz, dl, dg, cz, cl, cg, calib_pct):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIG_DIR.mkdir(parents=True, exist_ok=True)

    schemes = ["Ours\n(dynamic)", f"Ours\n(calib {calib_pct})", "Paper\nSDM"]
    zero = [dz, cz, PAPER_SDM["zero"]]
    le4 = [dl, cl, PAPER_SDM["le4_nonzero"]]
    gt4 = [dg, cg, PAPER_SDM["gt4"]]

    x = np.arange(len(schemes))
    w = 0.55

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    b1 = ax.bar(x, zero, w, label="Zero", color="#4C72B0")
    b2 = ax.bar(x, le4, w, bottom=zero, label="4-bit", color="#55A868")
    b3 = ax.bar(x, gt4, w, bottom=np.array(zero)+np.array(le4), label=">4-bit", color="#C44E52")

    ax.set_ylabel("Fraction of temporal differences")
    ax.set_title("Fig 5 reproduction: SDM temporal-difference bit-width", pad=18)
    ax.set_xticks(x)
    ax.set_xticklabels(schemes)
    ax.set_ylim(0, 1.18)   # headroom so >4-bit labels don't hit the title
    # Legend below the plot so it never overlaps the >4-bit annotations on the bars
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.10),
              ncol=3, framealpha=0.95)

    # Value labels inside each segment (zero / 4-bit), and >4-bit just above the bar
    for i in range(len(schemes)):
        if zero[i] > 0.04:
            ax.text(i, zero[i]/2, f"{zero[i]*100:.0f}%", ha="center", va="center",
                    color="white", fontsize=9, fontweight="bold")
        if le4[i] > 0.04:
            ax.text(i, zero[i]+le4[i]/2, f"{le4[i]*100:.0f}%", ha="center", va="center",
                    color="white", fontsize=9, fontweight="bold")
        # >4-bit label above the stacked bar (bars top out near 1.0, headroom above)
        top = zero[i] + le4[i] + gt4[i]
        ax.text(i, top + 0.02, f">4bit: {gt4[i]*100:.1f}%",
                ha="center", va="bottom", fontsize=8.5, color="#C44E52", fontweight="bold")

    plt.tight_layout()
    out = FIG_DIR / "fig5_bitwidth_sdm.png"
    plt.savefig(out, dpi=150)
    print(f"Figure saved: {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-images", type=int, default=20)
    ap.add_argument("--calib-pct", type=float, default=99.0,
                    help="Percentile for calib absmax (lower = tighter = more >4-bit)")
    ap.add_argument("--step-stride", type=int, default=1,
                    help="Stride over step pairs (1 = all adjacent pairs)")
    args = ap.parse_args()

    if not TRACE_ROOT.exists():
        print(f"✗ Trace root not found: {TRACE_ROOT}")
        sys.exit(1)

    run(args.max_images, args.calib_pct, args.step_stride)


if __name__ == "__main__":
    main()

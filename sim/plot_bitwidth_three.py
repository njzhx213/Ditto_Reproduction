#!/usr/bin/env python3
"""
plot_bitwidth_three.py - Ditto temporal-difference bit-width across three workloads.

SDM (image diffusion) | DiT (image diffusion) | Fast-dLLM v2 (TEXT diffusion), all
measured with the SAME quantizer/classifier (comparable). Shows that Ditto's
temporal-value-similarity assumption holds across modalities, and is STRONGEST for
text diffusion.

Fairness note: bars use EQUAL-weight zero rates so the three are directly comparable
(per-layer mean). DiT's MAC-weighted 67.2% is annotated separately, since MAC-weight
is the right scalar for speedup but only DiT's was computed that way; the equal-weight
78.8% is the apples-to-apples number for this cross-workload bar.

    cd ~/Ditto && python3 sim/plot_bitwidth_three.py
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

FIG = Path.home() / "Ditto" / "figs" / "bitwidth_three_workloads.png"

# equal-weight temporal-difference bit-width (zero / <=4bit-nonzero / >4bit)
DATA = {
    "SDM\n(image)":        (0.459, 0.534, 0.007),
    "DiT\n(image)":        (0.788, 0.210, 0.002),   # equal-weight (MAC-wt 67.2% annotated)
    "Fast-dLLM v2\n(text)":(0.803, 0.194, 0.002),
}
COLORS = {"zero": "#4a7fb5", "le4": "#7fb47f", "gt4": "#d98a8a"}


def main():
    names = list(DATA.keys())
    x = np.arange(len(names))
    zero = [DATA[n][0] for n in names]
    le4 = [DATA[n][1] for n in names]
    gt4 = [DATA[n][2] for n in names]

    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.bar(x, zero, color=COLORS["zero"], label="zero (skipped)", width=0.6)
    ax.bar(x, le4, bottom=zero, color=COLORS["le4"], label="<=4-bit", width=0.6)
    ax.bar(x, gt4, bottom=[z + l for z, l in zip(zero, le4)],
           color=COLORS["gt4"], label=">4-bit", width=0.6)

    for i, n in enumerate(names):
        ax.text(i, zero[i] / 2, f"{zero[i]*100:.0f}%", ha="center", va="center",
                color="white", fontweight="bold", fontsize=11)
        ax.text(i, 1.02, f"{(le4[i]+gt4[i])*100:.0f}% computed", ha="center",
                fontsize=8, color="#555")

    # DiT MAC-weighted annotation
    ax.annotate("DiT MAC-weighted\nzero = 67.2%", xy=(1, 0.788), xytext=(1.35, 0.55),
                fontsize=8, color="#2c5a8a",
                arrowprops=dict(arrowstyle="->", color="#2c5a8a", lw=1))

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=10)
    ax.set_ylabel("Fraction of temporal-difference elements")
    ax.set_ylim(0, 1.12)
    ax.set_title("Ditto temporal-difference bit-width across three workloads\n"
                 "same quantizer/classifier; text diffusion is the most temporally sparse",
                 fontsize=10)
    ax.legend(loc="lower right", fontsize=9, frameon=True)
    ax.axhline(1.0, color="gray", ls=":", lw=0.8)
    fig.tight_layout()
    FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG, dpi=140, bbox_inches="tight")
    print(f"Saved: {FIG}")
    print("zero rate: SDM 45.9% < DiT 78.8% (eq) < Fast-dLLM 80.3% -- text most sparse")
    print(">4-bit stays tiny (<=0.7%) for all -- post-LN keeps even text-diffusion")
    print("just-unmasked tokens inside 4-bit (no large-outlier blow-up).")


if __name__ == "__main__":
    main()

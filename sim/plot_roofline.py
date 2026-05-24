#!/usr/bin/env python3
"""
plot_roofline.py - speedup vs bandwidth curve with paper reference lines overlaid.

Reads results/fig13_roofline.json and draws:
  TOP    : diff-only and +Defo speedup vs bandwidth (log-x), with reference lines
           ITC=1.0, paper Ditto=1.5x (marking where our Defo curve crosses it),
           and our attention-aware compute ceiling 9.03x.
  BOTTOM : Defo flip% vs bandwidth, with paper's 14.4% reference line (marking the
           bandwidth region where our flip% matches the paper).

We do NOT reshape into the paper's per-model bar chart (that would collapse our
bandwidth sweep into a single assumed-bandwidth point). Instead the paper's hard
numbers (1.5x, 14.4%) are overlaid as references; the curve shows WHERE we meet
them. The paper's DRAM is unpublished, so bandwidth is the explicit sweep axis;
we do NOT claim the paper used any specific bandwidth.

    cd ~/Ditto && python3 sim/plot_roofline.py
"""
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULT = Path.home() / "Ditto" / "results" / "fig13_roofline.json"
FIG = Path.home() / "Ditto" / "figs" / "fig13_roofline.png"
PAPER_SPEEDUP = 1.5
PAPER_FLIP = 0.144
CEILING = 8.89


def interp_cross(xs, ys, target):
    """Bandwidth where ys first crosses target (linear interp in log-x)."""
    for i in range(1, len(ys)):
        if (ys[i-1] - target) * (ys[i] - target) <= 0 and ys[i] != ys[i-1]:
            lx0, lx1 = np.log10(xs[i-1]), np.log10(xs[i])
            t = (target - ys[i-1]) / (ys[i] - ys[i-1])
            return 10 ** (lx0 + t * (lx1 - lx0))
    return None


def main():
    data = json.loads(RESULT.read_text())
    rows = data["rows"]
    bw = [r["bw"] for r in rows]
    diff = [r["diff_only"] for r in rows]
    defo = [r["defo"] for r in rows]
    flip = [r["flip"] * 100 for r in rows]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 7), sharex=True,
                                   gridspec_kw={"height_ratios": [3, 2]})

    # ---- top: speedup ----
    ax1.plot(bw, defo, "o-", color="#1f5fa8", lw=2, label="Ditto + Defo")
    ax1.plot(bw, diff, "s--", color="#d06010", lw=1.6, label="diff-only (no Defo)")
    ax1.axhline(1.0, color="gray", ls=":", lw=1)
    ax1.text(bw[0], 1.02, "ITC baseline (1.0x)", fontsize=8, color="gray", va="bottom")
    ax1.axhline(PAPER_SPEEDUP, color="green", ls="-.", lw=1.3)
    ax1.text(bw[-1], PAPER_SPEEDUP + 0.1, "paper Ditto 1.5x", fontsize=8,
             color="green", ha="right", va="bottom")
    ax1.axhline(CEILING, color="purple", ls="-.", lw=1.2)
    ax1.text(bw[0], CEILING - 0.5, f"compute ceiling {CEILING}x (attn-aware)",
             fontsize=8, color="purple", va="top")

    cross = interp_cross(bw, defo, PAPER_SPEEDUP)
    if cross:
        ax1.axvline(cross, color="green", ls=":", lw=1, alpha=0.6)
        ax1.plot([cross], [PAPER_SPEEDUP], "g*", ms=14)
        ax1.annotate(f"reaches 1.5x\n@ ~{cross:.0f} B/cyc (~{cross:.0f} GB/s)",
                     xy=(cross, PAPER_SPEEDUP), xytext=(cross * 0.30, 4.6),
                     fontsize=8, color="green", ha="left",
                     arrowprops=dict(arrowstyle="->", color="green", lw=1))

    ax1.set_xscale("log", base=2)
    ax1.set_ylabel("Speedup vs ITC")
    ax1.set_title("Ditto speedup vs DRAM bandwidth (SDM UNet, 401.6 GMAC incl. attention)\n"
                  "paper DRAM unpublished -> bandwidth is the explicit sweep axis",
                  fontsize=10)
    ax1.legend(loc="upper left", fontsize=9)
    ax1.grid(True, which="both", alpha=0.25)
    ax1.set_ylim(0, CEILING + 1)

    # ---- bottom: flip % ----
    ax2.plot(bw, flip, "o-", color="#1f5fa8", lw=2, label="Defo flip %")
    ax2.axhline(PAPER_FLIP * 100, color="green", ls="-.", lw=1.3)
    ax2.text(bw[0], PAPER_FLIP * 100 + 2, "paper Defo flip 14.4%",
             fontsize=8, color="green", va="bottom")
    fcross = interp_cross(bw, flip, PAPER_FLIP * 100)
    if fcross:
        ax2.axvline(fcross, color="green", ls=":", lw=1, alpha=0.6)
        ax2.plot([fcross], [PAPER_FLIP * 100], "g*", ms=14)
        ax2.annotate(f"flip = 14.4%\n@ ~{fcross:.0f} B/cyc",
                     xy=(fcross, PAPER_FLIP * 100), xytext=(fcross * 0.3, 40),
                     fontsize=8, color="green",
                     arrowprops=dict(arrowstyle="->", color="green", lw=1))
    ax2.set_xscale("log", base=2)
    ax2.set_xlabel("DRAM bandwidth (bytes/cycle ~ GB/s at 1 GHz)")
    ax2.set_ylabel("Defo layers\nflipped to act (%)")
    ax2.grid(True, which="both", alpha=0.25)
    ax2.set_ylim(0, 105)
    ax2.legend(loc="upper right", fontsize=9)

    fig.tight_layout()
    FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG, dpi=140, bbox_inches="tight")
    print(f"Saved: {FIG}")
    if cross:
        print(f"Defo reaches paper 1.5x at ~{cross:.0f} B/cyc (~{cross:.0f} GB/s).")
    if fcross:
        print(f"Defo flip matches paper 14.4% at ~{fcross:.0f} B/cyc "
              f"(~{fcross/1024:.1f} TB/s) -> reverse-estimate of the paper's DRAM class.")
    print("Honest: paper reports 1.5x and 14.4% (its DRAM unpublished); these are")
    print("overlaid as references. We do NOT claim the paper used any specific BW.")


if __name__ == "__main__":
    main()

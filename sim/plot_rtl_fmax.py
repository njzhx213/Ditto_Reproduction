#!/usr/bin/env python3
"""
plot_rtl_fmax.py - bar chart of the measured synthesis Fmax (Vivado 2025.1, ZU3EG, OOC).

Shows the headline RTL finding: the three carry-propagate PE variants all hit the same
~600 MHz wall (their critical path is the 32-bit accumulator carry chain), while the
carry-save accumulator removes that chain and reaches 3021 MHz -- a 5x improvement.
The pipeline (pe_diff_pipe) is *slightly slower* than the plain PE, because it split the
wrong logic. All numbers are measured, not modeled.

Output: figs/rtl_fmax.png
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# measured results (Vivado 2025.1, xczu3eg-sbva484-2-e, out-of-context, 1.0 ns target)
labels = ["pe_diff\n(single-cycle)", "pe_diff_pipe\n(3-stage)",
          "pe_diff_slot\n(4-bit/wide)", "pe_diff_csa\n(carry-save)"]
fmax   = [599.9, 575.7, 612.4, 3021.1]
# first three share the accumulator-carry-chain bottleneck; the fourth removes it
colors = ["#5b8fb5", "#5b8fb5", "#5b8fb5", "#d98a3d"]

fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(range(len(labels)), fmax, color=colors, width=0.62,
              edgecolor="black", linewidth=0.6, zorder=3)

# value labels on top of each bar
for i, (b, v) in enumerate(zip(bars, fmax)):
    ax.text(b.get_x() + b.get_width() / 2, v + 45, f"{v:.1f}",
            ha="center", va="bottom", fontsize=11, fontweight="bold")

# a light band marking the ~600 MHz wall the carry-propagate versions share
ax.axhspan(560, 620, color="#5b8fb5", alpha=0.10, zorder=0)
ax.text(1.0, 1080, "carry-propagate PEs share the ~600 MHz wall\n"
                   "(critical path = 32-bit accumulator carry chain)",
        ha="center", va="bottom", fontsize=9, color="#33556e")
# a thin connector from the note down toward the band
ax.annotate("", xy=(1.0, 630), xytext=(1.0, 1070),
            arrowprops=dict(arrowstyle="-", color="#33556e", lw=0.7, alpha=0.6))

# 5x annotation between the group and the carry-save bar
ax.annotate("", xy=(3, 3021.1), xytext=(3, 612.4),
            arrowprops=dict(arrowstyle="<->", color="#b5651d", lw=1.4))
ax.text(3.18, (3021.1 + 612.4) / 2, "5x\n(remove the\ncarry chain)",
        ha="left", va="center", fontsize=10, color="#b5651d", fontweight="bold")

ax.set_xticks(range(len(labels)))
ax.set_xticklabels(labels, fontsize=10)
ax.set_ylabel("Achieved Fmax (MHz)", fontsize=11)
ax.set_ylim(0, 3400)
ax.set_title("Ditto PE synthesis Fmax — Vivado 2025.1, ZU3EG (out-of-context)\n"
             "pipelining the wrong path: no gain; carry-save on the real bottleneck: 5x",
             fontsize=11)
ax.grid(axis="y", linestyle=":", alpha=0.4, zorder=0)

legend = [Patch(facecolor="#5b8fb5", edgecolor="black",
                label="carry-propagate accumulator (carry-chain bound)"),
          Patch(facecolor="#d98a3d", edgecolor="black",
                label="carry-save accumulator (carry-chain removed)")]
ax.legend(handles=legend, loc="upper left", fontsize=9, framealpha=0.9)

plt.tight_layout()
os.makedirs("figs", exist_ok=True)
out = "figs/rtl_fmax.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"saved {out}")

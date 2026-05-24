#!/usr/bin/env python3
"""
plot_energy.py - stacked-bar energy breakdown matching paper Fig 13 (SDM).

Bars: ITC (baseline=1.0), Cambricon-D (full-precision difference), Ditto+Defo.
Each bar stacked into six segments: core / sram / dram / eu / vpu / defo, same as
the paper. Reproduces the paper's qualitative structure:
  - ITC is Core-dominated (memory small)            -> matches Fig 13 ITC bar
  - Cam-D inverts ABOVE ITC, driven by DRAM blow-up  -> matches Fig 13 Cam-D bar
  - Ditto+Defo sits below ITC (energy saving)        -> matches Fig 13 Ditto bar

Numbers are recomputed from energy_model.py so the figure can't drift from it.
Honest: SRAM per-access is CACTI-measured; MAC/DRAM public 45nm; EU/VPU/Defo
modeled; effective buffer / VPU energy are the swept unknowns (not tuned).

    cd ~/Ditto && python3 sim/plot_energy.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import energy_model as E

FIG = Path.home() / "Ditto" / "figs" / "fig13_energy.png"

SEG_COLORS = {
    "core": "#f4a582",   # orange (Core) - paper's dominant ITC segment
    "sram": "#f7d774",   # yellow (SRAM)
    "dram": "#9fd49f",   # green (DRAM) - the segment that blows up for Cam-D
    "eu":   "#a6cee3",   # light blue (EU)
    "vpu":  "#b8b8d8",   # grey-violet (VPU)
    "defo": "#404040",   # dark (Defo)
}
SEG_ORDER = ["dram", "sram", "core", "vpu", "eu", "defo"]  # bottom->top like paper


def main():
    layers = E.enumerate_with_attention()
    itc = E.total_energy(layers, "itc")
    camd = E.total_energy(layers, "camd")
    flip = E.compute_flip(layers)
    defo = E.total_energy(layers, "ditto", defo_flip=flip)
    itc_tot = E.etotal(itc)

    bars = [("ITC", itc), ("Cam-D", camd), ("Ditto+Defo", defo)]
    labels = [b[0] for b in bars]
    x = range(len(bars))

    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    bottoms = [0.0] * len(bars)
    for seg in SEG_ORDER:
        vals = [b[1][seg] / itc_tot for b in bars]
        ax.bar(x, vals, bottom=bottoms, color=SEG_COLORS[seg],
               edgecolor="white", linewidth=0.6, label=seg.upper(), width=0.6)
        bottoms = [bt + v for bt, v in zip(bottoms, vals)]

    # totals + saving annotation
    for i, (name, e) in enumerate(bars):
        tot = E.etotal(e) / itc_tot
        ax.text(i, tot + 0.02, f"{tot:.2f}", ha="center", fontsize=9, fontweight="bold")

    ax.axhline(1.0, color="gray", ls=":", lw=1)
    ax.text(len(bars) - 0.5, 1.01, "ITC = 1.0", fontsize=8, color="gray",
            ha="right", va="bottom")

    saving = (1 - E.etotal(defo) / itc_tot) * 100
    ax.annotate(f"Ditto saves {saving:.0f}%\n(paper 17.74%; depends on\neffective buffer, see sweep)",
                xy=(2, E.etotal(defo) / itc_tot), xytext=(1.4, 0.30),
                fontsize=8, color="#1f5fa8",
                arrowprops=dict(arrowstyle="->", color="#1f5fa8", lw=1))
    ax.annotate("Cam-D inverts\n(DRAM blow-up)", xy=(1, E.etotal(camd) / itc_tot),
                xytext=(0.55, 1.45), fontsize=8, color="#2c7a2c",
                arrowprops=dict(arrowstyle="->", color="#2c7a2c", lw=1))

    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Relative energy (normalized to ITC)")
    ax.set_title("Ditto energy breakdown, SDM UNet (six segments, cf. paper Fig 13)\n"
                 "SRAM CACTI-measured 45nm; EU/VPU/Defo modeled", fontsize=10)
    ax.legend(ncol=6, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.08),
              frameon=False)
    ax.set_ylim(0, max(E.etotal(camd) / itc_tot, 1.0) + 0.25)
    fig.tight_layout()
    FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG, dpi=140, bbox_inches="tight")
    print(f"Saved: {FIG}")
    print(f"ITC=1.00  Cam-D={E.etotal(camd)/itc_tot:.2f}  "
          f"Ditto+Defo={E.etotal(defo)/itc_tot:.2f} (saves {saving:.1f}%)")
    print(f"ITC core fraction = {itc['core']/itc_tot:.0%} (paper Fig 13 ITC ~ Core-dominated)")


if __name__ == "__main__":
    main()

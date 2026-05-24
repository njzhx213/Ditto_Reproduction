#!/usr/bin/env python3
"""
plot_three_workload_trend.py - the SDM -> DiT -> Fast-dLLM Ditto trend.

Two panels telling one story: as the workload becomes more linear-dominated and more
temporally sparse (SDM image -> DiT image -> Fast-dLLM text), Ditto's benefit grows
monotonically and Cambricon-D's inversion shrinks.

Left  : energy saving (up) and Cam-D inversion x ITC (down) per workload.
Right : temporal-difference zero rate (up) vs attention MAC fraction (down) -- the two
        structural drivers, moving in opposite directions across the three workloads.

All numbers are from the per-workload runs (energy_model / dit_recompute /
fastdllm_structure); sparsity is each workload's value as actually used in its model
(SDM 45.9% dynamic, DiT 67.2% MAC-weighted, Fast-dLLM 80.3%).

    cd ~/Ditto && python3 sim/plot_three_workload_trend.py
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

FIG = Path.home() / "Ditto" / "figs" / "three_workload_trend.png"

W = ["SDM\n(image)", "DiT\n(image)", "Fast-dLLM v2\n(text)"]
SAVING = [34.0, 46.0, 69.7]        # energy saving % (Ditto+Defo vs ITC)
CAMD = [1.34, 1.14, 1.04]          # Cam-D x ITC (inversion)
ZERO = [45.9, 67.2, 80.3]          # temporal zero % (as used per model)
ATTN = [15.7, 3.6, 0.8]            # attention MAC fraction %
CEIL = [8.89, 16.49, 28.42]        # theoretical compute ceiling x


def main():
    x = np.arange(3)
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(12, 5))

    # left: saving (bars) + Cam-D (line, right axis)
    axL.bar(x, SAVING, color="#5b9bd5", width=0.55, label="Ditto energy saving")
    for i, v in enumerate(SAVING):
        axL.text(i, v + 1, f"{v:.0f}%", ha="center", fontsize=10, fontweight="bold")
    axL.set_ylabel("Ditto energy saving vs ITC (%)", color="#2c5a8a")
    axL.set_ylim(0, 85)
    axL.set_xticks(x); axL.set_xticklabels(W)
    axL2 = axL.twinx()
    axL2.plot(x, CAMD, "o-", color="#c0504d", lw=2, label="Cam-D x ITC (inversion)")
    for i, v in enumerate(CAMD):
        axL2.text(i, v + 0.01, f"{v:.2f}x", ha="center", fontsize=9, color="#c0504d")
    axL2.set_ylabel("Cambricon-D energy x ITC", color="#a03c39")
    axL2.set_ylim(1.0, 1.45)
    axL.set_title("Benefit grows; Cam-D inversion shrinks", fontsize=10)
    axL.text(0.02, 0.96, "more saving ->", transform=axL.transAxes, fontsize=8, color="#2c5a8a")

    # right: zero rate (bars) + attention fraction (line, right axis)
    axR.bar(x, ZERO, color="#70ad47", width=0.55, label="temporal zero rate")
    for i, v in enumerate(ZERO):
        axR.text(i, v + 1, f"{v:.0f}%", ha="center", fontsize=10, fontweight="bold")
    axR.set_ylabel("Temporal-difference zero rate (%)", color="#4a7a30")
    axR.set_ylim(0, 95)
    axR.set_xticks(x); axR.set_xticklabels(W)
    axR2 = axR.twinx()
    axR2.plot(x, ATTN, "s--", color="#7030a0", lw=2, label="attention MAC fraction")
    for i, v in enumerate(ATTN):
        axR2.text(i, v + 0.4, f"{v:.1f}%", ha="center", fontsize=9, color="#7030a0")
    axR2.set_ylabel("Attention MAC fraction (%)", color="#5a2480")
    axR2.set_ylim(0, 20)
    axR.set_title("More temporally sparse; less attention-bound", fontsize=10)

    fig.suptitle("Ditto across three workloads: SDM -> DiT -> Fast-dLLM v2\n"
                 "more linear-dominated + more temporally sparse -> larger Ditto benefit "
                 "(theoretical ceiling 8.9x / 16.5x / 28.4x)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG, dpi=140, bbox_inches="tight")
    print(f"Saved: {FIG}")
    print("Monotonic trend across SDM -> DiT -> Fast-dLLM:")
    print(f"  saving   {SAVING}  (up)")
    print(f"  Cam-D    {CAMD}  (down, weaker inversion)")
    print(f"  zero%    {ZERO}  (up, more temporally sparse)")
    print(f"  attn%    {ATTN}  (down, more linear-dominated)")
    print(f"  ceiling  {CEIL}  (theoretical, not attainable)")


if __name__ == "__main__":
    main()

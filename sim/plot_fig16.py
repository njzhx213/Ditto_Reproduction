#!/usr/bin/env python3
"""
plot_fig16.py - formal reproduction of paper Fig 16.

Paper Fig 16 point: temporal-difference processing WITHOUT Defo can be SLOWER
than the ITC baseline under memory pressure; Defo recovers it by flipping
memory-bound layers back to original-activation execution.

Reads the existing results/fig13_sweep.json (produced by fig13_speedup.py) and
draws three curves vs the bandwidth parameter:
    - diff-only  (Ditto with NO Defo)      -> dips below 1.0 = slower than ITC
    - Ditto+Defo                            -> held at/above 1.0
    - ITC = 1.0 baseline                    -> reference line

Deliberately does NOT plot Defo flip-rate or a "paper 14.4%" reference: our flip
rate is a different measurement regime (analytical memory model, compute-heavy
trace layers only) and putting paper's 14.4% beside it implies a comparison that
isn't valid. We keep one neutral "paper 1.5x SDM target" reference.

Honesty note printed on the figure: the x-axis is the analytical model's
bandwidth PARAMETER (bytes/cycle), not a measured physical bandwidth.

    cd ~/Ditto && python3 sim/plot_fig16.py
"""
import json
import sys
from pathlib import Path

RESULT = Path.home() / "Ditto" / "results" / "fig13_sweep.json"
OUT = Path.home() / "Ditto" / "figs" / "fig16_defo_rescue.png"


def load_rows():
    if not RESULT.exists():
        sys.exit(f"{RESULT} not found -- run fig13_speedup.py first.")
    d = json.load(open(RESULT))
    rows = d.get("bandwidth_sweep")
    if not rows:
        sys.exit(f"'bandwidth_sweep' missing in {RESULT}; keys: {list(d.keys())}")
    need = {"bpc", "speedup_diff_only", "speedup_defo"}
    missing = need - set(rows[0].keys())
    if missing:
        sys.exit(f"row missing keys {missing}; row keys: {list(rows[0].keys())}")
    return rows


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = load_rows()
    bpc = [r["bpc"] for r in rows]
    diff = [r["speedup_diff_only"] for r in rows]
    defo = [r["speedup_defo"] for r in rows]

    fig, ax = plt.subplots(figsize=(8, 5))

    # shade the region where diff-only is SLOWER than ITC (the paper's point)
    ax.axhspan(min(min(diff), 0.5), 1.0, color="#C44E52", alpha=0.06)

    ax.semilogx(bpc, diff, "o--", color="#C44E52", lw=2, ms=6,
                label="Difference-only (no Defo)")
    ax.semilogx(bpc, defo, "o-", color="#4C72B0", lw=2.2, ms=6,
                label="Ditto + Defo")
    ax.axhline(1.0, color="#333", ls="-", lw=1.1, label="ITC baseline (1.0x)")
    ax.axhline(1.5, color="#55A868", ls=":", lw=1.4, label="paper 1.5x SDM target")

    # annotate the "slower than baseline" finding
    lo_i = min(range(len(diff)), key=lambda i: diff[i])
    ax.annotate(f"diff-only {diff[lo_i]:.2f}x\n(slower than ITC)",
                xy=(bpc[lo_i], diff[lo_i]),
                xytext=(bpc[lo_i] * 2.2, diff[lo_i] - 0.0 + 0.18),
                fontsize=9, color="#C44E52",
                arrowprops=dict(arrowstyle="->", color="#C44E52", lw=1))

    ax.set_xlabel("Bandwidth parameter  (analytical model bytes/cycle; "
                  "not a measured physical bandwidth)", fontsize=9.5)
    ax.set_ylabel("Speedup over ITC")
    ax.set_title("Fig 16: without Defo, temporal-difference can be slower "
                 "than baseline;\nDefo recovers it", fontsize=11)
    ax.legend(loc="upper left", fontsize=9.5, framealpha=0.95)
    ax.grid(True, which="both", ls=":", alpha=0.3)

    # caption strip
    cap = ("Difference-only dips below 1.0x under tight memory (slower than ITC, "
           "the paper's Fig 16 point);\nDefo flips memory-bound layers back to "
           "activation execution and holds speedup >= 1.0x.")
    fig.text(0.5, -0.02, cap, ha="center", fontsize=8.2, color="#444")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(OUT, dpi=150, bbox_inches="tight")
    print(f"Saved: {OUT}")

    # also print the key numbers so they go into the summary doc
    print("\nKey numbers for the writeup:")
    print(f"  diff-only min speedup: {min(diff):.2f}x at bpc={bpc[diff.index(min(diff))]}")
    print(f"  Defo at that point:    {defo[diff.index(min(diff))]:.2f}x")
    print(f"  diff-only max speedup: {max(diff):.2f}x;  Defo max: {max(defo):.2f}x")
    crossed = [r for r in rows if r["speedup_defo"] >= 1.5]
    if crossed:
        print(f"  Defo crosses 1.5x at bpc={crossed[0]['bpc']} "
              f"({crossed[0]['speedup_defo']:.2f}x)")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
plot_fig13_both.py - SDM and DiT six-segment energy side by side (paper Fig 13 layout).

Paper Fig 13 shows SDM and DiT as two groups. We mirror that: left group SDM, right
group DiT, each with ITC / Cam-D / Ditto+Defo bars stacked into core/sram/dram/eu/
vpu/defo. Each workload uses ITS OWN bit-width constants (SDM: zero 45.9%; DiT: the
MAC-weighted 67.2% measured from the DiT trace), so the two groups are each correct,
not one borrowing the other's sparsity. Values are recomputed live from the model so
the figure can't drift.

    cd ~/Ditto && python3 sim/plot_fig13_both.py
"""
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from diffusers import DiTTransformer2DModel

import energy_model as EN
import dit_structure as DS

FIG = Path.home() / "Ditto" / "figs" / "fig13_energy_three.png"
TRACE = Path.home() / "Ditto" / "traces" / "dit"
QMAX = 127

SEG_COLORS = {"core": "#f4a582", "sram": "#f7d774", "dram": "#9fd49f",
              "eu": "#a6cee3", "vpu": "#b8b8d8", "defo": "#404040"}
SEG_ORDER = ["dram", "sram", "core", "vpu", "eu", "defo"]


# ---- DiT's own MAC-weighted bit-width (same as dit_recompute) ----
def dit_constants():
    def parse(p):
        m = re.match(r"c(\d+)_b(\d+)_(.+)_t(\d+)\.npz", p.name)
        return (int(m.group(1)), int(m.group(2)), m.group(3), int(m.group(4))) if m else None
    files = sorted(TRACE.glob("*.npz"))
    acc = defaultdict(lambda: [0, 0, 0])
    groups = defaultdict(dict)
    for p in files:
        info = parse(p)
        if info:
            c, b, l, t = info
            groups[(c, b, l)][t] = p
    for (c, b, l), sm in groups.items():
        steps = sorted(sm)
        for i in range(len(steps) - 1):
            d0 = np.load(sm[steps[i]], allow_pickle=True)
            d1 = np.load(sm[steps[i + 1]], allow_pickle=True)
            prev = d0["input"].astype(np.float32)
            curr = d1["input"].astype(np.float32)
            if prev.shape != curr.shape:
                continue
            am = float(max(np.abs(prev).max(), np.abs(curr).max()))
            s = am / QMAX if am > 1e-12 else 1.0
            diff = (np.clip(np.round(curr / s), -QMAX, QMAX)
                    - np.clip(np.round(prev / s), -QMAX, QMAX)).astype(np.int32)
            a = acc[(b, l)]
            a[0] += int((diff == 0).sum())
            a[1] += int(((diff >= -8) & (diff <= 7) & (diff != 0)).sum())
            a[2] += int(((diff < -8) | (diff > 7)).sum())
    rates = {}
    for (b, l), (z, le, gt) in acc.items():
        t = z + le + gt
        rates[(b, l)] = (z / t, le / t, gt / t) if t else (0, 0, 0)

    def seg(bidx, lt):
        rep = [0] if bidx <= 4 else ([27] if bidx >= 23 else [9, 18])
        zs = [rates[(b, lt)] for b in rep if (b, lt) in rates]
        return tuple(np.mean([z[k] for z in zs]) for k in range(3)) if zs else (0.5, 0.5, 0.0)

    m = DiTTransformer2DModel.from_pretrained(
        "facebook/DiT-XL-2-256", subfolder="transformer").eval()
    shapes = {}

    def hook(nm):
        def h(mod, inp, out):
            y = out[0] if isinstance(out, (tuple, list)) else out
            shapes[nm] = (tuple(inp[0].shape), tuple(y.shape))
        return h
    hs = [mm.register_forward_hook(hook(n)) for n, mm in m.named_modules()
          if isinstance(mm, nn.Linear)]
    with torch.no_grad():
        m(torch.randn(1, 4, 32, 32), timestep=torch.tensor([1]), class_labels=torch.tensor([0]))
    for h in hs:
        h.remove()

    def lt_of(name):
        if "attn" in name and any(t in name for t in ("to_q", "to_k", "to_v", "to_out")):
            return "attn1-to_q"
        return "ff-net-2" if "ff.net.2" in name else "ff-net-0-proj"
    num = np.zeros(3); den = 0.0
    for n, (in_s, out_s) in shapes.items():
        mb = re.search(r"transformer_blocks\.(\d+)\.", n)
        if not mb:
            continue
        M = 1
        for d in in_s[:-1]:
            M *= d
        macs = M * in_s[-1] * out_s[-1]
        num += np.array(seg(int(mb.group(1)), lt_of(n))) * macs
        den += macs
    return tuple(num / den)   # (zero, le4, gt4)


def energy_groups(lef, gtf, layers):
    """Run six-segment energy with given bit-width constants; return ITC/Cam-D/Ditto.
    Energy's compute term uses only the nonzero split (le4/gt4); the zero fraction
    affects skip but compute_energy already scales by NONZERO."""
    EN.BW_LE4, EN.BW_GT4 = lef, gtf
    EN.NONZERO = lef + gtf
    EN.E_MAC_DIFF = (lef * EN.E_MAC4 + gtf * EN.E_MAC8) / EN.NONZERO
    itc = EN.total_energy(layers, "itc")
    camd = EN.total_energy(layers, "camd")
    flip = EN.compute_flip(layers)
    defo = EN.total_energy(layers, "ditto", defo_flip=flip)
    it = EN.etotal(itc)
    return {"ITC": itc, "Cam-D": camd, "Ditto": defo}, it


def main():
    # SDM with SDM constants
    sdm_layers = EN.enumerate_with_attention()
    sdm_g, sdm_it = energy_groups(0.534, 0.007, sdm_layers)

    # DiT with DiT's own constants
    print("Measuring DiT bit-width for the figure ...", flush=True)
    zf, lef, gtf = dit_constants()
    print(f"DiT MAC-weighted: zero {zf*100:.1f}% le4 {lef*100:.1f}% gt4 {gtf*100:.1f}%")
    dit_layers = DS.enumerate_dit()
    dit_g, dit_it = energy_groups(lef, gtf, dit_layers)

    # Fast-dLLM v2 with its own measured constants (load 7B, GQA enumeration)
    print("Loading Fast-dLLM v2 (7B) for the figure ...", flush=True)
    import fastdllm_structure as FD
    fd_layers, _, _ = FD.enumerate_fastdllm()
    fd_g, fd_it = energy_groups(FD.FDLLM_LE4, FD.FDLLM_GT4, fd_layers)
    print(f"Fast-dLLM: zero {FD.FDLLM_ZERO*100:.1f}% (le4 {FD.FDLLM_LE4*100:.1f}%, "
          f"gt4 {FD.FDLLM_GT4*100:.1f}%)")

    fig, ax = plt.subplots(figsize=(11.5, 5.4))
    bar_labels = ["ITC", "Cam-D", "Ditto"]
    groups = [("SDM", sdm_g, sdm_it), ("DiT", dit_g, dit_it),
              ("Fast-dLLM v2", fd_g, fd_it)]
    x = 0
    xticks, xticklabels = [], []
    group_centers = []
    for gname, g, it in groups:
        start = x
        for bl in bar_labels:
            e = g[bl]
            bottom = 0.0
            for seg in SEG_ORDER:
                v = e[seg] / it
                ax.bar(x, v, bottom=bottom, color=SEG_COLORS[seg],
                       edgecolor="white", linewidth=0.5, width=0.8,
                       label=seg.upper() if (x == 0) else None)
                bottom += v
            ax.text(x, bottom + 0.02, f"{bottom:.2f}", ha="center", fontsize=8)
            xticks.append(x); xticklabels.append(bl)
            x += 1
        group_centers.append((start + x - 1) / 2)
        x += 1  # gap between groups

    ax.axhline(1.0, color="gray", ls=":", lw=1)
    for c, (gname, _, _) in zip(group_centers, groups):
        ax.text(c, -0.16, gname, ha="center", fontsize=12, fontweight="bold")
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels, fontsize=9)
    ax.set_ylabel("Relative energy (normalized to each workload's ITC)")
    ax.set_title("Six-segment energy: SDM vs DiT vs Fast-dLLM v2 (paper Fig 13 layout)\n"
                 "each workload uses its own measured bit-width "
                 "(zero 45.9% / 67.2% / 80.3%); SRAM CACTI-measured", fontsize=10)
    ax.legend(ncol=6, fontsize=8, loc="upper center",
              bbox_to_anchor=(0.5, -0.12), frameon=False)
    ax.set_ylim(0, 1.55)
    fig.tight_layout()
    FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG, dpi=140, bbox_inches="tight")
    print(f"Saved: {FIG}")
    for gname, g, it in groups:
        print(f"  {gname}: ITC 1.00  Cam-D {EN.etotal(g['Cam-D'])/it:.2f}  "
              f"Ditto {EN.etotal(g['Ditto'])/it:.2f}")


if __name__ == "__main__":
    main()

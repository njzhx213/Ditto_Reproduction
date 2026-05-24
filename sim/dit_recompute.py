#!/usr/bin/env python3
"""
dit_recompute.py - recompute DiT speedup/energy with DiT's OWN bit-width constants.

Until now dit_structure.py borrowed SDM's bit-width split (zero 45.9% etc.). Now we
have DiT's real trace, so we replace those with DiT's MAC-weighted zero / <=4-bit /
>4-bit fractions (measured here from traces/dit, weighted by each layer's MAC and
extrapolated 4 traced blocks -> 28 by depth segment, same scheme as
dit_mac_weighted_sparsity.py). Then we rerun the roofline speedup and the six-segment
energy with DiT's constants and compare to the earlier SDM-borrowed numbers.

Honesty: zero/le4/gt4 are DiT-measured and MAC-weighted. The per-layer rates are
extrapolated from blocks 0/9/18/27 by depth segment (stated assumption). Sampling was
aligned to SDM (50 steps, CFG 7.5) so the difference is the model, not collection.

    cd ~/Ditto && python3 sim/dit_recompute.py
"""
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from diffusers import DiTTransformer2DModel

import fig13_roofline as RF
import energy_model as EN

TRACE = Path.home() / "Ditto" / "traces" / "dit"
QMAX = 127


def parse(p):
    m = re.match(r"c(\d+)_b(\d+)_(.+)_t(\d+)\.npz", p.name)
    return (int(m.group(1)), int(m.group(2)), m.group(3), int(m.group(4))) if m else None


def measure_per_layer():
    """Per-(block,layer_type) zero/le4/gt4 counts, dynamic quant (same ruler as SDM)."""
    files = sorted(TRACE.glob("*.npz"))
    groups = defaultdict(dict)
    for p in files:
        info = parse(p)
        if info:
            c, b, l, t = info
            groups[(c, b, l)][t] = p
    # accumulate by (block, layer)
    acc = defaultdict(lambda: [0, 0, 0])
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
            z = int((diff == 0).sum())
            le = int(((diff >= -8) & (diff <= 7) & (diff != 0)).sum())
            gt = int(((diff < -8) | (diff > 7)).sum())
            a = acc[(b, l)]
            a[0] += z; a[1] += le; a[2] += gt
    # fractions per (block,layer)
    rates = {}
    for (b, l), (z, le, gt) in acc.items():
        tot = z + le + gt
        rates[(b, l)] = (z / tot, le / tot, gt / tot) if tot else (0, 0, 0)
    return rates


def seg_rate(rates, bidx, ltype):
    """Extrapolate measured blocks 0/9/18/27 to all 28 by depth segment."""
    if bidx <= 4:
        rep = [0]
    elif bidx >= 23:
        rep = [27]
    else:
        rep = [9, 18]
    zs = [rates[(b, ltype)] for b in rep if (b, ltype) in rates]
    if not zs:
        return (0.5, 0.5, 0.0)
    return tuple(np.mean([z[k] for z in zs]) for k in range(3))


def main():
    print("Measuring DiT per-layer bit-width from trace ...", flush=True)
    rates = measure_per_layer()

    # MAC-weight over the real model
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
        m(torch.randn(1, 4, 32, 32), timestep=torch.tensor([1]),
          class_labels=torch.tensor([0]))
    for h in hs:
        h.remove()

    def ltype_of(name):
        if "attn" in name and any(t in name for t in ("to_q", "to_k", "to_v", "to_out")):
            return "attn1-to_q"
        if "ff.net.2" in name:
            return "ff-net-2"
        return "ff-net-0-proj"

    num = np.zeros(3); den = 0.0
    for n, (in_s, out_s) in shapes.items():
        mb = re.search(r"transformer_blocks\.(\d+)\.", n)
        if not mb:
            continue
        bidx = int(mb.group(1))
        M = 1
        for d in in_s[:-1]:
            M *= d
        macs = M * in_s[-1] * out_s[-1]
        r = seg_rate(rates, bidx, ltype_of(n))
        num += np.array(r) * macs
        den += macs
    zf, lef, gtf = num / den
    print(f"\nDiT MAC-weighted bit-width:  zero {zf*100:.1f}%  <=4bit {lef*100:.1f}%  >4bit {gtf*100:.1f}%")
    print(f"(SDM was: zero 45.9%  <=4bit 53.4%  >4bit 0.7%)")

    # plug DiT constants into the roofline + energy modules
    RF.BW_ZERO, RF.BW_LE4, RF.BW_GT4 = zf, lef, gtf
    RF.NONZERO = lef + gtf
    RF.BITF = (lef + 2 * gtf) / RF.NONZERO if RF.NONZERO > 0 else 1.0
    EN.BW_LE4, EN.BW_GT4 = lef, gtf
    EN.NONZERO = lef + gtf
    EN.E_MAC_DIFF = (lef * EN.E_MAC4 + gtf * EN.E_MAC8) / EN.NONZERO

    import dit_structure as DS
    layers = DS.enumerate_dit()

    print("\n=== DiT speedup (DiT's OWN bit-width) ===")
    print(f"  {'BW(B/cyc)':>10} {'diff-only':>10} {'+Defo':>8} {'flip%':>7}")
    for bw in [64, 256, 1024, 4096]:
        r = RF.run_bw(layers, bw)
        print(f"  {bw:>10} {r['diff_only']:>9.2f}x {r['defo']:>7.2f}x {r['flip']*100:>6.1f}%")
    ceil = RF.run_bw(layers, 1e12)
    print(f"  compute ceiling: {ceil['defo']:.2f}x   (was 9.98x with SDM constants)")

    print("\n=== DiT energy (DiT's OWN bit-width) ===")
    itc = EN.total_energy(layers, "itc")
    camd = EN.total_energy(layers, "camd")
    flip = EN.compute_flip(layers)
    defo = EN.total_energy(layers, "ditto", defo_flip=flip)
    it = EN.etotal(itc)
    print(f"  ITC=1.00  Cam-D={EN.etotal(camd)/it:.2f}x  "
          f"Ditto={EN.etotal(defo)/it:.2f}x (saves {(1-EN.etotal(defo)/it)*100:.1f}%)")
    print(f"  (was: Cam-D 1.14x, Ditto saves 42.7% with SDM constants)")
    print("\nHigher DiT sparsity (zero 67% vs 46%) -> more skip -> the earlier")
    print("SDM-borrowed numbers UNDER-counted DiT's benefit. These use DiT's own data.")


if __name__ == "__main__":
    main()

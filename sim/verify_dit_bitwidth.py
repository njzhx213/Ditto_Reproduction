#!/usr/bin/env python3
"""
verify_dit_bitwidth.py - sanity-check the DiT 78.8% temporal-difference zero rate.

78.8% (vs SDM 45.9%) is high enough to self-check before trusting it. Three checks:
  1. per-(block,layer) zero rate -- is it uniform (real) or do a few layers sit near
     100% (a degenerate/constant-activation artifact pulling up the mean)?
  2. activation magnitude -- if some layers' absmax ~0, prev/curr both quantize to 0
     and diff is trivially zero (false sparsity).
  3. one sampled pair's raw diff histogram -- confirm zeros are genuine small changes.

    cd ~/Ditto && python3 sim/verify_dit_bitwidth.py
"""
from collections import defaultdict
from pathlib import Path
import re
import numpy as np

TRACE = Path.home() / "Ditto" / "traces" / "dit"
QMAX = 127


def parse(p):
    m = re.match(r"c(\d+)_b(\d+)_(.+)_t(\d+)\.npz", p.name)
    return (int(m.group(1)), int(m.group(2)), m.group(3), int(m.group(4))) if m else None


def main():
    files = sorted(TRACE.glob("*.npz"))
    groups = defaultdict(dict)
    for p in files:
        info = parse(p)
        if info:
            c, b, l, t = info
            groups[(c, b, l)][t] = p

    # per (block,layer) aggregated over classes+steps
    bl_zero = defaultdict(lambda: [0, 0])     # key (block,layer) -> [zero, total]
    bl_absmax = defaultdict(list)
    sample_diff = None
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
            bl_absmax[(b, l)].append(am)
            s = am / QMAX if am > 1e-12 else 1.0
            diff = (np.clip(np.round(curr / s), -QMAX, QMAX)
                    - np.clip(np.round(prev / s), -QMAX, QMAX)).astype(np.int32)
            bl_zero[(b, l)][0] += int((diff == 0).sum())
            bl_zero[(b, l)][1] += diff.size
            if sample_diff is None and b == 9 and l == "attn1-to-q":
                sample_diff = diff.ravel()

    print("=== per-(block,layer) zero rate (over all classes/steps) ===")
    print(f"{'block':>6} {'layer':>16} {'zero%':>7} {'mean absmax':>12}")
    for (b, l) in sorted(bl_zero):
        z, t = bl_zero[(b, l)]
        am = np.mean(bl_absmax[(b, l)])
        flag = "  <-- absmax~0?" if am < 1e-3 else ""
        print(f"{b:>6} {l:>16} {z/t*100:>6.1f}% {am:>12.4f}{flag}")

    # overall
    Z = sum(v[0] for v in bl_zero.values())
    T = sum(v[1] for v in bl_zero.values())
    print(f"\noverall zero rate: {Z/T*100:.1f}%")

    if sample_diff is not None:
        print(f"\n=== sample diff histogram (block9 attn1-to-q, one pair) ===")
        vals, counts = np.unique(sample_diff, return_counts=True)
        order = np.argsort(-counts)[:9]
        for i in order:
            print(f"  diff={int(vals[i]):>4}: {counts[i]/sample_diff.size*100:>5.1f}%")
        print(f"  (zero here: {(sample_diff==0).mean()*100:.1f}%)")

    print("\nInterpretation:")
    print("- uniform 70-85% across layers + nonzero absmax -> 78.8% is REAL (DiT is")
    print("  genuinely more temporally sparse than SDM).")
    print("- a few layers ~100% with absmax~0 -> those are degenerate, pulling up the")
    print("  mean -> exclude them and recompute.")


if __name__ == "__main__":
    main()

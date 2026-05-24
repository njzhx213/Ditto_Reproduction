#!/usr/bin/env python3
"""
probe_trace.py - quick reconnaissance before writing fig17_defo_accuracy.py.

Tells us two things:
  1) the 6 trace layer names + their input/output shapes (so fig17 can compute
     per-layer MACs correctly: conv vs linear/attention).
  2) whether a simple symmetric 4-bit quantizer on the temporal difference
     delta = in_t - in_{t-1} gives a sane zero-ratio (~0.4-0.5, cf Fig 5's 44%).

Reads only a couple of images/steps -- runs in seconds.

    cd ~/Ditto && python3 sim/probe_trace.py
"""
import glob
import numpy as np


def quant_zero_ratio(delta, n_bits=4):
    """Symmetric per-tensor 4-bit quant; fraction of values rounding to 0."""
    a = delta.astype(np.float32).ravel()
    amax = np.abs(a).max()
    if amax == 0:
        return 1.0
    qmax = (1 << (n_bits - 1)) - 1          # 4-bit signed -> 7
    scale = amax / qmax
    q = np.round(a / scale)
    return float(np.mean(q == 0))


def main():
    # discover the 6 layer names from image_000
    files = sorted(glob.glob("traces/sdm/image_000/step_000/*.npz"))
    layer_names = [f.split("/")[-1].replace(".npz", "") for f in files]
    print(f"Layers per (image,step): {len(layer_names)}")
    print(f"Names: {layer_names}\n")

    print(f"{'layer':>14} {'input shape':>22} {'output shape':>22} {'rank':>5}")
    print("-" * 68)
    for name in layer_names:
        d = np.load(f"traces/sdm/image_000/step_000/{name}.npz")
        ish, osh = d["input"].shape, d["output"].shape
        print(f"{name:>14} {str(ish):>22} {str(osh):>22} {len(ish):>5}")

    # sanity: zero-ratio of delta across the first few sampling steps, conv_in + one more
    print("\n4-bit delta zero-ratio sanity (image_000, consecutive steps):")
    print(f"{'layer':>14} {'step':>6} {'zero_ratio':>11}")
    print("-" * 34)
    for name in [layer_names[0], layer_names[-1]]:
        prev = None
        for s in range(4):
            d = np.load(f"traces/sdm/image_000/step_{s:03d}/{name}.npz")
            cur = d["input"]
            if prev is not None:
                zr = quant_zero_ratio(cur - prev)
                print(f"{name:>14} {s:>6} {zr:>10.3f}")
            prev = cur


if __name__ == "__main__":
    main()

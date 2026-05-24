#!/usr/bin/env python3
"""
dit_mac_weighted_sparsity.py - MAC-weighted DiT temporal-difference sparsity.

The per-layer measured zero rates (verify_dit_bitwidth.py) vary 37-96% by depth and
layer type, so an EQUAL-weight mean (78.8%) is misleading. bit-width affects speedup
through how many MACs are skipped, so the meaningful scalar is the MAC-WEIGHTED zero
rate. We only traced 4 representative blocks (0/9/18/27); we extrapolate them to all
28 blocks by depth segment, then weight each layer by its true MAC count (from the
model). This is reported ALONGSIDE the full per-layer distribution, not instead of it.

Extrapolation (stated assumption):
  block 0-4   (shallow) -> block 0 measured rates
  block 5-22  (middle)  -> mean of block 9 & 18 measured rates
  block 23-27 (deep)    -> block 27 measured rates
Within a block, each layer type uses its own measured rate. Middle blocks are 18/28
of the model and have LOW attention sparsity, so the MAC-weighted rate should fall
well below the equal-weight 78.8%.

    cd ~/Ditto && python3 sim/dit_mac_weighted_sparsity.py
"""
import torch
import torch.nn as nn
from diffusers import DiTTransformer2DModel

# measured zero rates from verify_dit_bitwidth.py (input activations)
MEASURED = {
    0:  {"attn1.to_q": 0.872, "ff.net.0.proj": 0.839, "ff.net.2": 0.962},
    9:  {"attn1.to_q": 0.424, "ff.net.0.proj": 0.480, "ff.net.2": 0.798},
    18: {"attn1.to_q": 0.371, "ff.net.0.proj": 0.547, "ff.net.2": 0.793},
    27: {"attn1.to_q": 0.741, "ff.net.0.proj": 0.832, "ff.net.2": 0.897},
}


def block_rates(bidx):
    """Extrapolate measured representative blocks to block bidx by depth segment."""
    if bidx <= 4:
        return MEASURED[0]
    if bidx >= 23:
        return MEASURED[27]
    # middle: average block 9 and 18
    return {k: (MEASURED[9][k] + MEASURED[18][k]) / 2 for k in MEASURED[9]}


def layer_zero_rate(bidx, name):
    """Pick the measured rate for a layer by matching its type; attention QK/PV and
    the to_k/to_v/to_out projections reuse the to_q rate (same attention block);
    out-proj / unmatched linears default to the block's ff.net.0 rate."""
    r = block_rates(bidx)
    if "attn" in name and ("to_q" in name or "to_k" in name or "to_v" in name or "to_out" in name):
        return r["attn1.to_q"]
    if "ff.net.0" in name or "ff.net.2" in name:
        return r["ff.net.2"] if "ff.net.2" in name else r["ff.net.0.proj"]
    return r["ff.net.0.proj"]   # other proj-like linears


def main():
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

    # MAC-weighted zero rate over all linears, mapping name -> block idx
    import re
    num = den = 0.0
    seg = {"shallow": [0, 0.0], "middle": [0, 0.0], "deep": [0, 0.0]}
    for n, (in_s, out_s) in shapes.items():
        mb = re.search(r"transformer_blocks\.(\d+)\.", n)
        if not mb:
            continue
        bidx = int(mb.group(1))
        M = 1
        for d in in_s[:-1]:
            M *= d
        macs = M * in_s[-1] * out_s[-1]
        z = layer_zero_rate(bidx, n)
        num += z * macs
        den += macs
        s = "shallow" if bidx <= 4 else ("deep" if bidx >= 23 else "middle")
        seg[s][0] += 1
        seg[s][1] += z * macs

    print("=== DiT MAC-weighted temporal sparsity (linears) ===")
    print(f"MAC-weighted zero rate : {num/den*100:.1f}%")
    print(f"(equal-weight mean was : 78.8%  -- inflated by shallow/MLP-output layers)")
    print(f"\nby depth segment (MAC-weighted contribution):")
    for s, (cnt, w) in seg.items():
        print(f"  {s:>8}: {w/den*100:5.1f}% of total weighted-zero mass")
    print("\nNote: extrapolated from 4 traced blocks (0/9/18/27) to 28 by depth")
    print("segment; middle blocks (18/28) dominate and have low attention sparsity,")
    print("so the MAC-weighted rate sits below the equal-weight 78.8%.")
    print("Reported alongside the per-layer distribution, not as a single headline.")


if __name__ == "__main__":
    main()

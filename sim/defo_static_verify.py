#!/usr/bin/env python3
"""
defo_static_verify.py - is the 1.04x "tiny benefit" real, or an artifact?

Two independent checks on the SD UNet:

(B) REAL EXECUTION ORDER. defo_static.py used named_modules() DFS order as a
    proxy for forward order, and counted ALL norms as hard difference boundaries.
    Here we hook every leaf and record the ACTUAL forward call order, then re-cut
    the difference segments. If segment length stays ~2, the dense-norm story is
    real. If it grows, the 2.08 was a traversal artifact.

(A) BETTER METRIC. defo_static counted prev-READ COUNT (2 per segment), which
    barely changed (1.04x). But Defo's real saving is that INTERMEDIATE
    ACTIVATIONS inside a difference segment need not be materialized to DRAM:
    difference flows as delta through consecutive linear ops. So the right metric
    is BYTES of intermediate activation that diff-mode avoids writing+reading.
      old model:  every linear layer's input+output activation touches DRAM
      diff/Defo:  only segment-boundary activations touch DRAM; interior ones stay
                  on-chip as delta
    We compute both in bytes and report the reduction.

Honesty: boundaries = nonlinear leaf ops (GroupNorm/LayerNorm/SiLU/GEGLU). We do
NOT model residual-add bypass (a known simplification). Both checks use the same
boundary definition so they are comparable; the question here is whether the
SEGMENT STRUCTURE and the METRIC change the conclusion, not the boundary rule.

    cd ~/Ditto && python3 sim/defo_static_verify.py
"""
from __future__ import annotations

import torch
import torch.nn as nn
from diffusers import UNet2DConditionModel

LINEAR_TYPES = (nn.Conv2d, nn.Linear)
NONLINEAR_TYPES = (nn.GroupNorm, nn.LayerNorm, nn.SiLU, nn.GELU, nn.Softmax)
NONLINEAR_HINTS = ("geglu", "gelu", "silu", "softmax", "swish")


def classify(mod):
    tn = type(mod).__name__.lower()
    if isinstance(mod, NONLINEAR_TYPES) or any(h in tn for h in NONLINEAR_HINTS):
        return "nonlinear"
    if isinstance(mod, LINEAR_TYPES):
        return "linear"
    return "skip"


def segment(seq):
    """seq: list of (name, cls, in_bytes, out_bytes). Returns segments (lists)."""
    segs, cur, n_nl = [], [], 0
    for item in seq:
        cls = item[1]
        if cls == "nonlinear":
            n_nl += 1
            if cur:
                segs.append(cur)
                cur = []
        elif cls == "linear":
            cur.append(item)
    if cur:
        segs.append(cur)
    return segs, n_nl


def main():
    print("Loading SDM v1.4 UNet ...", flush=True)
    unet = UNet2DConditionModel.from_pretrained(
        "CompVis/stable-diffusion-v1-4", subfolder="unet").eval()

    # ---- hook every relevant leaf to capture REAL forward order + activation sizes
    call_seq = []  # (name, cls, in_bytes, out_bytes) in actual execution order

    def make_hook(name, mod):
        cls = classify(mod)

        def hook(m, inp, out):
            if cls == "skip":
                return
            x = inp[0] if isinstance(inp, (tuple, list)) and len(inp) else None
            y = out[0] if isinstance(out, (tuple, list)) else out
            ib = x.numel() if hasattr(x, "numel") else 0
            ob = y.numel() if hasattr(y, "numel") else 0
            call_seq.append((name, cls, int(ib), int(ob)))
        return hook

    handles = []
    for name, mod in unet.named_modules():
        if classify(mod) in ("linear", "nonlinear"):
            handles.append(mod.register_forward_hook(make_hook(name, mod)))

    with torch.no_grad():
        unet(torch.randn(1, 4, 64, 64), torch.tensor(1), torch.randn(1, 77, 768))
    for h in handles:
        h.remove()

    # ============ Check B: real-order segmentation ============
    segs, n_nl = segment(call_seq)
    seg_lens = [len(s) for s in segs]
    n_linear = sum(seg_lens)
    print("\n=== Check B: segments under REAL forward order ===")
    print(f"Hooked leaf calls (linear+nonlinear): {len(call_seq)}")
    print(f"Nonlinear boundaries: {n_nl}   Linear ops: {n_linear}")
    print(f"Segments: {len(segs)}   length min={min(seg_lens)} max={max(seg_lens)} "
          f"mean={n_linear/len(segs):.2f}")
    print(f"  (named_modules-order earlier gave: 128 segments, mean 2.08)")
    if n_linear / len(segs) > 2.5:
        print("  -> segments are LONGER under real order: the 2.08 was partly a")
        print("     traversal artifact.")
    else:
        print("  -> segment length similar: dense-norm structure is REAL, not an artifact.")

    # ============ Check A: activation-materialization bytes ============
    # OLD: every linear op materializes in+out activation to DRAM
    old_bytes = sum(ib + ob for (_, cls, ib, ob) in call_seq if cls == "linear")
    # DIFF/Defo: only segment-boundary activations touch DRAM. Interior linear
    # ops keep their intermediate as on-chip delta. Per segment we materialize:
    #   the segment's first input  (sum-in)  + the segment's last output (re-diff)
    diff_bytes = 0
    for s in segs:
        first_in = s[0][2]
        last_out = s[-1][3]
        diff_bytes += first_in + last_out
    print("\n=== Check A: intermediate-activation DRAM bytes ===")
    print(f"OLD  (every linear op in+out -> DRAM): {old_bytes/1e6:.1f} M elements")
    print(f"DEFO (only segment boundaries -> DRAM): {diff_bytes/1e6:.1f} M elements")
    red = old_bytes / diff_bytes if diff_bytes else float("nan")
    print(f"Reduction in intermediate-activation traffic: {red:.2f}x")
    print(f"  (prev-READ-COUNT metric earlier gave only 1.04x; this BYTE metric")
    print(f"   measures what diff-mode actually avoids materializing.)")

    # weight bytes are unaffected (same in both), report context
    print(f"\nNote: this is intermediate-ACTIVATION traffic only; weights are loaded")
    print(f"identically in both modes, so total-memory reduction is diluted by")
    print(f"weight bytes (the weight-dominated regime we found earlier).")


if __name__ == "__main__":
    main()

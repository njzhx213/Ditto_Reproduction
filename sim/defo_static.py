#!/usr/bin/env python3
"""
defo_static.py - Defo's STATIC-GRAPH analysis half (the part not yet built).

Background
----------
Defo has two halves:
  (1) runtime per-layer min(Cycle_act, Cycle_diff)   -- already implemented.
  (2) static-graph analysis (THIS FILE): decide WHERE difference computation can
      propagate, by locating NONLINEAR boundaries.

Why nonlinear boundaries matter
-------------------------------
For a LINEAR op f:   f(h_t) - f(h_{t-1}) = f(h_t - h_{t-1}) = f(delta).
  -> difference propagates THROUGH linear layers; no need to reconstruct the full
     activation between them.
For a NONLINEAR op g:  g(h_t) - g(h_{t-1}) != g(delta).
  -> difference CANNOT cross it. We must sum (delta + prev -> h_t) BEFORE the
     nonlinearity, and re-difference AFTER it.

So the graph splits into "difference segments": maximal runs of linear ops
between two nonlinear boundaries. Inside a segment, delta flows freely and prev
tensors are read ONLY at the segment boundaries (to sum / re-difference), NOT at
every layer. This is the second memory-saving mechanism of Defo.

Why structure-based (not torch.fx)
----------------------------------
torch.fx.symbolic_trace fails on this UNet (TraceError: Proxy cannot be iterated
-- control flow in attention/timestep). BUT all 149 nonlinear ops are explicit
nn.Modules (GroupNorm x61, LayerNorm x48, SiLU x24, GEGLU x16; verified by
probe_fx.py), and diffusers blocks have FIXED internal structure. So we analyze
by walking the module tree in execution order and classifying each leaf as
linear / nonlinear / passthrough. This is reliable for boundary location even
without the fx edge list.

Honesty boundaries
------------------
  - "Execution order" is approximated by named_modules() depth-first order, which
    matches forward order for these sequential blocks; cross-block residual adds
    are not edges we model (they don't change nonlinear-boundary counting).
  - The corrected prev traffic assumes prev is read once at each segment boundary
    (2 reads per segment: sum-in + re-diff-out). This is a mechanism-based
    inference; the paper gives no per-segment access counts.
  - Compares against the OLD assumption (prev read at EVERY linear layer) to show
    how much the static analysis reduces prev traffic.

    cd ~/Ditto && python3 sim/defo_static.py
"""
from __future__ import annotations

import torch.nn as nn

# leaf module classification
LINEAR_TYPES = (nn.Conv2d, nn.Linear)
NONLINEAR_TYPES = (nn.GroupNorm, nn.LayerNorm, nn.SiLU, nn.GELU, nn.Softmax)
NONLINEAR_NAME_HINTS = ("geglu", "gelu", "silu", "softmax", "swish")
PASSTHROUGH_TYPES = (nn.Dropout, nn.Identity)


def classify_leaf(mod):
    """Return 'linear' | 'nonlinear' | 'passthrough' | 'other' for a leaf module."""
    tn = type(mod).__name__.lower()
    if isinstance(mod, NONLINEAR_TYPES) or any(h in tn for h in NONLINEAR_NAME_HINTS):
        return "nonlinear"
    if isinstance(mod, LINEAR_TYPES):
        return "linear"
    if isinstance(mod, PASSTHROUGH_TYPES):
        return "passthrough"
    return "other"


def is_leaf(mod):
    return len(list(mod.children())) == 0 or type(mod).__name__ == "GEGLU"
    # GEGLU has children (a Linear) but acts as a nonlinear unit; treat as leaf.


def walk_execution_order(model):
    """Depth-first leaf sequence approximating forward execution order.
    Returns list of (qualified_name, classification)."""
    seq = []

    def recurse(mod, prefix):
        # GEGLU: treat as a single nonlinear leaf (its gate is the nonlinearity)
        if type(mod).__name__ == "GEGLU":
            seq.append((prefix or type(mod).__name__, "nonlinear"))
            return
        children = list(mod.named_children())
        if not children:
            seq.append((prefix, classify_leaf(mod)))
            return
        for name, child in children:
            recurse(child, f"{prefix}.{name}" if prefix else name)

    recurse(model, "")
    return seq


def segment_by_nonlinear(seq):
    """Split the linear-op sequence into difference segments at nonlinear ops.
    Returns (segments, n_nonlinear). Each segment = list of linear op names."""
    segments = []
    cur = []
    n_nl = 0
    for name, cls in seq:
        if cls == "nonlinear":
            n_nl += 1
            if cur:
                segments.append(cur)
                cur = []
            # nonlinear op itself is a boundary; not part of any linear segment
        elif cls == "linear":
            cur.append(name)
        # passthrough / other: ignored (don't break a segment)
    if cur:
        segments.append(cur)
    return segments, n_nl


def prev_traffic_comparison(seq):
    """Compare prev-tensor reads under:
      OLD model: prev read at EVERY linear layer  -> reads = n_linear_layers
      Defo static: prev read only at segment boundaries -> reads = 2 * n_segments
    (2 = sum-in at segment start + re-difference-out at segment end)
    Returns dict of counts and the reduction factor."""
    segments, n_nl = segment_by_nonlinear(seq)
    n_linear = sum(len(s) for s in segments)
    old_reads = n_linear                 # prev at every linear layer
    defo_reads = 2 * len(segments)       # prev only at segment boundaries
    seg_lens = [len(s) for s in segments]
    return {
        "n_linear_layers": n_linear,
        "n_nonlinear": n_nl,
        "n_segments": len(segments),
        "seg_len_min": min(seg_lens) if seg_lens else 0,
        "seg_len_max": max(seg_lens) if seg_lens else 0,
        "seg_len_mean": sum(seg_lens) / len(seg_lens) if seg_lens else 0,
        "old_prev_reads": old_reads,
        "defo_prev_reads": defo_reads,
        "reduction": old_reads / defo_reads if defo_reads else float("nan"),
    }


def main():
    from diffusers import UNet2DConditionModel
    print("Loading SDM v1.4 UNet ...", flush=True)
    unet = UNet2DConditionModel.from_pretrained(
        "CompVis/stable-diffusion-v1-4", subfolder="unet").eval()

    seq = walk_execution_order(unet)
    cls_count = {}
    for _, c in seq:
        cls_count[c] = cls_count.get(c, 0) + 1

    print("\n=== Defo static-graph analysis (structure-based) ===")
    print(f"Leaf ops in execution order: {len(seq)}")
    print(f"  classification: {cls_count}")

    stats = prev_traffic_comparison(seq)
    print(f"\nNonlinear boundaries (sum/re-diff points): {stats['n_nonlinear']}")
    print(f"Linear ops (Conv+Linear):                  {stats['n_linear_layers']}")
    print(f"Difference segments:                       {stats['n_segments']}")
    print(f"  segment length (linear ops): min={stats['seg_len_min']} "
          f"max={stats['seg_len_max']} mean={stats['seg_len_mean']:.2f}")

    print(f"\n--- prev-tensor traffic: old model vs Defo static analysis ---")
    print(f"  OLD  (prev read at every linear layer): {stats['old_prev_reads']} reads")
    print(f"  DEFO (prev read at segment boundaries): {stats['defo_prev_reads']} reads")
    print(f"  reduction factor: {stats['reduction']:.2f}x fewer prev reads")
    print(f"\nImplication: our earlier memory model charged prev_in+prev_out at EVERY")
    print(f"layer. Defo's static analysis shows prev is needed only at the {stats['n_nonlinear']}")
    print(f"nonlinear boundaries, cutting prev traffic ~{stats['reduction']:.1f}x. This")
    print(f"lowers the diff-mode memory overhead below the per-layer assumption.")
    print(f"\n(Structure-based: torch.fx unavailable on this UNet; all 149 nonlinear")
    print(f"ops are explicit modules so boundary location is exact. Per-segment prev")
    print(f"count = 2/segment is a mechanism-based inference, not paper-stated.)")


if __name__ == "__main__":
    main()

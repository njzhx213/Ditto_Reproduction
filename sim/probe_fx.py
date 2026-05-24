#!/usr/bin/env python3
"""
probe_fx.py - can we get a real dataflow graph of the SD UNet for Defo's
static-graph analysis?

Defo's static half needs the GRAPH (who feeds whom) so it can locate nonlinear
boundaries (SiLU/GroupNorm/Softmax/LayerNorm/GEGLU) and mark the linear segments
between them where difference computation can propagate. named_modules() only
gives a flat module LIST (no edges), so we try torch.fx.symbolic_trace to get the
edges.

This probe answers ONE question: does symbolic_trace work on this UNet?
  - if yes  -> static analysis runs on the real fx graph (faithful).
  - if no   -> we see WHERE it breaks, and fall back to a structure-based
               approximation (documented as approximate).

It also inventories nonlinear ops by module type as a fallback, so the run is
useful either way.

    cd ~/Ditto && python3 sim/probe_fx.py
"""
import torch
import torch.nn as nn
from diffusers import UNet2DConditionModel

NONLINEAR_TYPES = (nn.SiLU, nn.GELU, nn.GroupNorm, nn.LayerNorm, nn.Softmax)
NONLINEAR_NAME_HINTS = ("silu", "gelu", "geglu", "groupnorm", "layernorm",
                        "softmax", "swish", "act_fn", "nonlinearity")


def main():
    print("Loading SDM v1.4 UNet ...", flush=True)
    unet = UNet2DConditionModel.from_pretrained(
        "CompVis/stable-diffusion-v1-4", subfolder="unet").eval()

    # ---- 1. inventory module types (fallback signal, always works) ----
    from collections import Counter
    type_counts = Counter()
    nonlinear_by_type = Counter()
    for name, m in unet.named_modules():
        tn = type(m).__name__
        type_counts[tn] += 1
        if isinstance(m, NONLINEAR_TYPES) or any(h in tn.lower() for h in NONLINEAR_NAME_HINTS):
            nonlinear_by_type[tn] += 1

    print("\n--- module type inventory (top 20) ---")
    for tn, c in type_counts.most_common(20):
        mark = "  <- nonlinear" if tn in nonlinear_by_type else ""
        print(f"  {c:>4}  {tn}{mark}")

    print("\n--- candidate NONLINEAR module types ---")
    if nonlinear_by_type:
        for tn, c in nonlinear_by_type.most_common():
            print(f"  {c:>4}  {tn}")
    else:
        print("  (none matched by type/name -- activations may be functional calls)")

    # ---- 2. try torch.fx symbolic_trace ----
    print("\n--- attempting torch.fx.symbolic_trace(unet) ---")
    try:
        import torch.fx as fx
        gm = fx.symbolic_trace(unet)
        nodes = list(gm.graph.nodes)
        print(f"  SUCCESS: traced graph has {len(nodes)} nodes.")
        # count node ops
        op_counts = Counter(n.op for n in nodes)
        print(f"  node ops: {dict(op_counts)}")
        # count functional nonlinear calls (call_function / call_method)
        func_nl = [n for n in nodes if n.op in ("call_function", "call_method")
                   and any(h in str(n.target).lower() for h in NONLINEAR_NAME_HINTS
                           + ("relu", "sigmoid", "tanh"))]
        print(f"  functional nonlinear calls in graph: {len(func_nl)}")
        print("  => static analysis CAN use the real fx graph.")
    except Exception as e:
        print(f"  FAILED: {type(e).__name__}")
        msg = str(e)
        print(f"  reason (first 300 chars): {msg[:300]}")
        print("  => fx tracing not viable as-is; static analysis will use a")
        print("     structure-based approximation over named_modules ordering.")


if __name__ == "__main__":
    main()

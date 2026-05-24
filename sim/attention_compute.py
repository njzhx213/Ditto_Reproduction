#!/usr/bin/env python3
"""
attention_compute.py - bring attention QK/PV into the compute model.

Our Linear/Conv enumeration missed the attention matmuls Q*K and P*V (probe found
they are 63 G = 15.7% of the true total; 61 G is self-attention). This adds them
and models Ditto's attention difference cost per the paper:

  ITC baseline (per QK or PV): one full 8-bit GEMM.
  Ditto self-attention:  TWO sub-operations (Q_t.dK + dQ.K_{t+1}), each with
                         zero-skip + 4-bit. cost = 2 * nonzero * bitf / (PE*lanes).
  Ditto cross-attention: K,V constant across steps -> ONE sub-operation
                         (dQ.K), like a plain linear difference.

We recompute total MACs and the bare compute speedup ceiling with attention
included, and compare to the Linear/Conv-only 10.40x.

Honesty: attention MEMORY is approximated as a generic GEMM here (we do NOT yet
model Q_t acting as a re-read 'weight'); this file changes the COMPUTE side only,
which is where the missed 63 G matters most. Memory-side attention is future work.

    cd ~/Ditto && python3 sim/attention_compute.py
"""
from __future__ import annotations

import torch
import torch.nn as nn
from diffusers import UNet2DConditionModel

N_PE_ITC = 27648
N_PE_DITTO = 39398
LANES = 4
BW_ZERO, BW_LE4, BW_GT4 = 0.459, 0.534, 0.007
NONZERO = BW_LE4 + BW_GT4
BITF = (BW_LE4 + 2 * BW_GT4) / NONZERO


def comp_itc(macs):
    return macs / N_PE_ITC


def comp_diff_linear(macs):
    """Plain linear / conv / cross-attn: one difference sub-op."""
    return macs * NONZERO * BITF / (N_PE_DITTO * LANES)


def comp_diff_attn_self(macs):
    """Self-attention QK/PV: TWO sub-ops, each zero-skip + 4-bit."""
    return 2.0 * macs * NONZERO * BITF / (N_PE_DITTO * LANES)


def enumerate_all():
    """Linear+Conv layers (as before) PLUS attention QK/PV matmuls."""
    unet = UNet2DConditionModel.from_pretrained(
        "CompVis/stable-diffusion-v1-4", subfolder="unet").eval()
    shapes = {}
    attn_recs = []

    def lin_hook(nm):
        def h(m, inp, out):
            y = out[0] if isinstance(out, (tuple, list)) else out
            shapes[nm] = (tuple(inp[0].shape), tuple(y.shape), type(m).__name__)
        return h

    def attn_hook(nm, mod):
        def h(m, args, kwargs, out):
            hs = args[0] if args else kwargs.get("hidden_states")
            ctx = args[1] if (len(args) > 1 and torch.is_tensor(args[1])) else None
            if kwargs.get("encoder_hidden_states") is not None:
                ctx = kwargs["encoder_hidden_states"]
            is_cross = ctx is not None
            sq = hs.shape[1]
            sk = ctx.shape[1] if is_cross else hs.shape[1]
            heads = getattr(m, "heads", 8)
            dim = m.to_q.out_features
            hd = max(1, dim // heads)
            B = hs.shape[0]
            qk = B * heads * sq * sk * hd     # MACs of Q*K
            pv = B * heads * sq * sk * hd     # MACs of P*V
            attn_recs.append((nm, is_cross, qk, pv))
        return h

    hs_handles = [m.register_forward_hook(lin_hook(n))
                  for n, m in unet.named_modules()
                  if isinstance(m, (nn.Conv2d, nn.Linear))]
    at_handles = [m.register_forward_hook(attn_hook(n, m), with_kwargs=True)
                  for n, m in unet.named_modules()
                  if type(m).__name__ == "Attention"]
    with torch.no_grad():
        unet(torch.randn(1, 4, 64, 64), torch.tensor(1), torch.randn(1, 77, 768))
    for h in hs_handles + at_handles:
        h.remove()

    mods = dict(unet.named_modules())
    layers = []
    for n, (in_s, out_s, tn) in shapes.items():
        mod = mods[n]
        if isinstance(mod, nn.Conv2d):
            M = in_s[0] * out_s[2] * out_s[3]
            K = (in_s[1] // mod.groups) * mod.kernel_size[0] * mod.kernel_size[1]
            N = out_s[1]
        else:
            M = 1
            for d in in_s[:-1]:
                M *= d
            K, N = mod.in_features, mod.out_features
        layers.append({"cat": "linear", "macs": float(M * K * N)})

    for nm, is_cross, qk, pv in attn_recs:
        cat = "attn_cross" if is_cross else "attn_self"
        layers.append({"cat": cat, "macs": float(qk)})
        layers.append({"cat": cat, "macs": float(pv)})
    return layers


def speedup(layers):
    itc = sum(comp_itc(L["macs"]) for L in layers)
    ditto = 0.0
    for L in layers:
        if L["cat"] == "attn_self":
            ditto += comp_diff_attn_self(L["macs"])
        else:                                   # linear, conv, attn_cross
            ditto += comp_diff_linear(L["macs"])
    return itc / ditto if ditto else 0.0


def main():
    layers = enumerate_all()
    lin = [L for L in layers if L["cat"] == "linear"]
    aself = [L for L in layers if L["cat"] == "attn_self"]
    across = [L for L in layers if L["cat"] == "attn_cross"]
    g = lambda xs: sum(L["macs"] for L in xs) / 1e9

    print("=== Attention-aware compute model ===\n")
    print(f"Linear+Conv MACs:     {g(lin):.1f} G  ({len(lin)} layers)")
    print(f"Self-attn QK/PV MACs: {g(aself):.1f} G  ({len(aself)} matmuls)")
    print(f"Cross-attn QK/PV MACs:{g(across):.1f} G  ({len(across)} matmuls)")
    total = g(layers)
    print(f"TOTAL MACs:           {total:.1f} G   (was 338.6 G linear-only, "
          f"+{100*(total-338.6)/338.6:.1f}%)\n")

    su_all = speedup(layers)
    su_lin = speedup(lin)
    print(f"Bare compute speedup, linear/conv only: {su_lin:.2f}x")
    print(f"Bare compute speedup, WITH attention:   {su_all:.2f}x")
    print(f"\nThe ceiling shifts because self-attention pays 2 sub-ops (not 1),")
    print(f"so its Ditto speedup is ~half a normal layer's; with self-attn at "
          f"{g(aself):.0f}G of {total:.0f}G, it pulls the overall ceiling "
          f"{'down' if su_all < su_lin else 'up'} from {su_lin:.2f}x to {su_all:.2f}x.")
    print("\n(Compute side only; attention memory approximated as generic GEMM,")
    print(" flagged as future work.)")


if __name__ == "__main__":
    main()

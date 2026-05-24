#!/usr/bin/env python3
"""
probe_attention.py - how much MAC did our Linear-only enumeration miss?

Our enumerate_layers() hooks nn.Conv2d and nn.Linear. But attention's core
matmuls Q*K and P*V are torch operations INSIDE the Attention module, NOT
nn.Linear -- so they were never counted. The paper treats these specially
(two-matrix difference, >2x potential speedup). Before deciding how deeply to
model attention difference, we must know how much MAC these matmuls represent in
SDM (conv-dominated; QK/PV may be small) and split self- vs cross-attention.

For each diffusers Attention module we hook it and read the Q/K/V shapes from a
forward pass, then compute:
  QK MACs  = B * heads * seq_q * seq_k * head_dim
  PV MACs  = B * heads * seq_q * seq_k * head_dim
self vs cross is detected by whether encoder_hidden_states (context) was passed.

    cd ~/Ditto && python3 sim/probe_attention.py
"""
import torch
import torch.nn as nn
from diffusers import UNet2DConditionModel


def main():
    print("Loading SDM v1.4 UNet ...", flush=True)
    unet = UNet2DConditionModel.from_pretrained(
        "CompVis/stable-diffusion-v1-4", subfolder="unet").eval()

    records = []  # (name, is_cross, seq_q, seq_k, dim, heads, qk_macs, pv_macs)

    def make_hook(name, mod):
        def hook(m, args, kwargs, out):
            # diffusers Attention.forward(hidden_states, encoder_hidden_states=None, ...)
            hs = args[0] if args else kwargs.get("hidden_states")
            ctx = None
            if len(args) > 1 and torch.is_tensor(args[1]):
                ctx = args[1]
            if kwargs.get("encoder_hidden_states") is not None:
                ctx = kwargs.get("encoder_hidden_states")
            is_cross = ctx is not None
            seq_q = hs.shape[1]
            seq_k = ctx.shape[1] if is_cross else hs.shape[1]
            heads = getattr(m, "heads", 8)
            dim = m.to_q.out_features
            head_dim = max(1, dim // heads)
            B = hs.shape[0]
            qk = B * heads * seq_q * seq_k * head_dim
            pv = B * heads * seq_q * seq_k * head_dim
            records.append((name, is_cross, seq_q, seq_k, dim, heads, qk, pv))
        return hook

    handles = []
    for name, mod in unet.named_modules():
        if type(mod).__name__ == "Attention":
            handles.append(mod.register_forward_hook(make_hook(name, mod),
                                                     with_kwargs=True))
    with torch.no_grad():
        unet(torch.randn(1, 4, 64, 64), torch.tensor(1), torch.randn(1, 77, 768))
    for h in handles:
        h.remove()

    # totals
    self_qkpv = sum(r[6] + r[7] for r in records if not r[1])
    cross_qkpv = sum(r[6] + r[7] for r in records if r[1])
    total_attn = self_qkpv + cross_qkpv
    n_self = sum(1 for r in records if not r[1])
    n_cross = sum(1 for r in records if r[1])

    print(f"\n=== Attention QK/PV matmuls (missed by Linear-only hooks) ===")
    print(f"Attention modules: {len(records)}  (self={n_self}, cross={n_cross})")
    print(f"\n{'module':>40} {'type':>6} {'seq_q':>6} {'seq_k':>6} {'heads':>6} {'QK+PV(M)':>10}")
    print("-" * 86)
    for name, is_cross, sq, sk, dim, h, qk, pv in records[:12]:
        print(f"{name[-40:]:>40} {'cross' if is_cross else 'self':>6} "
              f"{sq:>6} {sk:>6} {h:>6} {(qk+pv)/1e6:>10.1f}")
    if len(records) > 12:
        print(f"  ... and {len(records)-12} more")

    # compare to the Linear+Conv total we already count (338.6 G)
    LINEAR_CONV_GMAC = 338.6
    print(f"\n--- MAC accounting ---")
    print(f"Self-attention QK+PV:  {self_qkpv/1e9:.2f} G")
    print(f"Cross-attention QK+PV: {cross_qkpv/1e9:.2f} G")
    print(f"Total attention matmul: {total_attn/1e9:.2f} G  (MISSED so far)")
    print(f"Linear+Conv (counted):  {LINEAR_CONV_GMAC:.1f} G")
    grand = total_attn/1e9 + LINEAR_CONV_GMAC
    print(f"Grand total:            {grand:.1f} G")
    print(f"\nAttention matmul share of true total: "
          f"{100*total_attn/1e9/grand:.1f}%")
    print(f"Our previous 338.6 G UNDERCOUNTED the true total by "
          f"{100*(total_attn/1e9)/LINEAR_CONV_GMAC:.1f}%")
    print("\nDecision rule: if attention share > ~15%, must model it precisely and")
    print("re-run affected results; if < ~5%, implement the difference mechanism")
    print("functionally (correctness) + a documented approximation in the perf model.")


if __name__ == "__main__":
    main()

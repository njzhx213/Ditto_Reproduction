#!/usr/bin/env python3
"""
mem_decompose.py - is weight (B) memory over-counted, and is it worse for DiT?

Our mem_lines counts (A_input + B_weight + C_output) elements per layer, every
layer, as if all three are re-streamed each time. But WEIGHTS are read-only and
reused across all tokens and all diffusion steps; if resident they cost ~one load,
not a full per-layer stream. Activations (A, C) are genuinely per-token/per-step.

If B dominates memory and is bigger (relatively) in DiT than SDM, then counting
full B every layer inflates DiT's memory -> the core/memory mismatch vs paper.
This script measures B's share in each workload to decide if an amortized-weight
fix is warranted (a GENERAL correction, not a DiT-only tune).

    cd ~/Ditto && python3 sim/mem_decompose.py
"""
import torch
import torch.nn as nn
from diffusers import UNet2DConditionModel, DiTTransformer2DModel


def enum_sdm():
    unet = UNet2DConditionModel.from_pretrained(
        "CompVis/stable-diffusion-v1-4", subfolder="unet").eval()
    shapes = {}

    def hook(nm):
        def h(m, inp, out):
            y = out[0] if isinstance(out, (tuple, list)) else out
            shapes[nm] = (tuple(inp[0].shape), tuple(y.shape))
        return h
    hs = [m.register_forward_hook(hook(n)) for n, m in unet.named_modules()
          if isinstance(m, (nn.Conv2d, nn.Linear))]
    with torch.no_grad():
        unet(torch.randn(1, 4, 64, 64), torch.tensor(1), torch.randn(1, 77, 768))
    for h in hs:
        h.remove()
    mods = dict(unet.named_modules())
    A = B = C = 0.0
    for n, (in_s, out_s) in shapes.items():
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
        A += M * K
        B += K * N
        C += M * N
    return A, B, C


def enum_dit():
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
    A = B = C = 0.0
    for n, (in_s, out_s) in shapes.items():
        M = 1
        for d in in_s[:-1]:
            M *= d
        K, N = in_s[-1], out_s[-1]
        A += M * K
        B += K * N
        C += M * N
    return A, B, C


def report(name, A, B, C):
    tot = A + B + C
    print(f"{name}:")
    print(f"  A input-act  : {A/1e6:8.1f} M  ({A/tot*100:4.1f}%)")
    print(f"  B weight     : {B/1e6:8.1f} M  ({B/tot*100:4.1f}%)")
    print(f"  C output-act : {C/1e6:8.1f} M  ({C/tot*100:4.1f}%)")
    print(f"  activations (A+C) = {(A+C)/tot*100:.1f}% , weight = {B/tot*100:.1f}%")
    return B / tot


def main():
    print("Decomposing per-layer memory operands (linear/conv only)\n")
    sa, sb, sc = enum_sdm()
    bf_sdm = report("SDM UNet", sa, sb, sc)
    print()
    da, db, dc = enum_dit()
    bf_dit = report("DiT-XL/2", da, db, dc)
    print(f"\nWeight (B) share: SDM {bf_sdm*100:.1f}%  vs  DiT {bf_dit*100:.1f}%")
    if bf_dit > bf_sdm + 0.05:
        print("=> Weights dominate memory MORE in DiT. Counting full B every layer")
        print("   over-counts DiT memory -> the core/memory mismatch vs paper.")
        print("   An amortized-weight model (weights ~resident, read once not per-stream)")
        print("   is a GENERAL fix, larger effect on DiT. Worth implementing.")
    else:
        print("=> Weight share similar; weight over-count is NOT the main cause.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
gen_dit_trace.py - Phase 1 trace collection for DiT-XL/2 (GPU).

Collects real activations so we can measure DiT's TEMPORAL-DIFFERENCE bit-width
distribution (Fig 5 analog) instead of borrowing SDM's sparsity constants. We hook
4 representative transformer blocks (0/9/18/27, shallow->deep) across all DDIM
denoising steps for 2 class labels, and save per-(class,block,layer,step) input/
output activations as float16 .npz -- the SAME schema as the SDM trace, so the
existing reproduce_fig5_bitwidth.py can consume it with minimal change.

Scope (deliberately small; bit-width is a statistic that converges fast):
  2 classes x 25 DDIM steps x 4 blocks x (a few Linears each) -> ~1-2 GB, vs SDM 19GB.

Run on the GPU box (PACE-ICE H100 or local CUDA):
    cd ~/Ditto && python3 sim/gen_dit_trace.py
Output: traces/dit/<class>_<block>_<layer>_<step>.npz
"""
import os
from pathlib import Path

import numpy as np
import torch
from diffusers import DiTPipeline

OUT = Path.home() / "Ditto" / "traces" / "dit"
REP_BLOCKS = [0, 9, 18, 27]          # shallow / mid / deep representatives
# 8 representative ImageNet classes spanning animals / objects / scenes, so the
# bit-width statistic is not tied to one content type.
CLASSES = [207,   # golden retriever
           388,   # giant panda
           933,   # cheeseburger
           973,   # coral reef
           417,   # balloon
           555,   # fire engine
           985,   # daisy
           282]   # tiger cat
STEPS = 50                           # ALIGNED to SDM (PLMS 50-step, paper Table I)
GUIDANCE = 7.5                       # ALIGNED to SDM's guidance_scale=7.5
# NOTE: 7.5 is chosen to match the SDM collection for comparability of the
# temporal-difference statistic, NOT DiT's own default (~4.0). The conditioning
# input itself cannot be aligned (SDM = text/COCO caption via cross-attn; DiT =
# ImageNet class via ada_norm) -- that is a genuine model difference, so we align
# the sampling DYNAMICS (steps, CFG, denoiser granularity) and let each model use
# its natural conditioning. Steps are the dominant factor for diff sparsity.
# 3 representative Linears per block (option B): one attention projection (to_q;
# to_k/to_v/to_out share near-identical temporal-diff statistics, so they are
# redundant) plus the two MLP ends (ff.0 up-proj, ff.2 down-proj). Covers
# "attention projection" + "MLP both ends" without the redundant attn projections.
HOOK_SUFFIXES = ("attn1.to_q", "ff.net.0.proj", "ff.net.2")


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {dev}")
    OUT.mkdir(parents=True, exist_ok=True)

    pipe = DiTPipeline.from_pretrained("facebook/DiT-XL-2-256").to(dev)
    blocks = pipe.transformer.transformer_blocks
    print(f"DiT loaded, {len(blocks)} blocks; hooking blocks {REP_BLOCKS}")

    # state shared with hooks: current (class_id, step_idx)
    state = {"class": None, "step": -1}
    saved = {"n": 0, "bytes": 0}

    def make_hook(block_id, layer_name):
        def hook(mod, inp, out):
            x = inp[0]
            y = out[0] if isinstance(out, (tuple, list)) else out
            rec = {
                "input": x.detach().to(torch.float16).cpu().numpy(),
                "output": y.detach().to(torch.float16).cpu().numpy(),
                "layer_name": np.array(f"block{block_id}.{layer_name}"),
                "timestep": np.array(state["step"]),
                "image_idx": np.array(state["class"]),   # class id (SDM-compatible key)
            }
            f = OUT / f"c{state['class']}_b{block_id}_{layer_name.replace('.', '-')}_t{state['step']:02d}.npz"
            np.savez_compressed(f, **rec)
            saved["n"] += 1
            saved["bytes"] += rec["input"].nbytes + rec["output"].nbytes
        return hook

    # register hooks on representative blocks' chosen Linears
    handles = []
    for bid in REP_BLOCKS:
        blk = blocks[bid]
        for name, mod in blk.named_modules():
            if any(name.endswith(suf) for suf in HOOK_SUFFIXES) and isinstance(mod, torch.nn.Linear):
                handles.append(mod.register_forward_hook(make_hook(bid, name)))
    print(f"{len(handles)} hooks registered")

    # step counter: wrap the scheduler.step to advance our step index
    orig_step = pipe.scheduler.step
    def counting_step(*a, **k):
        state["step"] += 1
        return orig_step(*a, **k)
    pipe.scheduler.step = counting_step

    for cls in CLASSES:
        state["class"] = cls
        state["step"] = -1
        gen = torch.Generator(device=dev).manual_seed(0)
        print(f"  generating class {cls} ...", flush=True)
        with torch.no_grad():
            pipe(class_labels=[cls], num_inference_steps=STEPS,
                 guidance_scale=GUIDANCE, generator=gen, output_type="numpy")

    for h in handles:
        h.remove()
    pipe.scheduler.step = orig_step

    print(f"\nsaved {saved['n']} npz files, ~{saved['bytes']/1e9:.2f} GB to {OUT}")
    print("Next: adapt reproduce_fig5_bitwidth.py to read traces/dit/ (same schema)")
    print("to get DiT's temporal-difference bit-width distribution (vs SDM 45.9% zero).")


if __name__ == "__main__":
    main()

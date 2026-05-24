#!/usr/bin/env python3
"""
fastdllm_structure.py - Ditto speedup/energy for Fast-dLLM v2 (Qwen2.5-7B, GQA).

Third workload in the Ditto framework. Enumerates all Linear MACs from one forward
(handling GQA: 28 query heads but only 4 KV heads -> k/v_proj are 1/7 of q_proj, and
QK/PV use the grouped KV), adds attention QK/PV, then runs the validated roofline +
six-segment energy with Fast-dLLM's OWN measured temporal sparsity (zero 80.3%,
<=4-bit 19.4%, >4-bit 0.2%; gen_fastdllm_ditto.py), so it is comparable to SDM/DiT.

Sequence length: representative 256 tokens (prompt+block scale; aligns with DiT for
comparison). Honest: a real decode varies length per block; 256 is a representative
single-forward snapshot for structural MAC accounting, not a full-generation trace.

    cd ~/Ditto && python3 sim/fastdllm_structure.py
"""
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM

import fig13_roofline as RF
import energy_model as EN

MODEL = "/home/njzhx/models/Fast_dLLM_v2_7B"
SEQ = 256
# Fast-dLLM v2 measured Ditto temporal-difference bit-width (gen_fastdllm_ditto.py)
FDLLM_ZERO, FDLLM_LE4, FDLLM_GT4 = 0.803, 0.194, 0.002


def enumerate_fastdllm():
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, trust_remote_code=True, torch_dtype=torch.bfloat16).to("cuda").eval()
    c = model.config
    heads = c.num_attention_heads          # 28
    kv_heads = c.num_key_value_heads       # 4 (GQA)
    hd = c.hidden_size // heads            # 128

    shapes = {}

    def hook(nm):
        def h(m, i, o):
            y = o[0] if isinstance(o, (tuple, list)) else o
            shapes[nm] = (tuple(i[0].shape), tuple(y.shape))
        return h
    hs = [mm.register_forward_hook(hook(n)) for n, mm in model.named_modules()
          if isinstance(mm, nn.Linear)]
    ids = torch.randint(0, 1000, (1, SEQ), device="cuda")
    with torch.no_grad():
        model.model(ids)
    for h in hs:
        h.remove()

    layers = []
    # all Linear projections (q/k/v/o, gate/up/down) -- GQA already reflected in shapes
    for n, (in_s, out_s) in shapes.items():
        M = 1
        for d in in_s[:-1]:
            M *= d
        K, N = in_s[-1], out_s[-1]
        layers.append({"cat": "linear", "macs": float(M * K * N),
                       "A": float(M * K), "B": float(K * N), "C": float(M * N)})

    # attention QK/PV per layer, GQA: Q has `heads` heads, K/V have `kv_heads`.
    # Effective QK/PV cost ~ heads * seq * seq * hd for the scores/context, but the
    # KV operands are shared across head-groups (kv_heads distinct K/V). We count the
    # full Q-head work (each query head attends), which is the compute that runs.
    n_blocks = len(model.model.layers)
    for _ in range(n_blocks):
        for (M, Kd, Nn) in [(SEQ, hd, SEQ), (SEQ, SEQ, hd)]:   # QK^T, then (scores)V
            macs = heads * M * Kd * Nn
            layers.append({"cat": "attn_self", "macs": float(macs),
                           "A": float(heads * M * Kd),
                           "B": float(kv_heads * Kd * Nn),   # KV operand: only kv_heads
                           "C": float(heads * M * Nn)})
    return layers, heads, kv_heads


def main():
    print("Loading Fast-dLLM v2 (7B) and enumerating (one forward) ...", flush=True)
    layers, heads, kv_heads = enumerate_fastdllm()

    lin = [L for L in layers if L["cat"] == "linear"]
    att = [L for L in layers if L["cat"] == "attn_self"]
    mac_lin = sum(L["macs"] for L in lin) / 1e9
    mac_att = sum(L["macs"] for L in att) / 1e9
    total = mac_lin + mac_att
    print(f"\n=== Fast-dLLM v2 structure (GQA {heads}Q/{kv_heads}KV heads, seq {SEQ}) ===")
    print(f"  linear/proj MACs : {mac_lin:8.2f} G")
    print(f"  self-attn QK/PV  : {mac_att:8.2f} G")
    print(f"  TOTAL            : {total:8.2f} G")
    print(f"  attention fraction: {mac_att/total*100:.1f}%  (SDM 15.7%, DiT 3.6%)")

    # plug Fast-dLLM's own bit-width into roofline + energy
    RF.BW_ZERO, RF.BW_LE4, RF.BW_GT4 = FDLLM_ZERO, FDLLM_LE4, FDLLM_GT4
    RF.NONZERO = FDLLM_LE4 + FDLLM_GT4
    RF.BITF = (FDLLM_LE4 + 2 * FDLLM_GT4) / RF.NONZERO
    EN.BW_LE4, EN.BW_GT4 = FDLLM_LE4, FDLLM_GT4
    EN.NONZERO = FDLLM_LE4 + FDLLM_GT4
    EN.E_MAC_DIFF = (FDLLM_LE4 * EN.E_MAC4 + FDLLM_GT4 * EN.E_MAC8) / EN.NONZERO

    print("\n=== speedup (Fast-dLLM's own bit-width, zero 80.3%) ===")
    print(f"  {'BW(B/cyc)':>10} {'diff-only':>10} {'+Defo':>8} {'flip%':>7}")
    for bw in [64, 256, 1024, 4096]:
        r = RF.run_bw(layers, bw)
        print(f"  {bw:>10} {r['diff_only']:>9.2f}x {r['defo']:>7.2f}x {r['flip']*100:>6.1f}%")
    print(f"  compute ceiling: {RF.run_bw(layers, 1e12)['defo']:.2f}x")

    print("\n=== energy (Fast-dLLM's own bit-width) ===")
    itc = EN.total_energy(layers, "itc")
    camd = EN.total_energy(layers, "camd")
    flip = EN.compute_flip(layers)
    defo = EN.total_energy(layers, "ditto", defo_flip=flip)
    it = EN.etotal(itc)
    print(f"  ITC=1.00  Cam-D={EN.etotal(camd)/it:.2f}x  "
          f"Ditto={EN.etotal(defo)/it:.2f}x (saves {(1-EN.etotal(defo)/it)*100:.1f}%)")
    print(f"  ITC core fraction = {itc['core']/it:.0%}")
    print("\nSDM saves ~34% / DiT ~46% / Fast-dLLM here. Higher temporal sparsity")
    print("(80.3%) -> more skip. Ceiling is theoretical (BW->inf); realistic-BW")
    print("speedup is memory/Defo bound like the others. GQA handled in enumeration.")


if __name__ == "__main__":
    main()

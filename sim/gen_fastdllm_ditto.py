#!/usr/bin/env python3
"""
gen_fastdllm_ditto.py - apply Ditto's temporal-difference bit-width to Fast-dLLM v2.

This brings Fast-dLLM v2 (Qwen2.5-7B diffusion LLM) into the Ditto framework as a
THIRD workload, using DITTO's mechanism (per-element temporal difference -> int8
quantize -> zero-skip + bit-width classify), distinct from the module-level
reuse-skipping already delivered separately. It tests whether Ditto's
"temporal value similarity" assumption -- proven for image diffusion (SDM 45.9%,
DiT 67.2% zero) -- also holds for TEXT diffusion.

Method: reuse the Phase-B StepCacheManager's verified step lifecycle (it pairs each
layer's H_in at adjacent denoising steps in finalize_step). We SUBCLASS it and, at
each pairing, quantize the H_in difference with the SAME ruler as SDM/DiT and
accumulate zero / <=4-bit / >4-bit counts. Nothing is dumped to disk except the final
counts (so 7B x ~300 steps does not blow up). The Phase-B code is untouched.

Honest expectation: Phase-B sim data showed most tokens ~0.99 similar but a few newly
-unmasked tokens drop to ~0.03 -- so the difference is likely BIMODAL (many ~zero
diffs from settled tokens + a few large diffs from just-unmasked tokens), unlike
image diffusion's uniform small changes. High zero%, possibly higher >4-bit outliers.

    cd ~/Ditto && python3 sim/gen_fastdllm_ditto.py
"""
import sys
import re
import numpy as np
import torch

sys.path.insert(0, "/home/njzhx/fastdllm-skipping-export/src")
sys.path.insert(0, "/home/njzhx/Fast-dLLM/v2")

from step_cache import StepCacheManager, HookSession, attach_hooks  # Phase-B (untouched)
import generation_functions
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import types

MODEL_PATH = "/home/njzhx/models/Fast_dLLM_v2_7B"
REP_LAYERS = [0, 9, 18, 27]     # representative layers (align with DiT's 4 blocks)
N_SAMPLES = 4
QMAX = 127


def classify(diff):
    z = int((diff == 0).sum())
    le = int(((diff >= -8) & (diff <= 7) & (diff != 0)).sum())
    gt = int(((diff < -8) | (diff > 7)).sum())
    return z, le, gt


class DittoBitwidthManager(StepCacheManager):
    """Subclass: at each adjacent-step H_in pairing, accumulate Ditto diff bit-width
    for the representative layers. Counts only; no per-step dump."""

    def __init__(self, **kw):
        super().__init__(**kw)
        self.ditto_counts = {l: [0, 0, 0] for l in REP_LAYERS}  # layer -> [z,le,gt]
        self.ditto_pairs = 0

    def finalize_step(self):
        is_first = (self.local_step_in_block == 0)
        if not is_first:
            for layer_id in REP_LAYERS:
                k = (layer_id, "H_in")
                if k in self.this_step and k in self.prev_step:
                    cur = self.this_step[k]
                    prev = self.prev_step[k]
                    if cur.shape != prev.shape:
                        ml = min(cur.shape[-2], prev.shape[-2])
                        cur = cur[..., -ml:, :]
                        prev = prev[..., -ml:, :]
                    c = cur.float().cpu().numpy()
                    p = prev.float().cpu().numpy()
                    am = float(max(np.abs(c).max(), np.abs(p).max()))
                    s = am / QMAX if am > 1e-12 else 1.0
                    diff = (np.clip(np.round(c / s), -QMAX, QMAX)
                            - np.clip(np.round(p / s), -QMAX, QMAX)).astype(np.int32)
                    z, le, gt = classify(diff)
                    a = self.ditto_counts[layer_id]
                    a[0] += z; a[1] += le; a[2] += gt
                    self.ditto_pairs += 1
        # roll (same as parent, but we skip the sim/record bookkeeping)
        self.prev_step = self.this_step
        self.this_step = {}
        self.is_first_step_in_block = False


def main():
    print("Loading Fast-dLLM v2 (7B) ...", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, trust_remote_code=True, torch_dtype=torch.bfloat16).to("cuda")
    model.eval()
    model.mdm_sample = types.MethodType(
        generation_functions.Fast_dLLM_QwenForCausalLM.batch_sample, model)

    mgr = DittoBitwidthManager(num_layers=28, policy=None, dump_dir=None)
    model.step_cache_manager = mgr

    ds = load_dataset("openai/gsm8k", "main", split="test")

    with HookSession(model, mgr):
        for i in range(N_SAMPLES):
            q = ds[i]["question"] + (" Please reason step by step, and put your "
                                     "final answer within \\boxed{}.")
            msgs = [{"role": "user", "content": q}]
            pt = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
            inp = tok([pt], return_tensors="pt").to("cuda")
            ids = inp["input_ids"]
            ml = ids.shape[1]
            model.current_sample_id = i
            print(f"  sample {i} ...", flush=True)
            with torch.no_grad():
                model.mdm_sample(
                    input_ids=ids, tokenizer=tok, block_size=32, small_block_size=8,
                    max_new_tokens=512, mask_id=151665, min_len=ml,
                    seq_len=torch.tensor([ml], device="cuda"),
                    use_block_cache=False, threshold=1.0)

    # aggregate
    print(f"\n=== Fast-dLLM v2 Ditto temporal-difference bit-width "
          f"({mgr.ditto_pairs} layer-pairs, H_in) ===\n")
    print(f"{'layer':>6} {'zero%':>7} {'<=4bit%':>8} {'>4bit%':>7}")
    tot = [0, 0, 0]
    for l in REP_LAYERS:
        z, le, gt = mgr.ditto_counts[l]
        t = z + le + gt
        tot[0] += z; tot[1] += le; tot[2] += gt
        if t:
            print(f"{l:>6} {z/t*100:>6.1f}% {le/t*100:>7.1f}% {gt/t*100:>6.1f}%")
    T = sum(tot)
    print("-" * 32)
    print(f"{'ALL':>6} {tot[0]/T*100:>6.1f}% {tot[1]/T*100:>7.1f}% {tot[2]/T*100:>6.1f}%")
    print(f"\nSDM 45.9% / 53.4% / 0.7%   DiT(MAC-wt) 67.2% / 32.6% / 0.2%")
    print("Same quantizer/classifier as SDM & DiT (comparable). H_in is the post-LN")
    print("attention input -- the same signal the module-skip study used.")
    print("\nIf zero% is high but >4bit% is also elevated, that is the predicted bimodal")
    print("text-diffusion pattern (settled tokens ~0 diff, just-unmasked tokens large).")


if __name__ == "__main__":
    main()

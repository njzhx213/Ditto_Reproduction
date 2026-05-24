"""
diagnose_bitwidth.py — Why is our temporal-diff bit-width distribution off?

Our Day-2 real-trace test showed ~43% >4-bit vs the paper's ~4%. Equivalence
holds (the math is exact), so this is purely a quantization-scale question:
what scale makes the temporal differences fall into the zero / 4-bit buckets
the way the paper reports?

This script diagnoses:
  1. The actual temporal cosine similarity of our traces (should be ~0.98 like
     paper Fig 3 if the traces are good).
  2. The fp-domain difference magnitude.
  3. How the choice of quantization scale (per-pair vs global-over-trajectory)
     changes the zero / 4-bit / >4-bit split.

Run:
    cd ~/Ditto
    python tests/diagnose_bitwidth.py
"""

from __future__ import annotations

from pathlib import Path
import numpy as np

TRACE = Path.home() / "Ditto/traces/sdm/image_000"


def load(step, layer="conv_in", which="input"):
    return np.load(TRACE / f"step_{step:03d}" / f"{layer}.npz")[which].astype(np.float32)


def main():
    print("=== Diagnose temporal similarity of real SDM activations ===\n")

    for layer in ["conv_in", "down_blocks_2_attentions_0"]:
        print(f"--- {layer} ---")
        for s in [5, 25, 45]:
            try:
                prev = load(s, layer)
                curr = load(s + 1, layer)
            except FileNotFoundError:
                print(f"  step {s}->{s+1}: trace missing")
                continue

            cos = (prev.flatten() @ curr.flatten()) / (
                np.linalg.norm(prev) * np.linalg.norm(curr) + 1e-9
            )
            fp_diff = curr - prev
            print(
                f"  step {s}->{s+1}: cos_sim={cos:.4f}  "
                f"act_range=[{prev.min():.2f},{prev.max():.2f}]  "
                f"diff_range=[{fp_diff.min():.3f},{fp_diff.max():.3f}]  "
                f"diff_std={fp_diff.std():.4f}"
            )
        print()

    # Effect of scale choice
    print("=== Effect of scale choice on zero/4-bit ratio (conv_in, step 25->26) ===")
    prev = load(25, "conv_in")
    curr = load(26, "conv_in")

    absmax_pair = max(np.abs(prev).max(), np.abs(curr).max())

    all_max = 0.0
    for s in range(50):
        try:
            all_max = max(all_max, float(np.abs(load(s, "conv_in")).max()))
        except FileNotFoundError:
            pass

    for label, absmax in [("per-pair", absmax_pair), ("global-50step", all_max)]:
        scale = absmax / 127
        p_int = np.clip(np.round(prev / scale), -127, 127).astype(np.int32)
        c_int = np.clip(np.round(curr / scale), -127, 127).astype(np.int32)
        d = c_int - p_int
        zero = (d == 0).mean()
        le4 = (np.abs(d) <= 7).mean()
        gt4 = (np.abs(d) > 7).mean()
        print(
            f"  {label:16s} scale={scale:.4f}  "
            f"zero={zero:.1%}  ≤4bit={le4:.1%}  >4bit={gt4:.1%}"
        )

    # Also try: what if we use a coarser scale (fewer int levels for activation)?
    print("\n=== What scale would the paper need? (reverse-engineer) ===")
    print("If most diffs should be ≤7 in int domain, and fp diff_std ~ X,")
    print("then scale should be ~ diff_std (so a 1-sigma fp change ≈ 1 int level).")
    fp_diff = curr - prev
    print(f"  conv_in step25->26: fp diff_std = {fp_diff.std():.4f}")
    print(f"  per-pair scale       = {absmax_pair/127:.4f}")
    print(f"  ratio (diff_std/scale) = {fp_diff.std()/(absmax_pair/127):.2f}")
    print("  (If this ratio >> 1, the diff spans many int levels -> too many >4-bit.)")


if __name__ == "__main__":
    main()

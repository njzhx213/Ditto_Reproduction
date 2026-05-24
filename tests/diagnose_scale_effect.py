"""
diagnose_scale_effect.py — Why is our >4-bit ~0% vs paper's 4%?

Our temporal differences almost never exceed 4-bit, while the paper reports ~4%
>4-bit on average. Equivalence is exact, so this is a quantization-scale question.

Hypothesis: our per-tensor dynamic scale is "loose" — a few outliers inflate
absmax, so the scale is large, so most differences map to small int deltas
(<=7). The paper's Q-Diffusion calibration may use a tighter scale (clipping
outliers), producing larger int deltas and thus more >4-bit elements.

This script sweeps the scale (via a percentile-based absmax instead of true max)
and shows how >4-bit fraction responds.

Run:
    cd ~/Ditto
    python tests/diagnose_scale_effect.py
"""

from __future__ import annotations

from pathlib import Path
import numpy as np

TRACE = Path.home() / "Ditto/traces/sdm/image_000"

LAYERS = {
    "conv_in": "conv_in",
    "down2_attn": "down_blocks_2_attentions_0",
    "mid_attn": "mid_block_attentions_0",
}


def load(step, layer, which="input"):
    return np.load(TRACE / f"step_{step:03d}" / f"{layer}.npz")[which].astype(np.float32)


def classify_signed_range(diff):
    """Return (zero, le4, gt4) fractions using signed-range [-8,7] = 4-bit."""
    zero = (diff == 0).mean()
    le4 = ((diff >= -8) & (diff <= 7) & (diff != 0)).mean()
    gt4 = ((diff < -8) | (diff > 7)).mean()
    return zero, le4, gt4


def quantize_with_absmax(x, absmax, n_bits=8):
    qmax = (1 << (n_bits - 1)) - 1
    scale = absmax / qmax
    return np.clip(np.round(x / scale), -qmax, qmax).astype(np.int32), scale


def main():
    print("=== Scale tightness effect on >4-bit fraction ===\n")
    print("For each layer (step 25->26), sweep the absmax percentile used for scale.")
    print("Tighter scale (lower percentile) = clip more outliers = larger int diffs.\n")

    step_prev, step_curr = 25, 26

    for key, layer_file in LAYERS.items():
        prev = load(step_prev, layer_file)
        curr = load(step_curr, layer_file)

        print(f"--- {key} ({layer_file}) ---")
        print(f"{'absmax basis':<22} {'scale':>9} {'zero%':>7} {'≤4bit%':>8} {'>4bit%':>8}")

        true_absmax = max(np.abs(prev).max(), np.abs(curr).max())

        for label, pct in [
            ("true max (100.0%)", 100.0),
            ("99.9 percentile", 99.9),
            ("99.5 percentile", 99.5),
            ("99.0 percentile", 99.0),
            ("98.0 percentile", 98.0),
        ]:
            if pct >= 100.0:
                absmax = true_absmax
            else:
                stacked = np.concatenate([np.abs(prev).flatten(), np.abs(curr).flatten()])
                absmax = np.percentile(stacked, pct)

            p_int, scale = quantize_with_absmax(prev, absmax)
            c_int, _ = quantize_with_absmax(curr, absmax)
            diff = c_int - p_int
            zero, le4, gt4 = classify_signed_range(diff)
            print(f"{label:<22} {scale:>9.4f} {zero*100:>6.1f} {le4*100+zero*100:>7.1f} {gt4*100:>7.1f}")
        print()

    print("Interpretation:")
    print("  If >4-bit% rises as we tighten the scale (lower percentile), then")
    print("  the paper's ~4% >4-bit likely comes from a tighter calibration scale")
    print("  (Q-Diffusion clips outliers). Our 'loose' dynamic absmax scale is why")
    print("  we see ~0% >4-bit. Both are 'correct'; the paper just calibrates tighter.")


if __name__ == "__main__":
    main()

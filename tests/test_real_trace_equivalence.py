"""
test_real_trace_equivalence.py — Validate Ditto on REAL SDM activation traces.

This is the moment of truth: load the actual SDM v1.4 activation traces collected
in Week 1 Day 1 (~/Ditto/traces/sdm/), and verify two things:

  1. EQUIVALENCE: the Ditto temporal-difference path reproduces the direct
     quantized MAC bitwise-exactly on real data.

  2. FIG 5 PREVIEW: the temporal-difference bit-width distribution (zero / 4-bit /
     >4-bit) on real SDM activations should approach the paper's reported
     ~44% zero, ~96% ≤4-bit, ~4% >4-bit.

Unlike the synthetic self-tests, this uses real adjacent-timestep activations,
which have genuine temporal similarity — so we expect the ">4-bit" fraction to
be much smaller than synthetic data (closer to the paper's 4%).

Usage:
    cd ~/Ditto
    python tests/test_real_trace_equivalence.py
    # or specify layer / steps:
    python tests/test_real_trace_equivalence.py --layer conv_in --image 0

Author: njzhx213
Project: Ditto Reproduction (HPCA 2025)
Phase: Week 1, Day 2 (Step A on real data)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# Make src/functional importable
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src" / "functional"))

from difference_processor import EncodingUnit, EncodedBatch  # noqa: E402
from compute_unit_func import PE, direct_mac                  # noqa: E402
from quantize import quantize_pair_shared_scale, quantize_per_tensor  # noqa: E402
from ditto_linear import ditto_linear_dot, ditto_linear_dot_int     # noqa: E402


TRACE_ROOT = Path.home() / "Ditto" / "traces" / "sdm"

# Layer file names (sanitized — dots replaced with underscores)
LAYER_FILES = {
    "conv_in": "conv_in",
    "down1_attn": "down_blocks_1_attentions_0",
    "down2_attn": "down_blocks_2_attentions_0",
    "mid_attn": "mid_block_attentions_0",
    "up1_attn": "up_blocks_1_attentions_0",
    "up2_attn": "up_blocks_2_attentions_0",
}


def load_activation(image_idx: int, step_idx: int, layer_file: str, which: str = "input") -> np.ndarray:
    """Load one activation tensor from the trace .npz."""
    path = TRACE_ROOT / f"image_{image_idx:03d}" / f"step_{step_idx:03d}" / f"{layer_file}.npz"
    if not path.exists():
        raise FileNotFoundError(f"Trace not found: {path}")
    data = np.load(path)
    return data[which].astype(np.float32)


def test_equivalence_one_layer(
    image_idx: int,
    layer_file: str,
    step_prev: int,
    step_curr: int,
    n_rows: int = 64,
) -> dict:
    """
    Test Ditto equivalence + bit-width stats on one layer's adjacent-step activations.

    We flatten the activation to 2D [tokens/spatial, channels] and take n_rows
    rows, treating each row as a dot-product input against a fake weight vector.
    """
    prev = load_activation(image_idx, step_prev, layer_file, "input")
    curr = load_activation(image_idx, step_curr, layer_file, "input")

    # Flatten to 2D: collapse all but the last dim into rows.
    # conv activations are [B, C, H, W]; attention are [B, T, D].
    prev_2d = prev.reshape(-1, prev.shape[-1])
    curr_2d = curr.reshape(-1, curr.shape[-1])

    assert prev_2d.shape == curr_2d.shape, f"shape mismatch {prev_2d.shape} vs {curr_2d.shape}"

    N = prev_2d.shape[-1]
    M = min(n_rows, prev_2d.shape[0])

    # --- PER-TENSOR quantization (paper's "dynamic quantization") ---
    # Compute ONE shared scale over the WHOLE tensor (both adjacent steps),
    # not per-row. Per-row scale is unstable on activations with outliers and
    # artificially inflates the temporal-difference bit-width.
    q_prev, q_curr = quantize_pair_shared_scale(prev, curr)
    prev_int_2d = q_prev.int_data.reshape(-1, N)
    curr_int_2d = q_curr.int_data.reshape(-1, N)

    # Fake weight, quantized per-tensor (equivalence independent of weight values)
    rng = np.random.default_rng(42)
    weight = rng.standard_normal(N).astype(np.float32)
    q_weight = quantize_per_tensor(weight)
    w_int = q_weight.int_data

    n_exact = 0
    agg_zero = agg_low4 = agg_high4 = agg_total = 0
    total_cycles = 0

    for m in range(M):
        res = ditto_linear_dot_int(prev_int_2d[m], curr_int_2d[m], w_int)
        if res.is_exact:
            n_exact += 1
        s = res.encoded_batch.stats
        agg_zero += s.n_zero
        agg_low4 += s.n_low4
        agg_high4 += s.n_high4
        agg_total += s.n_total
        total_cycles += res.pe_cycles

    return {
        "layer": layer_file,
        "shape": prev.shape,
        "rows_tested": M,
        "input_dim": N,
        "exact_rows": n_exact,
        "all_exact": n_exact == M,
        "zero_ratio": agg_zero / max(agg_total, 1),
        "low4_ratio": agg_low4 / max(agg_total, 1),
        "high4_ratio": agg_high4 / max(agg_total, 1),
        "le4bit_ratio": (agg_zero + agg_low4) / max(agg_total, 1),
        "total_cycles": total_cycles,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", type=int, default=0)
    ap.add_argument("--step-prev", type=int, default=25)
    ap.add_argument("--step-curr", type=int, default=26)
    ap.add_argument("--n-rows", type=int, default=64)
    ap.add_argument("--layer", type=str, default="all",
                    help="Layer key (conv_in/down1_attn/.../all)")
    args = ap.parse_args()

    if not TRACE_ROOT.exists():
        print(f"✗ Trace directory not found: {TRACE_ROOT}")
        print("  Run collect_sdm_traces.py first.")
        sys.exit(1)

    # Which layers to test
    if args.layer == "all":
        layers = list(LAYER_FILES.items())
    else:
        if args.layer not in LAYER_FILES:
            print(f"✗ Unknown layer '{args.layer}'. Options: {list(LAYER_FILES.keys())}")
            sys.exit(1)
        layers = [(args.layer, LAYER_FILES[args.layer])]

    print(f"=== Real SDM trace equivalence test ===")
    print(f"Image {args.image}, step {args.step_prev} -> {args.step_curr}, {args.n_rows} rows/layer")
    print(f"Trace root: {TRACE_ROOT}")
    print()

    header = f"{'layer':<30} {'shape':<22} {'exact':<10} {'zero%':>7} {'4bit%':>7} {'>4bit%':>8} {'≤4bit%':>8}"
    print(header)
    print("-" * len(header))

    all_pass = True
    agg = {"zero": [], "low4": [], "high4": [], "le4": []}

    for key, layer_file in layers:
        try:
            r = test_equivalence_one_layer(
                args.image, layer_file, args.step_prev, args.step_curr, args.n_rows
            )
        except FileNotFoundError as e:
            print(f"{key:<30} SKIP (trace missing: {e})")
            continue

        exact_str = f"{r['exact_rows']}/{r['rows_tested']}"
        flag = "✓" if r["all_exact"] else "✗"
        if not r["all_exact"]:
            all_pass = False

        print(
            f"{layer_file:<30} {str(r['shape']):<22} {exact_str:<8}{flag:<2} "
            f"{r['zero_ratio']*100:>6.1f} {r['low4_ratio']*100:>6.1f} "
            f"{r['high4_ratio']*100:>7.1f} {r['le4bit_ratio']*100:>7.1f}"
        )

        agg["zero"].append(r["zero_ratio"])
        agg["low4"].append(r["low4_ratio"])
        agg["high4"].append(r["high4_ratio"])
        agg["le4"].append(r["le4bit_ratio"])

    print("-" * len(header))
    if agg["zero"]:
        print(
            f"{'AVERAGE':<30} {'':<22} {'':<10} "
            f"{np.mean(agg['zero'])*100:>6.1f} {np.mean(agg['low4'])*100:>6.1f} "
            f"{np.mean(agg['high4'])*100:>7.1f} {np.mean(agg['le4'])*100:>7.1f}"
        )
    print()
    print(f"Paper Fig 5 SDM (temporal diff): zero ~44%, ≤4-bit ~96%, >4-bit ~4%")
    print()

    if all_pass:
        print("✓ EQUIVALENCE HOLDS on all tested layers (Ditto == direct, bitwise).")
    else:
        print("✗ Some layers failed equivalence — investigate.")
        sys.exit(1)


if __name__ == "__main__":
    main()

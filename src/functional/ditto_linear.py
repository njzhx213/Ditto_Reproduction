"""
ditto_linear.py — Full Ditto linear-layer execution (the 3-stage algorithm).

Implements paper Section IV-A (Fig 7): the temporal-difference processing of a
linear layer, combining:
  - quantize.py        (A8W8 dynamic quantization, Step A)
  - difference_processor.py  (Encoding Unit: diff + classify + reorder)
  - compute_unit_func.py     (PE: mixed-precision MAC)

Three stages (paper Fig 7):
  Stage 1: δ = quant(act_t) - quant(act_{t-1})            [Encoding Unit]
  Stage 2: partial = δ × W                                 [Compute Unit / PE]
  Stage 3: out_t = out_{t-1} + partial                     [VPU summation]

Key identity (integer-exact in the quantized domain):
    out_t = out_{t-1} + δ × W
          = act_{t-1}_int × W + (act_t_int - act_{t-1}_int) × W
          = act_t_int × W
    i.e. the Ditto path reproduces the direct quantized MAC exactly.

This module operates on a single "row" (a 1-D activation vector × a 1-D weight
vector producing one output scalar). A real layer is many such dot products;
the equivalence holds element-wise so a single row is sufficient to validate
correctness. The cycle simulator (Week 2) will handle full matrices.

Author: njzhx213
Project: Ditto Reproduction (HPCA 2025)
Phase: Week 1, Day 2 (Step A)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    from difference_processor import EncodingUnit, EncodedBatch
    from compute_unit_func import PE, direct_mac
    from quantize import quantize_pair_shared_scale, quantize_per_tensor, QuantTensor
except ImportError:
    from .difference_processor import EncodingUnit, EncodedBatch
    from .compute_unit_func import PE, direct_mac
    from .quantize import quantize_pair_shared_scale, quantize_per_tensor, QuantTensor


@dataclass
class DittoLinearResult:
    """Result of one Ditto linear-layer dot product."""
    out_ditto: int            # out_{t-1} + δ×W  (int accumulator)
    out_direct: int           # act_t_int × W    (golden reference)
    delta_dot_w: int          # δ×W from the PE
    encoded_batch: EncodedBatch
    pe_cycles: int
    is_exact: bool            # out_ditto == out_direct ?


def ditto_linear_dot(
    prev_act_fp: np.ndarray,
    curr_act_fp: np.ndarray,
    weight_fp: np.ndarray,
    out_prev_int: int | None = None,
) -> DittoLinearResult:
    """
    Execute one Ditto linear dot product with temporal difference processing.

    Args:
        prev_act_fp:  activation at t-1, fp16/fp32, shape [N]
        curr_act_fp:  activation at t, fp16/fp32, shape [N]
        weight_fp:    weights, fp16/fp32, shape [N]
        out_prev_int: precomputed out_{t-1} = act_{t-1}_int × W_int.
                      If None, computed internally (the "first step ran with
                      original activations" assumption).

    Returns:
        DittoLinearResult with both Ditto and direct outputs + exactness flag.
    """
    # --- Quantize activations with shared scale (adjacent time steps) ---
    q_prev, q_curr = quantize_pair_shared_scale(prev_act_fp, curr_act_fp)

    # --- Quantize weights separately (per-tensor) ---
    q_weight = quantize_per_tensor(weight_fp)

    prev_int = q_prev.int_data.astype(np.int32)
    curr_int = q_curr.int_data.astype(np.int32)
    w_int = q_weight.int_data.astype(np.int32)

    # --- out_{t-1}: the cached previous-step output (int domain) ---
    if out_prev_int is None:
        out_prev_int = direct_mac(prev_int, w_int)

    # --- Direct (golden) reference: act_t_int × W_int ---
    out_direct = direct_mac(curr_int, w_int)

    # --- Ditto path ---
    # Stage 1: Encoding Unit computes δ = curr_int - prev_int, classifies & reorders
    eu = EncodingUnit(n_lanes=4)
    batch = eu.process_tensor(
        q_prev.int_data,    # int8
        q_curr.int_data,    # int8
        q_weight.int_data,  # int8
    )

    # Stage 2: PE computes Σ δ_i × W_i
    pe = PE()
    delta_dot_w = pe.run_batch(batch)

    # Stage 3: out_t = out_{t-1} + δ×W
    out_ditto = out_prev_int + delta_dot_w

    return DittoLinearResult(
        out_ditto=out_ditto,
        out_direct=out_direct,
        delta_dot_w=delta_dot_w,
        encoded_batch=batch,
        pe_cycles=pe.cycles,
        is_exact=(out_ditto == out_direct),
    )


def ditto_linear_dot_int(
    prev_int: np.ndarray,    # int8, shape [N], already quantized (per-tensor upstream)
    curr_int: np.ndarray,    # int8, shape [N]
    w_int: np.ndarray,       # int8, shape [N]
    out_prev_int: int | None = None,
) -> DittoLinearResult:
    """
    Same as ditto_linear_dot, but takes PRE-QUANTIZED int8 inputs.

    This is the correct entry point when quantization scale is computed at the
    TENSOR level (the paper's per-tensor dynamic quantization), not per row.
    The caller quantizes the whole activation tensor with one shared scale, then
    feeds individual rows here. This avoids the per-row scale instability that
    inflates the temporal-difference bit-width.
    """
    prev_i32 = prev_int.astype(np.int32)
    curr_i32 = curr_int.astype(np.int32)
    w_i32 = w_int.astype(np.int32)

    if out_prev_int is None:
        out_prev_int = direct_mac(prev_i32, w_i32)
    out_direct = direct_mac(curr_i32, w_i32)

    eu = EncodingUnit(n_lanes=4)
    batch = eu.process_tensor(
        prev_int.astype(np.int8),
        curr_int.astype(np.int8),
        w_int.astype(np.int8),
    )

    pe = PE()
    delta_dot_w = pe.run_batch(batch)
    out_ditto = out_prev_int + delta_dot_w

    return DittoLinearResult(
        out_ditto=out_ditto,
        out_direct=out_direct,
        delta_dot_w=delta_dot_w,
        encoded_batch=batch,
        pe_cycles=pe.cycles,
        is_exact=(out_ditto == out_direct),
    )


def ditto_linear_layer(
    prev_act_fp: np.ndarray,    # [M, N] : M output rows, N input dim
    curr_act_fp: np.ndarray,    # [M, N]
    weight_fp: np.ndarray,      # [N] (shared weight vector for this demo)
) -> dict:
    """
    Apply ditto_linear_dot over M rows. Returns aggregate stats.

    For a real linear layer the weight would be [N, M]; here we use a shared
    weight vector across rows for simplicity since equivalence is per-row.
    """
    M = prev_act_fp.shape[0]
    n_exact = 0
    total_cycles = 0
    agg_zero = agg_low4 = agg_high4 = agg_total = 0

    for m in range(M):
        res = ditto_linear_dot(prev_act_fp[m], curr_act_fp[m], weight_fp)
        if res.is_exact:
            n_exact += 1
        total_cycles += res.pe_cycles
        s = res.encoded_batch.stats
        agg_zero += s.n_zero
        agg_low4 += s.n_low4
        agg_high4 += s.n_high4
        agg_total += s.n_total

    return {
        "rows": M,
        "exact_rows": n_exact,
        "all_exact": n_exact == M,
        "total_pe_cycles": total_cycles,
        "zero_ratio": agg_zero / max(agg_total, 1),
        "low4_ratio": agg_low4 / max(agg_total, 1),
        "high4_ratio": agg_high4 / max(agg_total, 1),
        "le4bit_ratio": (agg_zero + agg_low4) / max(agg_total, 1),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Self-test
# ─────────────────────────────────────────────────────────────────────────────

def _self_test():
    np.random.seed(13)

    M, N = 64, 320   # 64 output rows, 320 input dim (mimic conv_in channel count)

    # Two adjacent-timestep activations: highly similar (temporal similarity)
    prev = np.random.randn(M, N).astype(np.float32) * 2.5
    curr = prev + np.random.randn(M, N).astype(np.float32) * 0.08  # small temporal change

    # Fake weight (equivalence is independent of weight values)
    weight = np.random.randn(N).astype(np.float32)

    result = ditto_linear_layer(prev, curr, weight)

    print("=== Ditto linear layer self-test (quantized, fake weight) ===")
    print(f"Rows:                {result['rows']}")
    print(f"Exact rows:          {result['exact_rows']} / {result['rows']}")
    print(f"Total PE cycles:     {result['total_pe_cycles']}")
    print()
    print(f"Temporal diff bit-width distribution (paper Fig 5 territory):")
    print(f"  zero:      {result['zero_ratio']:.1%}")
    print(f"  4-bit:     {result['low4_ratio']:.1%}")
    print(f"  >4-bit:    {result['high4_ratio']:.1%}")
    print(f"  ≤4-bit:    {result['le4bit_ratio']:.1%}  (paper SDM target ~96%)")
    print()

    assert result["all_exact"], (
        f"EQUIVALENCE FAILED: only {result['exact_rows']}/{result['rows']} rows exact"
    )
    print("✓ All rows: Ditto quantized output == direct quantized output (bitwise-exact).")
    print("✓ Step A (A8W8 quant) integrated and equivalence preserved.")


if __name__ == "__main__":
    _self_test()

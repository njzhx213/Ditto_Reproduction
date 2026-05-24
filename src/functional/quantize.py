"""
quantize.py — A8W8 dynamic per-tensor quantization for the Ditto reproduction.

This is Step A of the two-step plan: after verifying the integer-exact Ditto
equivalence (Step C, in compute_unit_func.py), we add realistic A8W8
quantization so that the temporal-difference bit-width statistics (paper Fig 5)
become meaningful.

Quantization scheme (matching the paper's "simple dynamic quantization with
8-bit activation and weight"):
  - Per-tensor, dynamic (scale computed from each tensor's own range at runtime).
  - Symmetric signed int8: range [-127, 127], zero maps to 0.
  - scale = max(|x|) / 127
  - x_int = round(x / scale), clamped to [-127, 127]

Symmetric (zero_point = 0) is chosen because:
  1. It keeps the temporal difference identity clean: quant(a) - quant(b) has
     no zero-point offset to cancel.
  2. The paper's difference processing relies on quant(act_t) - quant(act_{t-1});
     with asymmetric quant the zero-points would need careful bookkeeping.

Author: njzhx213
Project: Ditto Reproduction (HPCA 2025)
Phase: Week 1, Day 2 (Step A)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class QuantTensor:
    """A quantized tensor with its scale (symmetric, zero_point implicitly 0)."""
    int_data: np.ndarray    # int8, same shape as original
    scale: float            # real_value ≈ int_data * scale

    def dequantize(self) -> np.ndarray:
        """Reconstruct approximate fp32 values."""
        return self.int_data.astype(np.float32) * self.scale


def quantize_per_tensor(x: np.ndarray, n_bits: int = 8) -> QuantTensor:
    """
    Symmetric per-tensor dynamic quantization.

    Args:
        x: fp16/fp32 tensor of any shape.
        n_bits: bit-width (default 8 -> int8).

    Returns:
        QuantTensor with int_data in [-(2^(n_bits-1)-1), 2^(n_bits-1)-1].
    """
    qmax = (1 << (n_bits - 1)) - 1     # 127 for int8
    x = x.astype(np.float32)

    absmax = float(np.abs(x).max())
    if absmax < 1e-12:
        # All-zero tensor -> scale 1, all ints zero
        return QuantTensor(int_data=np.zeros_like(x, dtype=np.int8), scale=1.0)

    scale = absmax / qmax
    x_int = np.round(x / scale).astype(np.int32)
    x_int = np.clip(x_int, -qmax, qmax).astype(np.int8)

    return QuantTensor(int_data=x_int, scale=scale)


def quantize_pair_shared_scale(
    prev: np.ndarray,
    curr: np.ndarray,
    n_bits: int = 8,
) -> tuple[QuantTensor, QuantTensor]:
    """
    Quantize two adjacent-timestep tensors with a SHARED scale.

    This is important for the Ditto temporal difference: if prev and curr used
    different scales, then quant(curr) - quant(prev) would not correspond to a
    clean integer difference of the underlying values. Using a shared scale
    (the max over both tensors) ensures the difference is computed in a single
    consistent integer grid.

    In the paper's hardware, the scale is fixed per layer across time steps
    (determined by calibration / running statistics), so a shared scale across
    adjacent steps is the faithful choice.

    Args:
        prev: activation at time step t-1 (fp16/fp32).
        curr: activation at time step t (fp16/fp32).

    Returns:
        (q_prev, q_curr) both quantized with the same scale.
    """
    qmax = (1 << (n_bits - 1)) - 1

    absmax = float(max(np.abs(prev).max(), np.abs(curr).max()))
    if absmax < 1e-12:
        z_prev = QuantTensor(np.zeros_like(prev, dtype=np.int8), 1.0)
        z_curr = QuantTensor(np.zeros_like(curr, dtype=np.int8), 1.0)
        return z_prev, z_curr

    scale = absmax / qmax

    prev_int = np.clip(np.round(prev.astype(np.float32) / scale), -qmax, qmax).astype(np.int8)
    curr_int = np.clip(np.round(curr.astype(np.float32) / scale), -qmax, qmax).astype(np.int8)

    return QuantTensor(prev_int, scale), QuantTensor(curr_int, scale)


# ─────────────────────────────────────────────────────────────────────────────
# Self-test
# ─────────────────────────────────────────────────────────────────────────────

def _self_test():
    np.random.seed(11)

    # Two similar tensors (mimic adjacent time steps)
    prev = np.random.randn(1000).astype(np.float32) * 3.0
    curr = prev + np.random.randn(1000).astype(np.float32) * 0.1  # small perturbation

    # Shared-scale quantization
    q_prev, q_curr = quantize_pair_shared_scale(prev, curr)

    print("=== A8W8 quantization self-test ===")
    print(f"prev range: [{prev.min():.3f}, {prev.max():.3f}]")
    print(f"curr range: [{curr.min():.3f}, {curr.max():.3f}]")
    print(f"shared scale: {q_prev.scale:.5f}")
    print(f"prev int range: [{q_prev.int_data.min()}, {q_prev.int_data.max()}]")
    print(f"curr int range: [{q_curr.int_data.min()}, {q_curr.int_data.max()}]")

    # Quantization error
    prev_err = np.abs(q_prev.dequantize() - prev).mean()
    print(f"mean abs quant error (prev): {prev_err:.5f}  (scale={q_prev.scale:.5f})")

    # Temporal difference distribution
    diff = q_curr.int_data.astype(np.int32) - q_prev.int_data.astype(np.int32)
    zero_frac = (diff == 0).mean()
    small_frac = ((np.abs(diff) <= 7) & (diff != 0)).mean()
    print(f"\nTemporal diff distribution (int8 domain):")
    print(f"  zero:        {zero_frac:.1%}")
    print(f"  |diff| ≤ 7:  {small_frac:.1%}")
    print(f"  |diff| > 7:  {(np.abs(diff) > 7).mean():.1%}")
    print("\n(With a 0.1-magnitude perturbation on a scale-3 signal, expect")
    print(" many zero/small diffs — the temporal-similarity story.)")

    assert q_prev.int_data.dtype == np.int8
    assert q_prev.scale == q_curr.scale, "pair must share scale"
    print("\n✓ Quantization self-test passed.")


if __name__ == "__main__":
    _self_test()

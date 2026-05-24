"""
compute_unit_func.py — Functional model of the Ditto Compute Unit / PE.

Mirrors paper Fig 12 (Section V-B). An adder-tree-based MAC unit that consumes
the EncodedElements produced by the Encoding Unit and accumulates a partial sum.

Mixed-precision handling:
  - A 4-bit element (metadata=0): multiply data × weight directly.
  - A high-nibble element (metadata=1): multiply data × weight, then shift << 4.
    The Encoding Unit splits each 8-bit difference into a low-nibble element
    (metadata=0) and a high-nibble element (metadata=1); the PE recombines them
    via shift-and-add across the adder tree, exactly as paper Fig 12 describes.

Equivalence identity (the key correctness property, integer-exact, no quant):
    For a single element with int8 difference δ and weight W,
        δ = (high_signed << 4) + low_unsigned
        δ × W = (high_signed × W) << 4 + (low_unsigned × W)
              └─ metadata=1 lane (shifted) ─┘ └─ metadata=0 lane ─┘
    Summed across all elements of a layer:
        out_t = out_{t-1} + Σ_i (δ_i × W_i) == Σ_i (act_t,i × W_i) = direct result.

This file is the Python reference model. RTL lives in rtl/pe.sv; the Cocotb
testbench compares RTL against this model.

Author: njzhx213
Project: Ditto Reproduction (HPCA 2025)
Phase: Week 1, Day 2
"""

from __future__ import annotations

from dataclasses import dataclass

# Import the Encoding Unit data structures (same package)
try:
    from difference_processor import EncodedElement, EncodedBatch, EncodingUnit
except ImportError:
    # When run as part of a package
    from .difference_processor import EncodedElement, EncodedBatch, EncodingUnit


# ─────────────────────────────────────────────────────────────────────────────
# Single PE
# ─────────────────────────────────────────────────────────────────────────────

class PE:
    """
    A single Processing Element: 4 multipliers + adder tree + shifter + psum reg.

    Paper Fig 12:
      - 4 multipliers (4-bit data × 8-bit weight)
      - shifters on 2 of the 4 lanes (for the high-nibble part of 8-bit data)
      - 3-stage adder tree (4 -> 2 -> 1)
      - 32-bit signed partial-sum register

    We model it functionally: feed it groups of up to 4 EncodedElements per
    "cycle" and it accumulates into self.psum. The cycle granularity matters
    for the Week 2 cycle simulator but not for correctness.
    """

    N_LANES = 4

    def __init__(self):
        self.psum: int = 0
        self.cycles: int = 0

    def reset(self):
        """Clear the partial sum at a layer boundary (paper: clear_psum)."""
        self.psum = 0
        self.cycles = 0

    def cycle(self, elements: list[EncodedElement]) -> int:
        """
        Process up to N_LANES elements in one cycle.

        Each element contributes data × weight to the adder tree, shifted left
        by 4 if metadata == 1 (it is the high nibble of an 8-bit value).

        Returns the sum produced this cycle (also accumulated into self.psum).
        """
        assert len(elements) <= self.N_LANES, (
            f"PE accepts at most {self.N_LANES} elements/cycle, got {len(elements)}"
        )

        lane_products = []
        for el in elements:
            prod = el.data * el.weight          # 4-bit × 8-bit multiply
            if el.metadata == 1:
                prod = prod << 4                # high nibble: shift into place
            lane_products.append(prod)

        cycle_sum = sum(lane_products)          # adder tree
        self.psum += cycle_sum                  # partial-sum accumulator
        self.cycles += 1
        return cycle_sum

    def run_batch(self, batch: EncodedBatch) -> int:
        """
        Run an entire EncodedBatch through this PE, N_LANES elements per cycle.

        Returns the final accumulated psum.
        """
        queue = batch.queue
        for i in range(0, len(queue), self.N_LANES):
            self.cycle(queue[i:i + self.N_LANES])
        return self.psum


# ─────────────────────────────────────────────────────────────────────────────
# Reference: direct integer MAC (the "golden" baseline)
# ─────────────────────────────────────────────────────────────────────────────

def direct_mac(activations, weights) -> int:
    """Direct dot product Σ act_i × W_i, full integer precision (golden ref)."""
    return int(sum(int(a) * int(w) for a, w in zip(activations, weights)))


# ─────────────────────────────────────────────────────────────────────────────
# Self-test: integer-exact equivalence (Step C of the two-step plan)
# ─────────────────────────────────────────────────────────────────────────────

def _self_test():
    """
    Verify the core Ditto equivalence on synthetic int8 data, with NO quantization:

        out_t = out_{t-1} + Σ_i (δ_i × W_i)   should equal   Σ_i (curr_i × W_i)

    This isolates the Encoding Unit + PE logic. If this passes, the shift-and-add
    nibble recombination and the distributive-property identity are both correct.
    """
    import numpy as np
    np.random.seed(7)

    N = 256  # one "row" of a linear layer

    # Synthetic int8 activations for two adjacent time steps.
    # Make them highly similar (small diff) to mimic temporal similarity,
    # but include some large diffs to exercise the 8-bit path.
    prev_act = np.random.randint(-128, 128, N, dtype=np.int32)
    delta = np.zeros(N, dtype=np.int32)
    r = np.random.rand(N)
    delta[r >= 0.5] = np.random.randint(0, 8, (r >= 0.5).sum())       # small (4-bit)
    delta[r >= 0.95] = np.random.randint(-128, -16, (r >= 0.95).sum())  # large (8-bit)
    curr_act = np.clip(prev_act + delta, -128, 127).astype(np.int32)

    weights = np.random.randint(-128, 128, N, dtype=np.int32)

    # --- Golden references ---
    out_prev = direct_mac(prev_act, weights)        # out_{t-1}
    out_curr_direct = direct_mac(curr_act, weights)  # Σ curr_i × W_i  (target)

    # --- Ditto path: encode differences, run through PE, add prev output ---
    eu = EncodingUnit(n_lanes=4)
    batch = eu.process_tensor(
        prev_act.astype(np.int8),
        curr_act.astype(np.int8),
        weights.astype(np.int8),
    )

    pe = PE()
    delta_dot_w = pe.run_batch(batch)               # Σ_i (δ_i × W_i)
    out_curr_ditto = out_prev + delta_dot_w

    # --- Report ---
    print("=== Ditto equivalence self-test (integer-exact, no quant) ===")
    print(f"N elements:            {N}")
    print(f"Zero diffs:            {batch.stats.n_zero}  ({batch.stats.zero_ratio:.1%})")
    print(f"4-bit diffs:           {batch.stats.n_low4}  ({batch.stats.low4_ratio:.1%})")
    print(f"8-bit diffs:           {batch.stats.n_high4}  ({batch.stats.high4_ratio:.1%})")
    print(f"PE cycles:             {pe.cycles}")
    print()
    print(f"out_prev (golden):     {out_prev}")
    print(f"Σ δ×W (PE):            {delta_dot_w}")
    print(f"out_curr Ditto:        {out_curr_ditto}")
    print(f"out_curr direct:       {out_curr_direct}")
    print(f"Difference:            {out_curr_ditto - out_curr_direct}")
    print()

    assert out_curr_ditto == out_curr_direct, (
        f"EQUIVALENCE FAILED: Ditto={out_curr_ditto} vs direct={out_curr_direct} "
        f"(diff={out_curr_ditto - out_curr_direct})"
    )
    print("✓ Ditto equivalence holds (bitwise-exact integer identity).")

    # --- Additional check: per-element δ×W recombination ---
    print("\n=== Per-element nibble recombination check ===")
    mismatches = 0
    for i in range(N):
        d = int(curr_act[i]) - int(prev_act[i])
        if d > 127:
            d -= 256
        elif d < -128:
            d += 256
        # Build a single-element batch and run through PE
        from difference_processor import encode_single_lane, EncodingStats
        elems = encode_single_lane(int(prev_act[i]), int(curr_act[i]), int(weights[i]), EncodingStats())
        pe_single = PE()
        got = pe_single.run_batch(EncodedBatch(queue=elems))
        expected = d * int(weights[i])
        if got != expected:
            mismatches += 1
            if mismatches <= 5:
                print(f"  MISMATCH i={i}: δ={d} W={weights[i]} got={got} expected={expected}")
    if mismatches == 0:
        print(f"✓ All {N} elements: PE(δ-encoded) × W == δ × W")
    else:
        print(f"✗ {mismatches} mismatches")
        raise AssertionError(f"{mismatches} per-element mismatches")


if __name__ == "__main__":
    _self_test()

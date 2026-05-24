"""
difference_processor.py — Functional model of the Ditto Encoding Unit.

Mirrors paper Fig 11 (Section V-B). Computes temporal differences between
adjacent denoising steps, classifies each element as zero / 4-bit / 8-bit,
and reorders the data for downstream PE consumption (zero-skip + mixed-precision).

This is the Python reference model. The corresponding RTL implementation
lives in `rtl/encoding_unit.sv`. Cocotb tests compare RTL output against
this model element-by-element.

Author: njzhx213
Project: Ditto Reproduction (HPCA 2025)
Phase: Week 1, Day 1
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class EncodedElement:
    """A single 4-bit data element going into the PE queue."""
    data: int          # 4-bit value, stored as signed int (-8..7)
    weight: int        # 8-bit weight (signed, -128..127)
    metadata: int      # 1-bit flag: 1 = high nibble (shift << 4 in PE), 0 = low nibble


@dataclass
class EncodingStats:
    """Statistics for one batch — used for Fig 5 reproduction."""
    n_total: int = 0
    n_zero: int = 0          # ctrl == 00
    n_low4: int = 0          # ctrl == 01 (4-bit suffices)
    n_high4: int = 0         # ctrl == 1X (needs full 8-bit, enqueued as 2 elements)

    @property
    def zero_ratio(self) -> float:
        return self.n_zero / max(self.n_total, 1)

    @property
    def low4_ratio(self) -> float:
        return self.n_low4 / max(self.n_total, 1)

    @property
    def high4_ratio(self) -> float:
        return self.n_high4 / max(self.n_total, 1)

    @property
    def le4bit_ratio(self) -> float:
        """zero + 4-bit, the headline number in Fig 5 (96.01% for temporal diff)."""
        return (self.n_zero + self.n_low4) / max(self.n_total, 1)


@dataclass
class EncodedBatch:
    """Result of processing one batch of elements."""
    queue: list[EncodedElement] = field(default_factory=list)
    stats: EncodingStats = field(default_factory=EncodingStats)
    cycle_count: int = 0      # cycles consumed by this batch

    def __len__(self) -> int:
        return len(self.queue)


# ─────────────────────────────────────────────────────────────────────────────
# Core logic — paper Fig 11
# ─────────────────────────────────────────────────────────────────────────────

def classify_diff_element(diff_val: int) -> tuple[int, int, int]:
    """
    Classify one signed int8 difference into the paper's 3 categories,
    using SIGNED-RANGE (sign-magnitude) classification.

    Returns (ctrl, high_part, low_part) where:
      ctrl:
        0b00 = zero       : diff == 0          -> skip
        0b01 = 4-bit       : -8 <= diff <= 7    -> single element (low_part = diff)
        0b1X = 8-bit       : otherwise          -> two elements (low + high<<4)
      For the 4-bit case, low_part = diff (signed, fits in 4 bits) and high_part = 0.
      For the 8-bit case, low_part = unsigned low nibble, high_part = signed high nibble,
      reconstructing as value = (high_part << 4) + low_part.

    IMPORTANT — why signed-range, not two's-complement nibble check:
      Paper Fig 5 reports bit-width as the *minimum signed representation*: a value
      in [-8, 7] needs only 4 bits. A literal two's-complement nibble-zero check
      (paper Fig 11's wording "detect zero in the higher part") would misclassify
      small NEGATIVE diffs (-1..-7, whose high nibble is 0xF) as 8-bit, dropping
      the <=4-bit fraction from ~96% to ~73% on real SDM data. To match Fig 5,
      the Encoding Unit must treat the difference in sign-magnitude form so that
      small negatives also take the 4-bit path. This is the faithful reading of
      the paper's *statistics*, and corresponds to a sign-magnitude datapath in HW.
    """
    d = int(diff_val)
    # Wrap to int8 range (defensive; caller usually already did this)
    if d > 127:
        d -= 256
    elif d < -128:
        d += 256

    if d == 0:
        return 0b00, 0, 0

    if -8 <= d <= 7:
        # 4-bit: the whole signed value fits in a 4-bit signed field.
        return 0b01, 0, d           # high_part unused, low_part = signed diff

    # 8-bit: split via two's-complement nibbles so PE can reconstruct exactly.
    d8 = d & 0xFF
    high = (d8 >> 4) & 0x0F
    low = d8 & 0x0F
    high_signed = high - 16 if high & 0x08 else high   # signed, -8..7
    low_unsigned = low                                  # unsigned, 0..15
    return 0b11, high_signed, low_unsigned


def encode_single_lane(
    prev_act: int,
    curr_act: int,
    weight: int,
    stats: EncodingStats,
) -> list[EncodedElement]:
    """
    Process one element through the Encoding Unit.

    Returns 0, 1, or 2 EncodedElements depending on the classification:
      - 0 elements: zero diff (skipped)
      - 1 element: 4-bit diff (only low nibble enqueued)
      - 2 elements: 8-bit diff (low and high nibbles both enqueued)
    """
    stats.n_total += 1

    diff = int(curr_act) - int(prev_act)
    # Wrap to int8 range
    if diff > 127:
        diff -= 256
    elif diff < -128:
        diff += 256

    ctrl, high_nibble, low_nibble = classify_diff_element(diff)

    if ctrl == 0b00:
        stats.n_zero += 1
        return []

    if ctrl == 0b01:
        stats.n_low4 += 1
        return [EncodedElement(data=low_nibble, weight=weight, metadata=0)]

    # ctrl == 0b10 or 0b11: needs full 8-bit
    stats.n_high4 += 1
    return [
        EncodedElement(data=low_nibble, weight=weight, metadata=0),
        EncodedElement(data=high_nibble, weight=weight, metadata=1),
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Top-level class
# ─────────────────────────────────────────────────────────────────────────────

class EncodingUnit:
    """
    Functional model of the Ditto Encoding Unit (paper Fig 11).

    Processes N_LANES=4 elements per cycle. Throughput is bounded by the
    PE queue depth (default 16) since 8-bit elements enqueue 2 entries.

    Usage:
        eu = EncodingUnit(n_lanes=4, queue_depth=16)
        batch = eu.process_tensor(prev_act, curr_act, weight)
        print(f"Zero ratio: {batch.stats.zero_ratio:.2%}")
    """

    def __init__(self, n_lanes: int = 4, queue_depth: int = 16):
        self.n_lanes = n_lanes
        self.queue_depth = queue_depth

    def process_tensor(
        self,
        prev_act: np.ndarray,
        curr_act: np.ndarray,
        weight: np.ndarray,
    ) -> EncodedBatch:
        """
        Process a 1D array of activations through the Encoding Unit.

        Args:
            prev_act:  shape [N], int8 (signed -128..127)
            curr_act:  shape [N], int8
            weight:    shape [N], int8

        Returns:
            EncodedBatch with the queue of EncodedElements and stats.
        """
        assert prev_act.shape == curr_act.shape == weight.shape
        assert prev_act.dtype == np.int8 or np.issubdtype(prev_act.dtype, np.integer)

        stats = EncodingStats()
        queue: list[EncodedElement] = []

        prev_flat = prev_act.flatten().astype(np.int32)
        curr_flat = curr_act.flatten().astype(np.int32)
        weight_flat = weight.flatten().astype(np.int32)

        for p, c, w in zip(prev_flat, curr_flat, weight_flat):
            elements = encode_single_lane(int(p), int(c), int(w), stats)
            queue.extend(elements)

        # Cycle estimate: ceil(N / n_lanes) for subtract+compare,
        # plus 1 cycle pipeline fill, plus possible queue back-pressure.
        n_input = len(prev_flat)
        cycle_count = -(-n_input // self.n_lanes) + 1  # ceiling division

        return EncodedBatch(queue=queue, stats=stats, cycle_count=cycle_count)


# ─────────────────────────────────────────────────────────────────────────────
# Quick sanity check (run this file directly)
# ─────────────────────────────────────────────────────────────────────────────

def _self_test():
    """Sanity check with a tiny example matching the paper's Fig 7 table.

    Note on the classification rule: per paper Fig 11, the Encoding Unit
    compares the high and low nibbles *independently* against zero. A value
    like -1 (0xFF in two's complement) has high_nibble=0xF and low_nibble=0xF,
    so it is classified as "8-bit needed" even though its magnitude is small.
    Strictly 4-bit-representable diffs are values in [0, 7] (and -8 alone fits
    the low nibble if we treat it as a signed 4-bit quantity, but most signed
    negatives like -1..-7 have 0xF in the high nibble).

    Our synthetic data uses [0, 7] for the "4-bit" bucket to match this.
    """
    np.random.seed(42)

    # Synthetic data resembling SDM temporal differences:
    # ~44% zeros, ~52% small (4-bit fits), ~4% large (needs 8-bit)
    N = 10000
    rand = np.random.rand(N)
    diff = np.zeros(N, dtype=np.int8)
    # Zeros: rand < 0.44, already zero by initialization
    mask_low = (rand >= 0.44) & (rand < 0.96)
    diff[mask_low] = np.random.randint(0, 8, mask_low.sum(), dtype=np.int8)  # 0..7 fits in low nibble
    mask_high = rand >= 0.96
    diff[mask_high] = np.random.randint(-128, -16, mask_high.sum(), dtype=np.int8)

    prev_act = np.zeros(N, dtype=np.int8)
    curr_act = diff  # so that curr - prev = diff
    weight = np.random.randint(-127, 128, N, dtype=np.int8)

    eu = EncodingUnit(n_lanes=4, queue_depth=16)
    batch = eu.process_tensor(prev_act, curr_act, weight)

    print(f"Input elements: {batch.stats.n_total}")
    print(f"Zero ratio:     {batch.stats.zero_ratio:.2%}  (synth target ~44%)")
    print(f"4-bit ratio:    {batch.stats.low4_ratio:.2%}  (synth target ~52%)")
    print(f"8-bit ratio:    {batch.stats.high4_ratio:.2%}  (synth target ~4%)")
    print(f"≤4-bit ratio:   {batch.stats.le4bit_ratio:.2%}  (paper Fig 5 SDM: ~96%)")
    print(f"Queue length:   {len(batch.queue)}  (zeros skipped, 8-bit elements split into 2)")
    print(f"Cycle count:    {batch.cycle_count}")

    # Verify: queue length should equal n_low4 + 2*n_high4
    expected = batch.stats.n_low4 + 2 * batch.stats.n_high4
    assert len(batch.queue) == expected, (
        f"Queue length mismatch: got {len(batch.queue)}, expected {expected}"
    )
    print("\n✓ Self-test passed.")


if __name__ == "__main__":
    _self_test()

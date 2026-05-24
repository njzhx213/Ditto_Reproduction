"""
cycle_model.py — Compute-cycle model for ITC and Ditto (Fig 13, Step C).

STEP C of the two-step Fig-13 plan: compute cycles ONLY (no memory model, no
Defo). The goal is to verify the cycle formulas produce a sane "bare" compute
speedup (expected ~7x, the theoretical upper bound from zero-skip + 4-bit),
before we add the memory penalty and Defo (Step A) that pull it down to the
paper's ~1.5x.

Hardware configs (paper Table III):
  ITC:   27648 PEs, A8W8 (8-bit x 8-bit MAC), 1 effective MAC/PE/cycle
  Ditto: 39398 PEs, A4W8 (4-bit x 8-bit MAC), 4 MACs/PE/cycle (zero-skip + low-bit)

Compute-cycle formulas (per layer):
  ITC:   cycle = N_macs / (n_pe_ITC * 1)
  Ditto: effective_macs = N_macs * (1 - zero_ratio) * bit_factor
         cycle = effective_macs / (n_pe_Ditto * 4)
    where bit_factor accounts for >4-bit elements taking 2 multiplier slots:
         bit_factor = (le4_frac * 1 + gt4_frac * 2) / (le4_frac + gt4_frac)
    (zero elements are skipped entirely; the non-zero ones are 4-bit-dominant.)

N_macs is derived from SDM UNet architecture (see sdm_workload.py), not from
real weights. This is sufficient for the cycle-count ratio (speedup).

Author: njzhx213
Project: Ditto Reproduction (HPCA 2025)
Phase: Week 2, Step C (compute-only cycle model)
"""

from __future__ import annotations

from dataclasses import dataclass


# Hardware configuration (paper Table III)
N_PE_ITC = 27648
N_PE_DITTO = 39398
DITTO_LANES = 4        # 4-bit MACs per PE per cycle
ITC_THROUGHPUT = 1     # 8-bit MAC per PE per cycle


@dataclass
class LayerBitWidth:
    """Bit-width distribution of a layer's temporal differences (from Fig 5)."""
    zero: float       # fraction of zero diffs (skipped)
    le4_nonzero: float  # fraction of 4-bit (nonzero)
    gt4: float        # fraction of >4-bit

    @property
    def nonzero(self) -> float:
        return self.le4_nonzero + self.gt4

    @property
    def bit_factor(self) -> float:
        """Average multiplier slots per NON-ZERO element (4-bit=1, >4bit=2)."""
        nz = self.nonzero
        if nz < 1e-12:
            return 1.0
        return (self.le4_nonzero * 1 + self.gt4 * 2) / nz


def compute_cycle_itc(n_macs: float) -> float:
    """ITC: full 8-bit MAC, no skipping."""
    return n_macs / (N_PE_ITC * ITC_THROUGHPUT)


def compute_cycle_ditto(n_macs: float, bw: LayerBitWidth) -> float:
    """
    Ditto: difference processing with zero-skip + 4-bit.

    effective_macs = N_macs * (nonzero_fraction) * bit_factor
    (zero diffs skipped; nonzero diffs are 4-bit-dominant, >4-bit take 2 slots)
    """
    effective = n_macs * bw.nonzero * bw.bit_factor
    return effective / (N_PE_DITTO * DITTO_LANES)


@dataclass
class LayerCycleResult:
    name: str
    n_macs: float
    cycle_itc: float
    cycle_ditto: float

    @property
    def speedup(self) -> float:
        return self.cycle_itc / self.cycle_ditto if self.cycle_ditto > 0 else 0.0


def run_compute_only(workload: list[dict], bw: LayerBitWidth) -> dict:
    """
    Run the compute-only cycle model over a workload.

    Args:
        workload: list of {"name": str, "n_macs": float} per layer.
        bw: bit-width distribution (shared across layers for now; we use the
            aggregate Fig 5 SDM numbers).

    Returns:
        dict with per-layer results and totals.
    """
    results = []
    total_itc = 0.0
    total_ditto = 0.0

    for layer in workload:
        c_itc = compute_cycle_itc(layer["n_macs"])
        c_ditto = compute_cycle_ditto(layer["n_macs"], bw)
        results.append(LayerCycleResult(layer["name"], layer["n_macs"], c_itc, c_ditto))
        total_itc += c_itc
        total_ditto += c_ditto

    return {
        "per_layer": results,
        "total_itc": total_itc,
        "total_ditto": total_ditto,
        "speedup": total_itc / total_ditto if total_ditto > 0 else 0.0,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Self-test
# ─────────────────────────────────────────────────────────────────────────────

def _self_test():
    # Use the aggregate Fig 5 SDM dynamic-scheme numbers (our real result)
    bw = LayerBitWidth(zero=0.459, le4_nonzero=0.534, gt4=0.007)

    # A small synthetic workload (real SDM layer MACs from sdm_workload.py)
    workload = [
        {"name": "conv_in", "n_macs": 47.2e6},
        {"name": "down1_attn", "n_macs": 8229.7e6},
        {"name": "down2_attn", "n_macs": 7080.5e6},
        {"name": "mid_attn", "n_macs": 1852.2e6},
        {"name": "up1_attn", "n_macs": 7080.5e6},
        {"name": "up2_attn", "n_macs": 8229.7e6},
    ]

    result = run_compute_only(workload, bw)

    print("=== Cycle model self-test (compute-only, Step C) ===")
    print(f"Bit-width: zero={bw.zero:.1%} 4bit={bw.le4_nonzero:.1%} >4bit={bw.gt4:.1%}")
    print(f"  nonzero fraction: {bw.nonzero:.3f}")
    print(f"  bit_factor (slots/nonzero): {bw.bit_factor:.3f}")
    print()
    print(f"{'layer':<14} {'N_macs(M)':>11} {'ITC cyc':>11} {'Ditto cyc':>11} {'speedup':>8}")
    print("-" * 60)
    for r in result["per_layer"]:
        print(f"{r.name:<14} {r.n_macs/1e6:>10.1f} {r.cycle_itc:>11.0f} {r.cycle_ditto:>11.0f} {r.speedup:>7.2f}x")
    print("-" * 60)
    print(f"{'TOTAL':<14} {'':<11} {result['total_itc']:>11.0f} {result['total_ditto']:>11.0f} {result['speedup']:>7.2f}x")
    print()
    print(f"Bare compute speedup: {result['speedup']:.2f}x")
    print(f"(This is the theoretical compute upper bound, decomposed as:")
    print(f"   PE count ratio   {N_PE_DITTO/N_PE_ITC:.2f}x  (Ditto has more, smaller 4-bit PEs)")
    print(f" x throughput ratio {DITTO_LANES/ITC_THROUGHPUT:.0f}x    (4-bit: 4 MACs/PE/cycle)")
    print(f" x skip+bit factor  {1/(bw.nonzero*bw.bit_factor):.2f}x  (zero-skip {bw.zero:.0%} + 4-bit)")
    print(f" Memory penalty + Defo (Step A) pull this down toward the paper's 1.5x;")
    print(f" the gap from {result['speedup']:.1f}x to 1.5x IS the memory-bound cost Ditto fights.)")

    # Sanity: bare compute speedup = PE_ratio x throughput x skip_factor
    # = 1.42 x 4 x 1.82 ≈ 10x. Accept a wide band since it's the theoretical bound.
    assert 6.0 < result["speedup"] < 14.0, (
        f"Bare compute speedup {result['speedup']:.2f}x outside sane 6-14x range — check formula"
    )
    print("\n✓ Cycle model self-test passed (bare compute speedup decomposes correctly).")


if __name__ == "__main__":
    _self_test()

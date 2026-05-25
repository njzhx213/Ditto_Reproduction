"""
test_ditto_top.py - integration test for the full Ditto compute path (ditto_top.v).

Verifies the three core blocks work together correctly:
  1. end-to-end accumulator == numpy dot product of the real differences,
  2. the PE's slot count is self-consistent (slots_total == per-lane 1/2 accounting),
  3. Defo picks the right mode given the PE's actual slot count + realistic layer memory.

Uses realistic layer parameters (n_macs = number of MAC-ops actually streamed, so the
compute/memory balance is meaningful) and checks both a compute-bound layer (small
activation -> DIFF) and a memory-bound layer (large activation -> ACT stop-loss), all
driven through the SAME hardware with the same accumulated slot count.

Run:  cd rtl && make top
"""
import numpy as np
import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer

LANES = 4
DW, WW, AW = 9, 8, 32
LI, LD = 1, 4


def pack(vals, width):
    v = 0
    m = (1 << width) - 1
    for i, x in enumerate(vals):
        v |= (int(x) & m) << (i * width)
    return v


def is_wide(d):
    return 1 if (d > 7 or d < -8) else 0


async def reset(dut):
    dut.rst.value = 1
    dut.valid.value = 0
    dut.diff_vec.value = 0
    dut.w_vec.value = 0
    dut.layer_n_macs.value = 0
    dut.act_bytes.value = 0
    dut.bw.value = 1
    await RisingEdge(dut.clk)
    await RisingEdge(dut.clk)
    dut.rst.value = 0
    await RisingEdge(dut.clk)


async def run_layer(dut, diffs, weights, act_bytes, bw):
    """Stream a layer through the full chain; return (acc, slots_total, mode_diff)."""
    n_groups = len(diffs)
    n_macs = n_groups * LANES                 # MAC-ops if no skip
    dut.layer_n_macs.value = n_macs
    dut.act_bytes.value = act_bytes
    dut.bw.value = bw
    golden = 0
    for g in range(n_groups):
        dut.diff_vec.value = pack(diffs[g], DW)
        dut.w_vec.value = pack(weights[g], WW)
        dut.valid.value = 1
        golden += sum(int(d) * int(w) for d, w in zip(diffs[g], weights[g]))
        await RisingEdge(dut.clk)
    dut.valid.value = 0
    await RisingEdge(dut.clk)
    await Timer(1, unit="ns")
    acc = int(dut.acc.value.to_signed())
    slots = int(dut.slots_total.value)
    mode = int(dut.mode_diff.value)
    return acc, slots, mode, golden, n_macs


@cocotb.test()
async def test_top_end_to_end(dut):
    """Full chain on SDM-like sparse data; acc==numpy, slots consistent."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset(dut)

    rng = np.random.RandomState(0)
    n = 500
    diffs = rng.randint(-254, 255, size=(n, LANES))
    diffs[rng.random((n, LANES)) < 0.46] = 0       # ~46% zero, SDM-like
    weights = rng.randint(-128, 128, size=(n, LANES))

    # compute-bound layer: small activation
    acc, slots, mode, golden, n_macs = await run_layer(
        dut, diffs, weights, act_bytes=50_000, bw=256)

    assert acc == golden, f"end-to-end acc {acc} != numpy {golden}"
    # slot self-consistency
    nz = int((diffs != 0).sum())
    exp_slots = sum(2 if is_wide(int(d)) else 1 for row in diffs for d in row if d != 0)
    assert slots == exp_slots, f"slots {slots} != expected {exp_slots}"
    dut._log.info(f"chain acc == numpy ({golden}); slots_total={slots} "
                  f"(bit_factor {slots/nz:.3f}); zero-skip {(1-nz/diffs.size)*100:.1f}%")
    # compute-bound -> Defo should pick DIFF
    assert mode == 1, f"compute-bound layer should pick DIFF, got mode={mode}"
    dut._log.info(f"Defo: compute-bound layer (act 50KB) -> DIFF (mode={mode})")


@cocotb.test()
async def test_top_defo_stoploss(dut):
    """Same data, but a memory-bound layer -> Defo stop-loss to ACT."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset(dut)

    rng = np.random.RandomState(0)
    n = 500
    diffs = rng.randint(-254, 255, size=(n, LANES))
    diffs[rng.random((n, LANES)) < 0.46] = 0
    weights = rng.randint(-128, 128, size=(n, LANES))

    # memory-bound layer: huge activation, low bandwidth
    acc, slots, mode, golden, n_macs = await run_layer(
        dut, diffs, weights, act_bytes=20_000_000, bw=64)

    assert acc == golden, f"acc {acc} != numpy {golden}"   # arithmetic still exact
    assert mode == 0, f"memory-bound layer should stop-loss to ACT, got mode={mode}"
    dut._log.info(f"Defo: memory-bound layer (act 20MB, bw 64) -> ACT stop-loss "
                  f"(mode={mode}); acc still exact ({golden})")
    dut._log.info("full Ditto path (EU -> slot PE -> Defo) integrated and consistent")

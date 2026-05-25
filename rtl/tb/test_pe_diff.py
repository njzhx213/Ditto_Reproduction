"""
test_pe_diff.py - cocotb testbench for the Ditto difference PE.

Clocked test (the PE has a clock + accumulator, unlike the combinational EU). Drives
random 4-lane (diff, weight) over many cycles and checks the hardware accumulator
equals the naive integer dot product over all lanes and cycles -- the golden
reference. Also checks that zero-skip is LOSSLESS: forcing is_zero on lanes whose
diff is actually 0 must not change the result.

Run:  cd rtl && make pe
"""
import random
import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer

LANES = 4
DW = 9          # diff width (signed)
WW = 8          # weight width (signed)
DMASK = (1 << DW) - 1
WMASK = (1 << WW) - 1


def pack(vals, width):
    v = 0
    m = (1 << width) - 1
    for i, x in enumerate(vals):
        v |= (x & m) << (i * width)
    return v


def pack_zero(diffs):
    """zero-skip flags: set when diff == 0 (lossless skip)."""
    z = 0
    for i, d in enumerate(diffs):
        if d == 0:
            z |= (1 << i)
    return z


async def reset(dut):
    dut.rst.value = 1
    dut.valid.value = 0
    dut.diff_vec.value = 0
    dut.w_vec.value = 0
    dut.is_zero_vec.value = 0
    await RisingEdge(dut.clk)
    await RisingEdge(dut.clk)
    dut.rst.value = 0
    await RisingEdge(dut.clk)


@cocotb.test()
async def test_accumulate(dut):
    """Random 4-lane diff*weight accumulated over 500 cycles; acc == numpy dot."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset(dut)

    random.seed(0)
    golden = 0
    for _ in range(500):
        diffs = [random.randint(-254, 254) for _ in range(LANES)]
        ws = [random.randint(-128, 127) for _ in range(LANES)]
        zflags = pack_zero(diffs)          # skip true zeros (lossless)
        dut.diff_vec.value = pack(diffs, DW)
        dut.w_vec.value = pack(ws, WW)
        dut.is_zero_vec.value = zflags
        dut.valid.value = 1
        # golden: zero lanes contribute 0 anyway
        golden += sum(d * w for d, w in zip(diffs, ws))
        await RisingEdge(dut.clk)

    dut.valid.value = 0
    await RisingEdge(dut.clk)             # let last accumulate land
    await Timer(1, unit="ns")
    assert int(dut.acc.value.to_signed()) == golden, \
        f"acc {int(dut.acc.value.to_signed())} != golden {golden}"
    dut._log.info(f"500-cycle accumulate matches golden dot product: {golden}")


@cocotb.test()
async def test_zero_skip_lossless(dut):
    """Force is_zero on lanes whose diff is genuinely 0; result must be unchanged
    vs computing them. (diff==0 contributes 0 either way -> skip is lossless.)"""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset(dut)

    random.seed(1)
    golden = 0
    n_skipped = 0
    for _ in range(300):
        # deliberately make ~half the lanes zero
        diffs = [0 if random.random() < 0.5 else random.randint(-254, 254)
                 for _ in range(LANES)]
        ws = [random.randint(-128, 127) for _ in range(LANES)]
        zflags = pack_zero(diffs)
        n_skipped += bin(zflags).count("1")
        dut.diff_vec.value = pack(diffs, DW)
        dut.w_vec.value = pack(ws, WW)
        dut.is_zero_vec.value = zflags
        dut.valid.value = 1
        golden += sum(d * w for d, w in zip(diffs, ws))
        await RisingEdge(dut.clk)

    dut.valid.value = 0
    await RisingEdge(dut.clk)
    await Timer(1, unit="ns")
    assert int(dut.acc.value.to_signed()) == golden, \
        f"acc {int(dut.acc.value.to_signed())} != golden {golden}"
    dut._log.info(f"zero-skip lossless: {n_skipped} lanes skipped, acc still == golden")


@cocotb.test()
async def test_reset_and_valid(dut):
    """rst clears acc; valid=0 holds acc (no accumulation)."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset(dut)

    # one accumulate
    dut.diff_vec.value = pack([10, 0, -5, 3], DW)
    dut.w_vec.value = pack([2, 7, 4, -1], WW)
    dut.is_zero_vec.value = pack_zero([10, 0, -5, 3])
    dut.valid.value = 1
    await RisingEdge(dut.clk)
    expect = 10*2 + 0*7 + (-5)*4 + 3*(-1)     # = 20 + 0 - 20 - 3 = -3
    # hold (valid=0): acc must not change
    dut.valid.value = 0
    await RisingEdge(dut.clk)
    await RisingEdge(dut.clk)
    await Timer(1, unit="ns")
    assert int(dut.acc.value.to_signed()) == expect, \
        f"hold failed: acc {int(dut.acc.value.to_signed())} != {expect}"
    # reset clears
    dut.rst.value = 1
    await RisingEdge(dut.clk)
    dut.rst.value = 0
    await Timer(1, unit="ns")
    assert int(dut.acc.value.to_signed()) == 0, "reset did not clear acc"
    dut._log.info(f"reset/valid control correct (held {expect}, then cleared)")

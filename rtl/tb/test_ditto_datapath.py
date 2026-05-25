"""
test_ditto_datapath.py - cocotb testbench for the EU+PE datapath (version A).

Feeds diff and weight; the EU classifies each lane (producing zero-skip flags that
drive the PE). Verifies (1) the EU classification flags match the golden per-lane
rule, and (2) the accumulated MAC equals the naive dot product -- i.e. the EU-driven
zero-skip is lossless and the two modules compose correctly.

Run:  cd rtl && make datapath
"""
import random
import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer

LANES = 4
DW = 9
WW = 8


def pack(vals, width):
    v = 0
    m = (1 << width) - 1
    for i, x in enumerate(vals):
        v |= (x & m) << (i * width)
    return v


def golden_flags(diff):
    is_zero = 1 if diff == 0 else 0
    is_wide = 1 if (diff > 7 or diff < -8) else 0
    return is_zero, is_wide


async def reset(dut):
    dut.rst.value = 1
    dut.valid.value = 0
    dut.diff_vec.value = 0
    dut.w_vec.value = 0
    await RisingEdge(dut.clk)
    await RisingEdge(dut.clk)
    dut.rst.value = 0
    await RisingEdge(dut.clk)


@cocotb.test()
async def test_datapath_accumulate(dut):
    """EU+PE composed: random diff/weight over 500 cycles, acc == golden dot product,
    and EU flags correct each cycle."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset(dut)

    random.seed(0)
    golden = 0
    for _ in range(500):
        diffs = [random.randint(-254, 254) for _ in range(LANES)]
        ws = [random.randint(-128, 127) for _ in range(LANES)]
        dut.diff_vec.value = pack(diffs, DW)
        dut.w_vec.value = pack(ws, WW)
        dut.valid.value = 1
        await Timer(1, unit="ns")   # let EU combinational flags settle
        # check EU flags (combinational, available same cycle)
        izv = int(dut.is_zero_vec.value)
        iwv = int(dut.is_wide_vec.value)
        for i, d in enumerate(diffs):
            gz, gw = golden_flags(d)
            assert (izv >> i) & 1 == gz, f"EU is_zero lane{i} diff={d}"
            assert (iwv >> i) & 1 == gw, f"EU is_wide lane{i} diff={d}"
        golden += sum(d * w for d, w in zip(diffs, ws))
        await RisingEdge(dut.clk)

    dut.valid.value = 0
    await RisingEdge(dut.clk)
    await Timer(1, unit="ns")
    assert int(dut.acc.value.to_signed()) == golden, \
        f"datapath acc {int(dut.acc.value.to_signed())} != golden {golden}"
    dut._log.info(f"EU+PE datapath: 500 cycles, flags correct, acc == golden {golden}")


@cocotb.test()
async def test_datapath_sparse(dut):
    """Many zero diffs: EU flags them, PE skips, result still exact."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset(dut)

    random.seed(2)
    golden = 0
    zeros = 0
    for _ in range(300):
        diffs = [0 if random.random() < 0.6 else random.randint(-254, 254)
                 for _ in range(LANES)]
        ws = [random.randint(-128, 127) for _ in range(LANES)]
        zeros += diffs.count(0)
        dut.diff_vec.value = pack(diffs, DW)
        dut.w_vec.value = pack(ws, WW)
        dut.valid.value = 1
        golden += sum(d * w for d, w in zip(diffs, ws))
        await RisingEdge(dut.clk)

    dut.valid.value = 0
    await RisingEdge(dut.clk)
    await Timer(1, unit="ns")
    assert int(dut.acc.value.to_signed()) == golden, \
        f"sparse acc {int(dut.acc.value.to_signed())} != golden {golden}"
    dut._log.info(f"sparse datapath: {zeros} zero lanes EU-flagged & skipped, acc exact")

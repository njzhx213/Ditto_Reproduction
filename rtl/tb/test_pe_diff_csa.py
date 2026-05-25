"""
test_pe_diff_csa.py - cocotb testbench for the carry-save accumulator PE.

Verifies the resolved accumulator (acc_s + acc_c) equals the numpy dot product, i.e.
carry-save accumulation is functionally identical to the carry-propagate version. The
point of this module is the SYNTHESIS Fmax (it should beat the ~600 MHz the
carry-propagate PEs hit, confirming the accumulator carry chain was the bottleneck);
functional correctness here just guarantees the optimization is lossless.

Run:  cd rtl && make csa
"""
import random
import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer

LANES = 4
DW, WW = 9, 8


def pack(vals, width):
    v = 0
    m = (1 << width) - 1
    for i, x in enumerate(vals):
        v |= (int(x) & m) << (i * width)
    return v


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
async def test_csa_accumulate(dut):
    """Resolved acc == numpy dot product over 500 cycles (CSA is lossless)."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset(dut)

    random.seed(0)
    golden = 0
    for _ in range(500):
        diffs = [random.randint(-254, 254) for _ in range(LANES)]
        ws = [random.randint(-128, 127) for _ in range(LANES)]
        zf = 0
        for i, d in enumerate(diffs):
            if d == 0:
                zf |= (1 << i)
        dut.diff_vec.value = pack(diffs, DW)
        dut.w_vec.value = pack(ws, WW)
        dut.is_zero_vec.value = zf
        dut.valid.value = 1
        golden += sum(d * w for d, w in zip(diffs, ws))
        await RisingEdge(dut.clk)

    dut.valid.value = 0
    await RisingEdge(dut.clk)
    await Timer(1, unit="ns")
    got = int(dut.acc.value.to_signed())
    assert got == golden, f"CSA resolved acc {got} != numpy {golden}"
    dut._log.info(f"carry-save accumulator: resolved acc == numpy dot product ({golden}) "
                  f"-- functionally identical to carry-propagate PE")


@cocotb.test()
async def test_csa_zero_skip(dut):
    """Zero-skip still lossless through the carry-save accumulator."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset(dut)

    random.seed(2)
    golden = 0
    for _ in range(300):
        diffs = [0 if random.random() < 0.5 else random.randint(-254, 254)
                 for _ in range(LANES)]
        ws = [random.randint(-128, 127) for _ in range(LANES)]
        zf = 0
        for i, d in enumerate(diffs):
            if d == 0:
                zf |= (1 << i)
        dut.diff_vec.value = pack(diffs, DW)
        dut.w_vec.value = pack(ws, WW)
        dut.is_zero_vec.value = zf
        dut.valid.value = 1
        golden += sum(d * w for d, w in zip(diffs, ws))
        await RisingEdge(dut.clk)

    dut.valid.value = 0
    await RisingEdge(dut.clk)
    await Timer(1, unit="ns")
    assert int(dut.acc.value.to_signed()) == golden
    dut._log.info(f"carry-save + zero-skip lossless: acc == numpy ({golden})")

"""
test_ditto_datapath_b.py - cocotb testbench for datapath B (EU sign-mag -> PE).

The PE now consumes the EU's sign+magnitude encoding (not the raw diff). Verifies the
sign-magnitude MAC still equals the naive integer dot product over all lanes/cycles --
proving the encoded datapath is arithmetically identical to direct multiply, and that
zero-skip remains lossless. Same golden reference as version A, so B's result must
match A's exactly.

Run:  cd rtl && make datapath_b
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
async def test_b_signmag_accumulate(dut):
    """Random diff/weight; EU encodes to sign-mag, PE_B does sign-magnitude MAC;
    acc must equal naive dot product (identical to version A)."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset(dut)

    random.seed(0)   # same seed as A -> expect the same final acc (-55652)
    golden = 0
    for _ in range(500):
        diffs = [random.randint(-254, 254) for _ in range(LANES)]
        ws = [random.randint(-128, 127) for _ in range(LANES)]
        dut.diff_vec.value = pack(diffs, DW)
        dut.w_vec.value = pack(ws, WW)
        dut.valid.value = 1
        golden += sum(d * w for d, w in zip(diffs, ws))
        await RisingEdge(dut.clk)

    dut.valid.value = 0
    await RisingEdge(dut.clk)
    await Timer(1, unit="ns")
    got = int(dut.acc.value.to_signed())
    assert got == golden, f"B acc {got} != golden {golden}"
    dut._log.info(f"datapath B (sign-mag MAC): acc == golden {golden} "
                  f"(matches version A, confirming encoding is lossless)")


@cocotb.test()
async def test_b_sign_correctness(dut):
    """Targeted sign cases: negative diffs must subtract correctly via sign-mag."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset(dut)

    cases = [
        ([-254, 254, -1, 1], [127, 127, -128, -128]),  # large +/- magnitudes
        ([-100, 50, -7, 8], [-3, 9, 100, -50]),         # mixed signs both sides
        ([0, -8, 7, -9], [5, 5, 5, 5]),                  # boundary + zero
    ]
    golden = 0
    for diffs, ws in cases:
        dut.diff_vec.value = pack(diffs, DW)
        dut.w_vec.value = pack(ws, WW)
        dut.valid.value = 1
        golden += sum(d * w for d, w in zip(diffs, ws))
        await RisingEdge(dut.clk)
    dut.valid.value = 0
    await RisingEdge(dut.clk)
    await Timer(1, unit="ns")
    got = int(dut.acc.value.to_signed())
    assert got == golden, f"sign-case acc {got} != golden {golden}"
    dut._log.info(f"sign-magnitude correctness: signed cases acc == golden {golden}")

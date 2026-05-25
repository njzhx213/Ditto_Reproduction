"""
test_encoding_unit_x4.py - cocotb testbench for the 4-lane parallel Encoding Unit.

Drives 4 packed signed diffs per cycle and checks each lane's outputs against the
same golden classification used for the single-element unit. Confirms the parallel
wrapper preserves per-lane correctness (it should: it just instantiates 4 verified
encoding_units).

Run:  cd rtl && make MODULE=test_encoding_unit_x4 TOPLEVEL=encoding_unit_x4 \
            VERILOG_SOURCES="$(pwd)/common/encoding_unit.v $(pwd)/common/encoding_unit_x4.v"
(or via the x4 target in the Makefile)
"""
import random
import cocotb
from cocotb.triggers import Timer

DIFF_WIDTH = 9
LANES = 4
MASK = (1 << DIFF_WIDTH) - 1


def golden(diff):
    is_zero = 1 if diff == 0 else 0
    is_wide = 1 if (diff > 7 or diff < -8) else 0
    sign = 1 if diff < 0 else 0
    mag = -diff if diff < 0 else diff
    return is_zero, is_wide, sign, mag


def pack(diffs):
    """Pack 4 signed diffs into the LANES*DIFF_WIDTH bus (two's complement per lane)."""
    val = 0
    for i, d in enumerate(diffs):
        val |= (d & MASK) << (i * DIFF_WIDTH)
    return val


async def check_vec(dut, diffs):
    dut.diff_vec.value = pack(diffs)
    await Timer(1, unit="ns")
    izv = int(dut.is_zero_vec.value)
    iwv = int(dut.is_wide_vec.value)
    sv = int(dut.sign_vec.value)
    mv = int(dut.mag_vec.value)
    for i, d in enumerate(diffs):
        gz, gw, gs, gm = golden(d)
        assert (izv >> i) & 1 == gz, f"lane{i} is_zero diff={d}"
        assert (iwv >> i) & 1 == gw, f"lane{i} is_wide diff={d}"
        assert (sv >> i) & 1 == gs, f"lane{i} sign diff={d}"
        lane_mag = (mv >> (i * DIFF_WIDTH)) & MASK
        assert lane_mag == gm, f"lane{i} mag diff={d} got {lane_mag} want {gm}"


@cocotb.test()
async def test_x4_boundaries(dut):
    """One vector hitting all four classes at once: zero / 4-bit / wide-/ wide+."""
    await check_vec(dut, [0, 5, -9, 100])      # zero, 4-bit, wide(neg), wide(pos)
    await check_vec(dut, [-8, 7, 8, -254])     # 4-bit edge, 4-bit edge, wide, wide
    dut._log.info("x4 boundary vectors match golden on all lanes")


@cocotb.test()
async def test_x4_random(dut):
    """Random 4-wide vectors."""
    random.seed(0)
    for _ in range(2000):
        await check_vec(dut, [random.randint(-254, 254) for _ in range(LANES)])
    dut._log.info("2000 random 4-lane vectors match golden")

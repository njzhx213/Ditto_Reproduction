"""
test_encoding_unit.py - cocotb testbench for the Ditto Encoding Unit.

Verifies encoding_unit.v against the SAME per-element classification used in the
performance model (classify_signed_range), which is the project's bitwise-verified
golden reference. This realizes the "RTL track reuses Functional Ditto as reference"
design: the RTL's correctness is defined by matching the validated Python model.

Run:  cd rtl && make
"""
import random
import cocotb
from cocotb.triggers import Timer


def golden(diff):
    """Per-element golden reference (matches classify_signed_range in the perf model).
    Returns (is_zero, is_wide_gt4, sign, magnitude)."""
    is_zero = 1 if diff == 0 else 0
    is_wide = 1 if (diff > 7 or diff < -8) else 0     # >4-bit (outside signed [-8,7])
    sign = 1 if diff < 0 else 0
    mag = -diff if diff < 0 else diff
    return is_zero, is_wide, sign, mag


async def check_one(dut, diff):
    # drive signed diff (cocotb handles two's complement for signed ports via int)
    dut.diff.value = diff
    await Timer(1, unit="ns")
    gz, gw, gs, gm = golden(diff)
    assert int(dut.is_zero.value) == gz, f"is_zero mismatch diff={diff}"
    assert int(dut.is_wide.value) == gw, f"is_wide mismatch diff={diff}"
    assert int(dut.sign.value) == gs, f"sign mismatch diff={diff}"
    assert int(dut.magnitude.value) == gm, f"magnitude mismatch diff={diff} got {int(dut.magnitude.value)} want {gm}"


@cocotb.test()
async def test_full_range(dut):
    """Exhaustive sweep over every possible diff value [-254, 254]."""
    n = 0
    for diff in range(-254, 255):
        await check_one(dut, diff)
        n += 1
    dut._log.info(f"exhaustive: {n} diff values match golden, 0 mismatches")


@cocotb.test()
async def test_boundaries(dut):
    """Spot-check the signed-4-bit classification boundaries."""
    for diff in [-9, -8, -1, 0, 1, 7, 8]:
        await check_one(dut, diff)
    dut._log.info("boundary cases (-9/-8/0/7/8) match golden")


@cocotb.test()
async def test_random(dut):
    """Random fuzz, mimicking real quantized temporal differences."""
    random.seed(0)
    for _ in range(2000):
        await check_one(dut, random.randint(-254, 254))
    dut._log.info("2000 random diffs match golden")

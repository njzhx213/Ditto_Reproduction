"""
test_vpu_restore.py - cocotb testbench for the VPU restore unit.

Two checks:
  1. restore accumulation: act_curr = running sum of diff results (act_prev + diff).
  2. identity loop: feeding restore the diffs produced by diff_generator's encode
     (diff = curr - prev) reconstructs the original activations exactly --
     restore(diff_generator(act)) == act. This proves Ditto's difference encode/decode
     is lossless, the basis for difference-domain compute on linear layers.

Run:  cd rtl && make vpu
"""
import numpy as np
import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer

LANES = 4
DW, AW = 9, 16


def pack(vals, width):
    v = 0
    m = (1 << width) - 1
    for i, x in enumerate(vals):
        v |= (int(x) & m) << (i * width)
    return v


def unpack_signed(packed, idx, width):
    raw = (packed >> (idx * width)) & ((1 << width) - 1)
    return raw - (1 << width) if raw >= (1 << (width - 1)) else raw


def diff_generator(curr_stream):
    """Software encode: diff = curr - prev (prev starts 0). Same as diff_generator.v."""
    prev = [0] * LANES
    out = []
    for curr in curr_stream:
        out.append([curr[l] - prev[l] for l in range(LANES)])
        prev = curr[:]
    return out


async def reset(dut):
    dut.rst.value = 1
    dut.valid.value = 0
    dut.diff_result.value = 0
    await RisingEdge(dut.clk)
    await RisingEdge(dut.clk)
    dut.rst.value = 0
    await RisingEdge(dut.clk)


@cocotb.test()
async def test_restore_accumulate(dut):
    """act_curr is the running sum of the diff results fed in."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset(dut)

    rng = np.random.RandomState(1)
    acc = [0] * LANES
    for _ in range(20):
        diffs = rng.randint(-50, 51, size=LANES).tolist()
        dut.diff_result.value = pack(diffs, DW)
        dut.valid.value = 1
        await RisingEdge(dut.clk)
        await Timer(1, unit="ns")
        acc = [acc[l] + diffs[l] for l in range(LANES)]
        got = [unpack_signed(int(dut.act_curr.value), l, AW) for l in range(LANES)]
        assert got == acc, f"restore acc {got} != {acc}"
    dut._log.info("restore accumulation correct over 20 steps")


@cocotb.test()
async def test_identity_loop(dut):
    """restore(diff_generator(act)) == act -- lossless difference encode/decode."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset(dut)

    rng = np.random.RandomState(0)
    steps = [rng.randint(-127, 128, size=LANES).tolist() for _ in range(8)]
    diffs = diff_generator(steps)      # software encode (matches diff_generator.v)

    for s in range(len(steps)):
        dut.diff_result.value = pack(diffs[s], DW)
        dut.valid.value = 1
        await RisingEdge(dut.clk)
        await Timer(1, unit="ns")
        got = [unpack_signed(int(dut.act_curr.value), l, AW) for l in range(LANES)]
        assert got == steps[s], f"step {s}: restored {got} != original {steps[s]}"
    dut._log.info("identity loop confirmed: restore(diff_generator(act)) == act "
                  "(difference encode/decode is lossless)")

"""
test_diff_generator.py - cocotb testbench for the Ditto datapath entry.

Streams quantized activations over several denoising steps and checks each output
difference equals curr - prev (with prev = 0 on the first step, flagged by first_step).
Confirms the prev-frame register updates correctly (this step's curr -> next step's
prev). This is the front end that turns activations into the differences the EU
consumes.

Run:  cd rtl && make diffgen
"""
import numpy as np
import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer

LANES = 4
IW, DW = 8, 9
DMASK = (1 << DW) - 1


def pack(vals, width):
    v = 0
    m = (1 << width) - 1
    for i, x in enumerate(vals):
        v |= (int(x) & m) << (i * width)
    return v


def unpack_signed(packed, idx, width):
    raw = (packed >> (idx * width)) & ((1 << width) - 1)
    return raw - (1 << width) if raw >= (1 << (width - 1)) else raw


async def reset(dut):
    dut.rst.value = 1
    dut.valid.value = 0
    dut.curr_vec.value = 0
    await RisingEdge(dut.clk)
    await RisingEdge(dut.clk)
    dut.rst.value = 0
    await RisingEdge(dut.clk)


@cocotb.test()
async def test_diff_stream(dut):
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset(dut)

    rng = np.random.RandomState(0)
    n_steps = 8
    steps = [rng.randint(-127, 128, size=LANES).tolist() for _ in range(n_steps)]

    prev = [0] * LANES
    for s in range(n_steps):
        dut.curr_vec.value = pack(steps[s], IW)
        dut.valid.value = 1
        await RisingEdge(dut.clk)
        await Timer(1, unit="ns")
        # output reflects this cycle's curr vs the prev register
        exp_diff = [steps[s][l] - prev[l] for l in range(LANES)]
        got = [unpack_signed(int(dut.diff_vec.value), l, DW) for l in range(LANES)]
        assert got == exp_diff, f"step {s}: diff {got} != {exp_diff}"
        exp_first = 1 if s == 0 else 0
        assert int(dut.first_step.value) == exp_first, f"step {s} first_step"
        # range check
        for d in got:
            assert -254 <= d <= 254, f"diff {d} out of range"
        prev = steps[s][:]    # update reference

    dut._log.info(f"{n_steps}-step activation stream: all diffs == curr-prev, "
                  f"first_step correct, range OK")
    dut._log.info("prev-frame register tracks the previous step (origin of Defo's "
                  "prev-frame cost)")


@cocotb.test()
async def test_first_step_is_curr(dut):
    """First step after reset: prev=0 so diff == curr (the reference frame)."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset(dut)
    curr = [50, -30, 7, -8]
    dut.curr_vec.value = pack(curr, IW)
    dut.valid.value = 1
    await RisingEdge(dut.clk)
    await Timer(1, unit="ns")
    got = [unpack_signed(int(dut.diff_vec.value), l, DW) for l in range(LANES)]
    assert got == curr, f"first-step diff {got} != curr {curr}"
    assert int(dut.first_step.value) == 1
    dut._log.info(f"first step: diff == curr {curr} (prev=0 baseline), first_step=1")

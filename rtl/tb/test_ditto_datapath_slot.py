"""
test_ditto_datapath_slot.py - cocotb testbench for the slot-based Ditto datapath.

Two things verified:
  (1) ARITHMETIC: the 4-bit/>4-bit slot-split multiply (1 slot for signed-4-bit lanes,
      2 nibble-split slots for wide lanes) still equals the naive integer dot product.
  (2) SLOT ACCOUNTING: the average multiplier slots per nonzero lane, measured by the
      hardware's slot counter, matches the performance model's bit_factor
      (= (le4*1 + gt4*2)/nonzero). This is the RTL <-> perf-model quantitative tie-in:
      the same bit-width statistic the speedup model assumes is what the gate-level
      slot counter actually consumes.

Run:  cd rtl && make datapath_slot
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


def is_wide(d):
    return 1 if (d > 7 or d < -8) else 0    # signed 4-bit boundary, same as EU


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
async def test_slot_arithmetic(dut):
    """Slot-split multiply must equal naive dot product (lossless micro-arch)."""
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
        golden += sum(d * w for d, w in zip(diffs, ws))
        await RisingEdge(dut.clk)

    dut.valid.value = 0
    await RisingEdge(dut.clk)
    await Timer(1, unit="ns")
    got = int(dut.acc.value.to_signed())
    assert got == golden, f"slot acc {got} != golden {golden}"
    dut._log.info(f"slot-split MAC: acc == golden {golden} (matches A/B; 4-bit/wide split lossless)")


@cocotb.test()
async def test_slot_count_matches_bitfactor(dut):
    """Average slots/nonzero-lane (HW counter) == perf-model bit_factor."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset(dut)

    random.seed(3)
    n_cycles = 4000
    nonzero = 0
    py_slots = 0
    for _ in range(n_cycles):
        diffs = [random.randint(-254, 254) for _ in range(LANES)]
        ws = [random.randint(-128, 127) for _ in range(LANES)]
        for d in diffs:
            if d != 0:
                nonzero += 1
                py_slots += 2 if is_wide(d) else 1
        dut.diff_vec.value = pack(diffs, DW)
        dut.w_vec.value = pack(ws, WW)
        dut.valid.value = 1
        await RisingEdge(dut.clk)

    dut.valid.value = 0
    await RisingEdge(dut.clk)
    await Timer(1, unit="ns")

    hw_total = int(dut.slots_total.value)
    hw_avg = hw_total / nonzero
    py_avg = py_slots / nonzero
    # perf-model bit_factor for uniform diff over signed 4-bit boundary
    le4 = 15 / 509          # nonzero values in [-8,7]: -8..-1,1..7 = 15
    gt4 = (509 - 1 - 15) / 509
    bit_factor = (le4 * 1 + gt4 * 2) / (le4 + gt4)

    dut._log.info(f"HW slots_total={hw_total}, nonzero={nonzero}")
    dut._log.info(f"HW avg slots/nonzero = {hw_avg:.4f}")
    dut._log.info(f"Python golden avg     = {py_avg:.4f}")
    dut._log.info(f"perf-model bit_factor = {bit_factor:.4f}")
    # HW counter must exactly match the python slot accounting
    assert abs(hw_avg - py_avg) < 1e-9, f"HW {hw_avg} != py {py_avg}"
    # and both should be close to the analytic bit_factor (uniform sampling)
    assert abs(hw_avg - bit_factor) < 0.05, f"HW avg {hw_avg} far from bit_factor {bit_factor}"
    dut._log.info("HW slot counter == perf-model bit_factor: RTL <-> perf-model tie-in confirmed")


@cocotb.test()
async def test_slot_boundary(dut):
    """Boundary lanes: diff=7 -> 1 slot, diff=8 -> 2 slots, diff=-8 -> 1, diff=-9 -> 2."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset(dut)
    # one cycle with [7, 8, -8, -9] -> slots 1+2+1+2 = 6
    diffs = [7, 8, -8, -9]
    ws = [3, 3, 3, 3]
    dut.diff_vec.value = pack(diffs, DW)
    dut.w_vec.value = pack(ws, WW)
    dut.valid.value = 1
    await RisingEdge(dut.clk)
    dut.valid.value = 0
    await Timer(1, unit="ns")
    assert int(dut.slots_this_cycle.value) == 6, \
        f"boundary slots {int(dut.slots_this_cycle.value)} != 6"
    expect = sum(d * w for d, w in zip(diffs, ws))
    assert int(dut.acc.value.to_signed()) == expect, "boundary acc wrong"
    dut._log.info("boundary: [7,8,-8,-9] -> 1+2+1+2 = 6 slots, acc correct")

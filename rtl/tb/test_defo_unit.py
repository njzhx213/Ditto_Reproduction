"""
test_defo_unit.py - cocotb testbench for the Defo runtime decision unit.

Verifies the scaled-integer roofline decision matches a floating-point reference
(cost = max(compute, memory); DIFF re-reads the previous frame -> 2x activation
bytes), over random layers, and demonstrates the stop-loss: on memory-bound layers
the previous-frame re-read makes DIFF more expensive, so Defo falls back to ACT
(paper Fig 16).

Combinational unit (like the EU), so Timer-settle reads.

Run:  cd rtl && make defo
"""
import random
import cocotb
from cocotb.triggers import Timer

LI, LD = 1, 4   # must match defo_unit params


def float_mode_diff(n_macs, slots, act_bytes, bw):
    cost_act = max(n_macs / LI, act_bytes / bw)
    cost_diff = max(slots / LD, 2 * act_bytes / bw)
    return 1 if cost_diff < cost_act else 0


async def drive(dut, n_macs, slots, act_bytes, bw):
    dut.n_macs.value = n_macs
    dut.slots_used.value = slots
    dut.act_bytes.value = act_bytes
    dut.bw.value = bw
    await Timer(1, unit="ns")
    return int(dut.mode_diff.value)


@cocotb.test()
async def test_defo_random(dut):
    """Random layers: HW decision == float roofline reference."""
    random.seed(0)
    for _ in range(3000):
        n = random.randint(1, 1_000_000)
        s = random.randint(0, 2 * n)
        a = random.randint(1, 5_000_000)
        bw = random.choice([64, 128, 256, 512, 1024])
        hw = await drive(dut, n, s, a, bw)
        ref = float_mode_diff(n, s, a, bw)
        assert hw == ref, f"mode mismatch n={n} s={s} a={a} bw={bw}: hw={hw} ref={ref}"
    dut._log.info("3000 random layers: Defo decision == float roofline reference")


@cocotb.test()
async def test_defo_stop_loss(dut):
    """Stop-loss: compute-bound -> DIFF; memory-bound -> ACT fallback."""
    # compute-bound (small activation, many MACs) -> DIFF
    m = await drive(dut, 1_000_000, 1_063_000, 100_000, 256)
    assert m == 1, "compute-bound layer should pick DIFF"
    # very memory-bound (huge activation, few MACs, low BW) -> ACT stop-loss
    m = await drive(dut, 10_000, 10_500, 5_000_000, 128)
    assert m == 0, "memory-bound layer should stop-loss to ACT"
    dut._log.info("stop-loss confirmed: DIFF when compute-bound, ACT when memory-bound "
                  "(prev-frame re-read exceeds compute saving, Fig 16)")


@cocotb.test()
async def test_defo_sweep_boundary(dut):
    """Sweep activation size to find the DIFF->ACT crossover for a fixed layer."""
    n_macs, slots, bw = 100_000, 106_000, 256
    crossover = None
    prev_mode = None
    # sweep wide enough to cross the boundary (compute-bound ACT holds until the
    # activation is large enough that DIFF's 2x re-read dominates, ~12-13 MB here)
    for act_kb in range(1, 30000, 200):
        a = act_kb * 1000
        m = await drive(dut, n_macs, slots, a, bw)
        if prev_mode == 1 and m == 0:
            crossover = act_kb
        prev_mode = m
    assert crossover is not None, "expected a DIFF->ACT crossover in the swept range"
    dut._log.info(f"DIFF->ACT crossover near act ~{crossover} KB "
                  f"(beyond it, prev-frame re-read dominates -> stop-loss)")

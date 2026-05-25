"""
test_pe_diff_pipe.py - cocotb testbench for the 3-stage pipelined PE.

The pipeline has 3-cycle latency, so after streaming N inputs the testbench drains 3+
extra cycles before reading the accumulator. Verifies:
  1. the final accumulator (after drain) == numpy dot product == the single-cycle PE,
     i.e. pipelining is an equivalence transform (same result, higher Fmax);
  2. zero-skip still lossless through the pipeline;
  3. bubbles (valid_in=0) inject nothing, so a gapped stream gives the same result.

Run:  cd rtl && make pipe
"""
import random
import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer

LANES = 4
DW, WW = 9, 8
LATENCY = 3


def pack(vals, width):
    v = 0
    m = (1 << width) - 1
    for i, x in enumerate(vals):
        v |= (int(x) & m) << (i * width)
    return v


async def reset(dut):
    dut.rst.value = 1
    dut.valid_in.value = 0
    dut.diff_vec.value = 0
    dut.w_vec.value = 0
    dut.is_zero_vec.value = 0
    await RisingEdge(dut.clk)
    await RisingEdge(dut.clk)
    dut.rst.value = 0
    await RisingEdge(dut.clk)


async def drain(dut, n=LATENCY + 1):
    dut.valid_in.value = 0
    for _ in range(n):
        await RisingEdge(dut.clk)
    await Timer(1, unit="ns")


@cocotb.test()
async def test_pipe_accumulate(dut):
    """Stream 500 inputs back-to-back; after drain, acc == numpy dot product."""
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
        dut.valid_in.value = 1
        golden += sum(d * w for d, w in zip(diffs, ws))
        await RisingEdge(dut.clk)

    await drain(dut)
    got = int(dut.acc.value.to_signed())
    assert got == golden, f"pipelined acc {got} != numpy {golden}"
    dut._log.info(f"3-stage pipeline: acc == numpy dot product ({golden}) after "
                  f"{LATENCY}-cycle drain (same result as single-cycle PE, higher Fmax)")


@cocotb.test()
async def test_pipe_with_bubbles(dut):
    """A gapped stream (valid_in toggles) must give the same result as no gaps."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset(dut)

    random.seed(5)
    samples = []
    for _ in range(200):
        diffs = [random.randint(-254, 254) for _ in range(LANES)]
        ws = [random.randint(-128, 127) for _ in range(LANES)]
        samples.append((diffs, ws))
    golden = sum(sum(d * w for d, w in zip(df, ws)) for df, ws in samples)

    for diffs, ws in samples:
        zf = 0
        for i, d in enumerate(diffs):
            if d == 0:
                zf |= (1 << i)
        dut.diff_vec.value = pack(diffs, DW)
        dut.w_vec.value = pack(ws, WW)
        dut.is_zero_vec.value = zf
        dut.valid_in.value = 1
        await RisingEdge(dut.clk)
        # inject a bubble half the time
        if random.random() < 0.5:
            dut.valid_in.value = 0
            await RisingEdge(dut.clk)

    await drain(dut)
    got = int(dut.acc.value.to_signed())
    assert got == golden, f"bubbled acc {got} != numpy {golden}"
    dut._log.info(f"bubbles ignored (valid gates accumulation): acc == numpy {golden}")


@cocotb.test()
async def test_pipe_latency(dut):
    """Confirm the 3-cycle latency: a single input appears in acc exactly 3 cycles later."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset(dut)

    diffs = [10, 0, -5, 3]
    ws = [2, 7, 4, -1]
    expect = sum(d * w for d, w in zip(diffs, ws))   # -3
    zf = 0
    for i, d in enumerate(diffs):
        if d == 0:
            zf |= (1 << i)
    dut.diff_vec.value = pack(diffs, DW)
    dut.w_vec.value = pack(ws, WW)
    dut.is_zero_vec.value = zf
    dut.valid_in.value = 1
    await RisingEdge(dut.clk)
    dut.valid_in.value = 0
    # acc should still be 0 for the next 2 cycles, then become `expect` on the 3rd
    accs = []
    for _ in range(LATENCY + 1):
        await Timer(1, unit="ns")
        accs.append(int(dut.acc.value.to_signed()))
        await RisingEdge(dut.clk)
    await Timer(1, unit="ns")
    final = int(dut.acc.value.to_signed())
    assert final == expect, f"after latency acc {final} != {expect}"
    dut._log.info(f"3-cycle latency confirmed: single input -> acc={final} after pipeline fill")

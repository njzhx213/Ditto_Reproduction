"""
test_pe_array_parallel.py - cocotb testbench for the 4x4 parallel PE array.

Computes C = diff[4x8] @ weight[8x4] by streaming K=8 (diff-column, weight-row) pairs,
and checks all 16 accumulators equal numpy's matmul. Also reports the zero-skip count.
This array's result is the golden reference that the systolic version (B) must match.

Run:  cd rtl && make array
"""
import numpy as np
import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer

M, K, N = 4, 8, 4
DW, WW, AW = 9, 8, 32


def pack(vals, width):
    v = 0
    m = (1 << width) - 1
    for i, x in enumerate(vals):
        v |= (int(x) & m) << (i * width)
    return v


def unpack_acc(flat_int, idx):
    raw = (flat_int >> (idx * AW)) & ((1 << AW) - 1)
    return raw - (1 << AW) if raw >= (1 << (AW - 1)) else raw


async def reset(dut):
    dut.rst.value = 1
    dut.valid.value = 0
    dut.diff_col.value = 0
    dut.w_row.value = 0
    await RisingEdge(dut.clk)
    await RisingEdge(dut.clk)
    dut.rst.value = 0
    await RisingEdge(dut.clk)


@cocotb.test()
async def test_array_matmul(dut):
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset(dut)

    rng = np.random.RandomState(0)
    diff = rng.randint(-254, 255, size=(M, K))
    diff[rng.random((M, K)) < 0.4] = 0          # ~40% zeros
    weight = rng.randint(-128, 128, size=(K, N))
    golden = diff.astype(np.int64) @ weight.astype(np.int64)
    exp_skips = int((diff == 0).sum()) * N       # each zero diff skips N MACs (one per col)

    for k in range(K):
        dut.diff_col.value = pack(diff[:, k], DW)     # M diff elements
        dut.w_row.value = pack(weight[k, :], WW)      # N weight elements
        dut.valid.value = 1
        await RisingEdge(dut.clk)

    dut.valid.value = 0
    await RisingEdge(dut.clk)
    await Timer(1, unit="ns")

    flat = int(dut.c_flat.value)
    mismatches = 0
    for i in range(M):
        for j in range(N):
            got = unpack_acc(flat, i * N + j)
            if got != golden[i][j]:
                mismatches += 1
                dut._log.error(f"C[{i}][{j}] = {got} != {golden[i][j]}")
    assert mismatches == 0, f"{mismatches} output mismatches"

    skips = int(dut.skips_total.value)
    assert skips == exp_skips, f"skips {skips} != expected {exp_skips}"
    dut._log.info(f"4x4 array == numpy matmul (all {M*N} outputs), "
                  f"{skips}/{M*N*K} MAC-ops zero-skipped")
    dut._log.info(f"golden C[0] = {golden[0].tolist()}")


@cocotb.test()
async def test_array_second_tile(dut):
    """A second independent tile (reset between) to confirm reusability."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset(dut)

    rng = np.random.RandomState(7)
    diff = rng.randint(-254, 255, size=(M, K))
    diff[rng.random((M, K)) < 0.3] = 0
    weight = rng.randint(-128, 128, size=(K, N))
    golden = diff.astype(np.int64) @ weight.astype(np.int64)

    for k in range(K):
        dut.diff_col.value = pack(diff[:, k], DW)
        dut.w_row.value = pack(weight[k, :], WW)
        dut.valid.value = 1
        await RisingEdge(dut.clk)
    dut.valid.value = 0
    await RisingEdge(dut.clk)
    await Timer(1, unit="ns")

    flat = int(dut.c_flat.value)
    for i in range(M):
        for j in range(N):
            assert unpack_acc(flat, i * N + j) == golden[i][j], f"tile2 C[{i}][{j}]"
    dut._log.info("second tile also matches numpy matmul")

"""
test_pe_array_systolic.py - cocotb testbench for the 4x4 systolic PE array (version B).

Drives the staggered boundary feed (diff row i enters col 0 delayed by i cycles;
weight col j enters row 0 delayed by j cycles) so operands meet on the diagonal
wavefront, then checks the accumulators equal numpy matmul -- the SAME result the
parallel array (A) produced. This confirms the systolic dataflow computes the
identical matmul; A is the cross-check.

Run:  cd rtl && make array_sys
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


@cocotb.test()
async def test_systolic_matmul(dut):
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    dut.rst.value = 1
    dut.en.value = 0
    dut.a_left.value = 0
    dut.w_top.value = 0
    await RisingEdge(dut.clk)
    await RisingEdge(dut.clk)
    dut.rst.value = 0
    await RisingEdge(dut.clk)

    rng = np.random.RandomState(0)            # same seed as A
    diff = rng.randint(-254, 255, size=(M, K))
    diff[rng.random((M, K)) < 0.4] = 0
    weight = rng.randint(-128, 128, size=(K, N))
    golden = diff.astype(np.int64) @ weight.astype(np.int64)

    # boundary feed: at cycle c, col-0 of row i gets diff[i][c-i]; row-0 of col j gets
    # weight[c-j][j]. Outside [0,K) -> 0. Run enough cycles to fill+compute+drain.
    T = (M - 1) + (N - 1) + (K - 1) + 2
    for c in range(T):
        a_col0 = []
        for i in range(M):
            k = c - i
            a_col0.append(diff[i][k] if 0 <= k < K else 0)
        w_row0 = []
        for j in range(N):
            k = c - j
            w_row0.append(weight[k][j] if 0 <= k < K else 0)
        dut.a_left.value = pack(a_col0, DW)
        dut.w_top.value = pack(w_row0, WW)
        dut.en.value = 1
        await RisingEdge(dut.clk)

    dut.en.value = 0
    await RisingEdge(dut.clk)
    await Timer(1, unit="ns")

    flat = int(dut.c_flat.value)
    mism = 0
    for i in range(M):
        for j in range(N):
            got = unpack_acc(flat, i * N + j)
            if got != golden[i][j]:
                mism += 1
                dut._log.error(f"C[{i}][{j}] = {got} != {golden[i][j]}")
    assert mism == 0, f"{mism} systolic output mismatches"
    dut._log.info(f"systolic 4x4 == numpy matmul (all {M*N} outputs)")
    dut._log.info(f"systolic C[0] = {[unpack_acc(flat, j) for j in range(N)]} "
                  f"(version A golden = {golden[0].tolist()})")


@cocotb.test()
async def test_systolic_matches_parallel(dut):
    """Explicit A==B cross-check on a different tile: systolic result must equal the
    plain matmul (which version A also equals)."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    dut.rst.value = 1
    dut.en.value = 0
    dut.a_left.value = 0
    dut.w_top.value = 0
    await RisingEdge(dut.clk)
    await RisingEdge(dut.clk)
    dut.rst.value = 0
    await RisingEdge(dut.clk)

    rng = np.random.RandomState(7)            # same tile as A's second test
    diff = rng.randint(-254, 255, size=(M, K))
    diff[rng.random((M, K)) < 0.3] = 0
    weight = rng.randint(-128, 128, size=(K, N))
    golden = diff.astype(np.int64) @ weight.astype(np.int64)

    T = (M - 1) + (N - 1) + (K - 1) + 2
    for c in range(T):
        a_col0 = [diff[i][c - i] if 0 <= c - i < K else 0 for i in range(M)]
        w_row0 = [weight[c - j][j] if 0 <= c - j < K else 0 for j in range(N)]
        dut.a_left.value = pack(a_col0, DW)
        dut.w_top.value = pack(w_row0, WW)
        dut.en.value = 1
        await RisingEdge(dut.clk)
    dut.en.value = 0
    await RisingEdge(dut.clk)
    await Timer(1, unit="ns")

    flat = int(dut.c_flat.value)
    for i in range(M):
        for j in range(N):
            assert unpack_acc(flat, i * N + j) == golden[i][j], f"C[{i}][{j}]"
    dut._log.info("systolic (B) == numpy matmul == parallel array (A) on second tile")

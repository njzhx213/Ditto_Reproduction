#!/usr/bin/env python3
"""
gen_trace_o3.py - per-layer trace in SimpleO3 format (bubble + concurrent mem).

Unlike the ReadWriteTrace path (serial, no compute, gave stuck 1.00x speedup),
SimpleO3 overlaps compute with concurrent memory (128-deep window, 16 MSHRs, LLC).
Each trace line is:
    bubble_count  load_addr  [store_addr]
where bubble_count = the COMPUTE cycles for that tile, so SimpleO3 runs
max(compute, memory) per tile with real concurrency.

Per-tile bubble (compute cycles):
  ITC / act : tile_macs / N_PE_ITC          (8-bit, 27648 PEs)
  diff      : tile_macs * NONZERO * BITF / (N_PE_DITTO * LANES)   (4-bit + skip)
              x2 sub-ops if attn_self (handled by caller via n_subop)

Memory per tile = the same A/B/C (+ prev) cache-line addresses as before, but now
attached to the tile's bubble. The first mem op of a tile carries the bubble; the
rest carry bubble 0 (compute already paid).

This module reuses choose_tiles from gen_trace and only changes the EMIT format.
Validated standalone on one layer before wiring into the full network.

    cd ~/Ditto && python3 sim/gen_trace_o3.py
"""
from __future__ import annotations

import math
import re
import subprocess
import tempfile
from pathlib import Path

import gen_trace as G   # reuse tiling, constants, DRAM config

CL = G.CL
REGION_STRIDE = G.REGION_STRIDE
RAMULATOR = G.RAMULATOR

N_PE_ITC = 27648
N_PE_DITTO = 39398
LANES = 4
BW_ZERO, BW_LE4, BW_GT4 = 0.459, 0.534, 0.007
NONZERO = BW_LE4 + BW_GT4
BITF = (BW_LE4 + 2 * BW_GT4) / NONZERO


def tile_bubble(tile_macs, mode, n_subop=1):
    """Compute cycles for a tile, by execution mode.

    Three DISTINCT compute models (the earlier bug made itc==act, forcing
    speedup to a 1.00x algebraic identity when Defo flipped every layer):
      itc : ITC baseline, 27648 PEs, native 8-bit         -> macs / 27648
      act : Ditto running original activations. Ditto PEs are 4-bit; an 8-bit MAC
            uses TWO 4-bit multipliers + shift, so effective 8-bit throughput is
            39398*4 lanes / 2 = 39398*2                    -> macs / (39398*2)
      diff: 4-bit + zero-skip on 39398 PEs x 4 lanes (x2 sub-ops for attn_self)
    """
    if mode == "itc":
        return max(1, round(tile_macs / N_PE_ITC))
    if mode == "act":
        return max(1, round(tile_macs / (N_PE_DITTO * 2)))
    # diff
    cyc = n_subop * tile_macs * NONZERO * BITF / (N_PE_DITTO * LANES)
    return max(1, round(cyc))


def gen_layer_o3(M, K, N, mode, buf_elems, n_subop=1):
    """SimpleO3-format trace lines for one layer. Returns (lines, info).
    mode in {itc, act, diff}. Memory pattern: diff reads prev (PA,PC); itc and act
    read only A,B,C. Bubble: per-mode compute model (see tile_bubble).

    The tile's compute cycles are SPREAD EVENLY across the tile's memory ops (not
    front-loaded onto the first op). Front-loading made a big bubble act as a
    spacer that pushed accesses apart and reduced bank conflicts -> a faster total
    for SLOWER compute, an artifact. Spreading lets SimpleO3's window overlap
    compute and concurrent memory so tile time ~ max(compute, memory)."""
    has_prev = (mode == "diff")
    Tm, Tn, Tk = G.choose_tiles(M, K, N, buf_elems)
    names = ["A", "B", "C"] + (["PA", "PC"] if has_prev else [])
    base = {nm: i * REGION_STRIDE for i, nm in enumerate(names)}

    lines = []

    def tile_addrs(im, jn, Tm_i, Tn_j):
        """All (addr, is_write) cache-line accesses for one tile, in order."""
        acc = []

        def runs(b, off, nbytes, write=False):
            a = (off // CL) * CL
            end = off + nbytes
            while a < end:
                acc.append((b + a, write))
                a += CL
        runs(base["A"], im * Tm * K, Tm_i * K)
        if has_prev:
            runs(base["PA"], im * Tm * K, Tm_i * K)
        runs(base["B"], jn * Tn * K, K * Tn_j)
        runs(base["C"], im * Tm * N + jn * Tn, Tm_i * Tn_j, write=True)
        if has_prev:
            runs(base["PC"], im * Tm * N + jn * Tn, Tm_i * Tn_j)
        return acc

    nM = math.ceil(M / Tm)
    nN = math.ceil(N / Tn)
    for im in range(nM):
        Tm_i = min(Tm, M - im * Tm)
        for jn in range(nN):
            Tn_j = min(Tn, N - jn * Tn)
            tile_macs = Tm_i * Tn_j * K
            total_bub = tile_bubble(tile_macs, mode, n_subop)
            acc = tile_addrs(im, jn, Tm_i, Tn_j)
            n_acc = len(acc)
            if n_acc == 0:
                continue
            # spread total_bub across the tile's accesses (integer split)
            base_b = total_bub // n_acc
            extra = total_bub - base_b * n_acc       # remainder on the first ops
            for idx, (addr, write) in enumerate(acc):
                bub = base_b + (1 if idx < extra else 0)
                if write:
                    lines.append(f"{bub} {addr} {addr}")
                else:
                    lines.append(f"{bub} {addr}")
    info = {"Tm": Tm, "Tn": Tn, "nM": nM, "nN": nN, "n_lines": len(lines)}
    return lines, info


def run_simpleo3(lines, workdir, llc_mb=192):
    """Run SimpleO3. LLC sized to ~paper 192MB SRAM by default."""
    tp = workdir / "o3.trace"
    cp = workdir / "o3.yaml"
    tp.write_text("\n".join(lines) + "\n")
    D = G.DRAM
    cp.write_text(f"""Frontend:
  impl: SimpleO3
  clock_ratio: 8
  num_expected_insts: {len(lines)}
  ipc: 4
  inst_window_depth: 128
  llc_capacity_per_core: {llc_mb}MB
  llc_num_mshr_per_core: 16
  traces:
    - {tp}
  Translation:
    impl: RandomTranslation
    max_addr: 2147483648
MemorySystem:
  impl: GenericDRAM
  clock_ratio: 3
  DRAM:
    impl: {D["impl"]}
    org:
      preset: {D["org"]}
      channel: {D["channel"]}
{D["extra"]}    timing:
      preset: {D["timing"]}
  Controller:
    impl: Generic
    Scheduler: {{impl: FRFCFS}}
    RefreshManager: {{impl: AllBank}}
    RowPolicy: {{impl: ClosedRowPolicy, cap: 4}}
  AddrMapper:
    impl: RoBaRaCoCh
""")
    out = subprocess.run([str(RAMULATOR), "-f", str(cp)],
                         capture_output=True, text=True)
    txt = out.stdout
    core = re.search(r"cycles_recorded_core_0:\s*(\d+)", txt)
    return {"core_cyc": int(core.group(1)) if core else None,
            "stderr": out.stderr[-300:]}


def main():
    if not RAMULATOR.exists():
        raise SystemExit(f"ramulator2 not found at {RAMULATOR}")
    tmp = Path(tempfile.mkdtemp(prefix="o3layer_"))
    buf = 192 * 1024 * 1024   # ~paper SRAM as on-chip buffer

    # representative layer: 256x1280x1280 (the one we benchmarked before)
    M, K, N = 256, 1280, 1280
    print(f"=== SimpleO3 per-layer validation: GEMM {M}x{K}x{N} ===\n")
    print(f"{'mode':>8} {'n_lines':>9} {'core_cyc':>10} {'note'}")
    print("-" * 50)
    results = {}
    for mode in ["act", "diff"]:
        lines, info = gen_layer_o3(M, K, N, mode, buf)
        r = run_simpleo3(lines, tmp)
        results[mode] = r["core_cyc"]
        if r["core_cyc"] is None:
            print(f"{mode:>8}  FAILED  stderr={r['stderr']}")
        else:
            print(f"{mode:>8} {info['n_lines']:>9} {r['core_cyc']:>10}")

    if results.get("act") and results.get("diff"):
        ratio = results["diff"] / results["act"]
        print(f"\ndiff/act core-cycle ratio: {ratio:.2f}x")
        print("act = 8-bit compute + act memory; diff = 4-bit+skip compute + more")
        print("memory (prev). SimpleO3 overlaps compute & concurrent memory, so this")
        print("ratio reflects real roofline, not the serial ReadWriteTrace sum.")
        print(f"\nCompute-only would favor diff (~10x less compute); if diff/act > 1")
        print(f"here, the prev memory traffic dominates -> the Fig 16 effect, now")
        print(f"with concurrency. If diff/act < 1, compute savings win at this BW.")


if __name__ == "__main__":
    main()

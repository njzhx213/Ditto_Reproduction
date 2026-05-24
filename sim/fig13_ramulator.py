#!/usr/bin/env python3
"""
fig13_ramulator.py - Step 1 of wiring Ramulator into the MAIN cycle model.

So far the main speedup/Defo line used the ANALYTICAL memory model (bytes/bpc).
This replaces the memory term with REAL Ramulator DDR4 cycles. Because Ramulator
computes the bandwidth/timing internally (from the DDR4 preset), there is NO bpc
knob anymore -- memory cycles are a physical result, not a swept parameter.

This Step-1 script does NOT run the full 282-layer network. It:
  1. wires layer memory -> Ramulator (act + diff), normalized to the 1GHz compute
     domain (x 5/6, see clock note in gen_trace),
  2. CACHES results by (M, K, N, mode) so repeated layer shapes run once,
  3. TIMES each Ramulator call,
  4. runs a small representative set + a few memory-heavy layers,
  5. prints per-layer timing and EXTRAPOLATES the full-network runtime,
so we can decide the full-run strategy from real numbers, not guesses.

Honesty: trace size for big layers can be huge; if a single layer is slow we will
add trace down-sampling (scale rows down, scale cycles back up) before full run.

    cd ~/Ditto && python3 sim/fig13_ramulator.py
"""
from __future__ import annotations

import time
from pathlib import Path
import tempfile

import gen_trace as G      # tiling trace generator + run_ramulator + clock norm
import fig13_speedup as F  # enumeration + compute model + Defo

MEM_CLK_TO_COMPUTE = 5.0 / 6.0     # 1.0GHz / 1.2GHz (DDR4-2400), see gen_trace
BUF_ELEMS = 256 * 1024             # on-chip buffer for tiling (int8 elements)

_cache = {}     # (M, K, N, mode) -> (mem_cycles_1GHz, n_trace_lines, seconds)


def ramulator_mem_cyc(M, K, N, mode, tmp):
    key = (M, K, N, mode)
    if key in _cache:
        return _cache[key]
    lines, _info = G.gen_layer_lines(M, K, N, mode, BUF_ELEMS)
    t0 = time.time()
    res = G.run_ramulator(lines, tmp)
    dt = time.time() - t0
    cyc = (res["cycles"] or 0) * MEM_CLK_TO_COMPUTE
    out = (cyc, len(lines), dt)
    _cache[key] = out
    return out


def main():
    if not G.RAMULATOR.exists():
        raise SystemExit(f"ramulator2 not found at {G.RAMULATOR}")

    print("Enumerating UNet (gen_trace, carries M,K,N) ...", flush=True)
    layers = G.enumerate_layers()   # each has kind, M, K, N, macs
    n = len(layers)
    print(f"Total layers: {n}")

    # pick a small but representative set: smallest, median, largest MAC + 3 mem-heavy
    by_macs = sorted(range(n), key=lambda i: layers[i]["macs"])
    by_actbytes = sorted(range(n),
                         key=lambda i: layers[i]["M"] * layers[i]["K"]
                         + layers[i]["K"] * layers[i]["N"]
                         + layers[i]["M"] * layers[i]["N"])
    pick = sorted(set([by_macs[0], by_macs[n // 2], by_macs[-1],
                       by_actbytes[-1], by_actbytes[-2], by_actbytes[-3]]))
    print(f"Running {len(pick)} representative/heavy layers (of {n}); "
          f"caching by (M,K,N,mode).\n")

    tmp = Path(tempfile.mkdtemp(prefix="fig13ram_"))
    print(f"{'idx':>5} {'kind':>7} {'M':>6} {'K':>6} {'N':>5} {'MAC(M)':>8} "
          f"{'mem_act':>10} {'mem_diff':>10} {'lines_d':>9} {'sec_act':>8} {'sec_diff':>8}")
    print("-" * 100)
    total_sec = 0.0
    per_layer_sec = []
    for i in pick:
        L = layers[i]
        M, K, N = L["M"], L["K"], L["N"]
        ca, la, ta = ramulator_mem_cyc(M, K, N, "act", tmp)
        cd, ld, td = ramulator_mem_cyc(M, K, N, "diff", tmp)
        total_sec += ta + td
        per_layer_sec.append(ta + td)
        print(f"{i:>5} {L['kind']:>7} {M:>6} {K:>6} {N:>5} {L['macs']/1e6:>8.1f} "
              f"{ca:>10.0f} {cd:>10.0f} {ld:>9} {ta:>8.2f} {td:>8.2f}")

    # ---- extrapolate full-network time, accounting for shape caching ----
    # count UNIQUE (M,K,N) across all layers (each needs act+diff once)
    uniq = set((L["M"], L["K"], L["N"]) for L in layers)
    avg_sec_per_shape = (total_sec / len(pick)) if pick else 0.0
    est_full = len(uniq) * avg_sec_per_shape
    print("-" * 100)
    print(f"\nMeasured: {len(pick)} layers in {total_sec:.1f}s "
          f"(avg {avg_sec_per_shape:.2f}s/layer incl. act+diff)")
    print(f"Unique (M,K,N) shapes in full net: {len(uniq)} of {n} layers "
          f"({100*len(uniq)/n:.0f}% unique -> caching helps)")
    print(f"ESTIMATED full-network Ramulator time (unique shapes only): "
          f"~{est_full:.0f}s (~{est_full/60:.1f} min)")
    if per_layer_sec:
        print(f"Slowest sampled layer: {max(per_layer_sec):.1f}s "
              f"(big layers dominate; trace down-sampling may be needed if this is large)")
    print("\nNext: if the estimate is acceptable, run the full net; if too slow,")
    print("add trace down-sampling (scale rows down, scale cycles up) for big layers.")


if __name__ == "__main__":
    main()

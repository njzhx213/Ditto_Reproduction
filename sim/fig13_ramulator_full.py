#!/usr/bin/env python3
"""
fig13_ramulator_full.py - MAIN cycle model with REAL Ramulator memory (full net).

This is the faithful path landed: the memory term of every layer's cycle count
comes from cycle-accurate Ramulator DDR4-2400 timing, NOT the analytical
bytes/bpc. There is no bandwidth knob -- bandwidth is a physical result of the
DDR4 preset.

Pipeline
--------
  for each layer (cached by (M,K,N,mode)):
      mem_cyc_act  = Ramulator(act trace)  x 5/6   (-> 1GHz compute domain)
      mem_cyc_diff = Ramulator(diff trace) x 5/6
  cyc_itc  = max(compute_itc,        mem_cyc_act)
  cyc_diff = max(compute_diff(zero), mem_cyc_diff)
  cyc_act  = max(compute_act_ditto,  mem_cyc_act)
  Defo: per layer pick min(cyc_act, cyc_diff); diff-only = always cyc_diff.
  speedup_X = sum(cyc_itc) / sum(cyc_X)

Outputs (all Ramulator-backed, no free parameter):
  - ITC / diff-only / +Defo full-network speedup
  - Defo flip fraction
  - bare diff/act memory ratio (-> paper Fig 8, 2.75x) and Defo-selected (-> Fig 14)
  - saved to results/fig13_ramulator.json

Runtime: ~10-18 min (49 unique shapes x act+diff; big convs dominate). One-shot;
results cached to JSON so plotting/writeup never re-runs it.

    cd ~/Ditto && python3 sim/fig13_ramulator_full.py
"""
from __future__ import annotations

import argparse
import json
import time
import tempfile
from pathlib import Path

import gen_trace as G
import fig13_speedup as F

MEM_CLK_TO_COMPUTE = 5.0 / 6.0
RESULT = Path.home() / "Ditto" / "results" / "fig13_ramulator.json"

_cache = {}


def mem_cyc(M, K, N, mode, buf_elems, tmp):
    key = (M, K, N, mode, buf_elems)
    if key in _cache:
        return _cache[key]
    lines, _ = G.gen_layer_lines(M, K, N, mode, buf_elems)
    res = G.run_ramulator(lines, tmp)
    cyc = (res["cycles"] or 0) * MEM_CLK_TO_COMPUTE
    _cache[key] = cyc
    return cyc


def run_full(layers, buf_elems, tmp, t_start):
    tot_itc = tot_diff_only = tot_defo = 0.0
    eff_act_sum = eff_diff_sum = eff_defo_sum = 0.0
    n_flip = 0
    for L in layers:
        M, K, N, macs = L["M"], L["K"], L["N"], L["macs"]
        ma = mem_cyc(M, K, N, "act", buf_elems, tmp)
        md = mem_cyc(M, K, N, "diff", buf_elems, tmp)
        cyc_itc = max(F.comp_itc(macs), ma)
        cyc_diff = max(F.comp_diff(macs), md)
        cyc_act = max(F.comp_act_ditto(macs), ma)
        tot_itc += cyc_itc
        tot_diff_only += cyc_diff
        eff_act_sum += ma
        eff_diff_sum += md
        if cyc_act < cyc_diff:
            tot_defo += cyc_act
            eff_defo_sum += ma
            n_flip += 1
        else:
            tot_defo += cyc_diff
            eff_defo_sum += md
    n = len(layers)
    return {
        "buf_mb": buf_elems / (1024 * 1024),
        "speedup_diff_only": tot_itc / tot_diff_only if tot_diff_only else 0.0,
        "speedup_defo": tot_itc / tot_defo if tot_defo else 0.0,
        "flip_frac": n_flip / n if n else 0.0,
        "mem_ratio_bare": eff_diff_sum / eff_act_sum if eff_act_sum else 0.0,
        "mem_ratio_defo": eff_defo_sum / eff_act_sum if eff_act_sum else 0.0,
    }


def main():
    ap = argparse.ArgumentParser()
    # paper Table III: Ditto has 192 MB SRAM. Sweep around it (in MB of int8 elems).
    ap.add_argument("--buf-mb", type=float, nargs="+",
                    default=[0.25, 4, 32, 96, 192],
                    help="on-chip buffer sizes (MB) to sweep; default brackets paper 192MB")
    args = ap.parse_args()

    if not G.RAMULATOR.exists():
        raise SystemExit(f"ramulator2 not found at {G.RAMULATOR}")

    print("Enumerating UNet ...", flush=True)
    layers = G.enumerate_layers()
    n = len(layers)
    uniq = set((L["M"], L["K"], L["N"]) for L in layers)
    print(f"{n} layers, {len(uniq)} unique shapes.")
    print(f"Sweeping on-chip buffer: {args.buf_mb} MB "
          f"(paper Table III Ditto SRAM = 192 MB)\n", flush=True)

    tmp = Path(tempfile.mkdtemp(prefix="fig13ramfull_"))
    t_start = time.time()
    rows = []
    for mb in args.buf_mb:
        buf_elems = int(mb * 1024 * 1024)
        r = run_full(layers, buf_elems, tmp, t_start)
        rows.append(r)
        print(f"BUF={mb:>6} MB | diff-only {r['speedup_diff_only']:.2f}x | "
              f"+Defo {r['speedup_defo']:.2f}x | flip {r['flip_frac']*100:5.1f}% | "
              f"bare-mem {r['mem_ratio_bare']:.2f}x | defo-mem {r['mem_ratio_defo']:.2f}x | "
              f"{time.time()-t_start:.0f}s", flush=True)

    print("\n=== Full-network, REAL Ramulator (DDR4-2400), buffer sweep ===")
    print("Paper refs: Fig 8 bare-mem 2.75x | Fig 14 Ditto 1.56x | "
          "Fig 17 flip ~14.4% | Fig 13 ~1.5x speedup")
    print(f"DRAM: {G.DRAM_ORG}/{G.DRAM_TIMING}, single channel.\n")
    best = None
    for r in rows:
        note = ""
        if 0.10 <= r["flip_frac"] <= 0.25:
            note = "  <- flip% near paper 14.4%"
            best = r
        print(f"  BUF {r['buf_mb']:>6.2f} MB: diff-only {r['speedup_diff_only']:.2f}x, "
              f"+Defo {r['speedup_defo']:.2f}x, flip {r['flip_frac']*100:.1f}%, "
              f"bare-mem {r['mem_ratio_bare']:.2f}x{note}")
    print("\nIf flip% is still ~100% at large buffers, the bottleneck is the single")
    print("-channel DDR4 itself (bandwidth too low for 340 GMACs), not the buffer.")

    RESULT.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULT, "w") as f:
        json.dump({"dram": f"{G.DRAM_ORG}/{G.DRAM_TIMING}", "n_layers": n,
                   "n_unique_shapes": len(uniq), "buffer_sweep": rows}, f, indent=2)
    print(f"\nSaved: {RESULT}")


if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()

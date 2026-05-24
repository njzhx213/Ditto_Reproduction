#!/usr/bin/env python3
"""
fig13_simpleo3_full.py - full-network speedup on SimpleO3 (concurrent mem + overlap).

Replaces the serial ReadWriteTrace memory model with SimpleO3 per-layer cycles
(compute bubble overlapped with concurrent memory, 16 MSHRs, 192MB LLC ~ paper
SRAM). This is the path that does NOT degenerate to a stuck 1.00x, because memory
concurrency is actually modeled.

Per layer we run SimpleO3 three ways and take the core cycle count:
  itc  : act trace, 8-bit bubbles      -> baseline
  act  : same as itc here (Ditto running a layer in original-activation mode)
  diff : diff trace, 4-bit+skip bubbles + prev memory
Defo: per layer min(act, diff); diff-only: always diff. speedup = sum(itc)/sum(X).

Linear/Conv only for now (attention added in a later step once this is healthy).

Runs a TIMING sample first and extrapolates, so we decide full-run cost from real
numbers (SimpleO3 is heavier than ReadWriteTrace). Pass --full to run everything.

    cd ~/Ditto && python3 sim/fig13_simpleo3_full.py          # timing sample
    cd ~/Ditto && python3 sim/fig13_simpleo3_full.py --full   # full network
"""
from __future__ import annotations

import sys
import time
import json
import tempfile
from pathlib import Path

import gen_trace as G
import gen_trace_o3 as O3

BUF = 192 * 1024 * 1024
RESULT = Path.home() / "Ditto" / "results" / "fig13_simpleo3.json"

_cache = {}   # (M,K,N,mode) -> core_cyc


def layer_cyc(M, K, N, mode, tmp):
    key = (M, K, N, mode)
    if key in _cache:
        return _cache[key]
    lines, _ = O3.gen_layer_o3(M, K, N, mode, BUF)
    r = O3.run_simpleo3(lines, tmp, llc_mb=192)
    cyc = r["core_cyc"] or 0
    _cache[key] = cyc
    return cyc


def main():
    full = "--full" in sys.argv
    if not G.RAMULATOR.exists():
        raise SystemExit("ramulator2 not found")

    print("Enumerating UNet (linear/conv) ...", flush=True)
    layers = G.enumerate_layers()
    n = len(layers)
    uniq = sorted(set((L["M"], L["K"], L["N"]) for L in layers))
    print(f"{n} layers, {len(uniq)} unique shapes. DRAM={G.DRAM['impl']} "
          f"ch={G.DRAM['channel']}, SimpleO3 LLC=192MB\n", flush=True)

    tmp = Path(tempfile.mkdtemp(prefix="o3full_"))

    if not full:
        # timing sample: smallest, median, largest by MACs + a couple mem-heavy
        bym = sorted(range(n), key=lambda i: layers[i]["macs"])
        pick = sorted(set([bym[0], bym[n // 2], bym[-1], bym[-2]]))
        print("TIMING SAMPLE (use --full for the whole net):")
        print(f"{'M':>6} {'K':>6} {'N':>5} {'itc_cyc':>9} {'act_cyc':>9} "
              f"{'diff_cyc':>9} {'d/a':>6} {'sec':>7}")
        print("-" * 64)
        tot = 0.0
        for i in pick:
            L = layers[i]
            t0 = time.time()
            ci = layer_cyc(L["M"], L["K"], L["N"], "itc", tmp)
            ca = layer_cyc(L["M"], L["K"], L["N"], "act", tmp)
            cd = layer_cyc(L["M"], L["K"], L["N"], "diff", tmp)
            dt = time.time() - t0
            tot += dt
            ratio = cd / ca if ca else 0
            print(f"{L['M']:>6} {L['K']:>6} {L['N']:>5} {ci:>9} {ca:>9} {cd:>9} "
                  f"{ratio:>5.2f}x {dt:>6.1f}s")
        avg = tot / len(pick)
        est = len(uniq) * avg
        print(f"\navg {avg:.1f}s/shape (act+diff); {len(uniq)} unique shapes")
        print(f"ESTIMATED full run: ~{est:.0f}s (~{est/60:.1f} min). "
              f"Re-run with --full when ready.")
        return

    # full network
    t_start = time.time()
    tot_itc = tot_diff = tot_defo = 0.0
    n_flip = 0
    last = time.time()
    done = 0
    for L in layers:
        M, K, N, macs = L["M"], L["K"], L["N"], L["macs"]
        ci = layer_cyc(M, K, N, "itc", tmp)    # ITC baseline (27648 PE, 8-bit)
        ca = layer_cyc(M, K, N, "act", tmp)    # Ditto act-mode (4-bit PEs x2 for 8-bit)
        cd = layer_cyc(M, K, N, "diff", tmp)
        tot_itc += ci
        tot_diff += cd
        if ca < cd:              # Defo flips to act
            tot_defo += ca
            n_flip += 1
        else:
            tot_defo += cd
        done += 1
        if time.time() - last > 20:
            print(f"  ... {done}/{n}, {len(_cache)} cached, "
                  f"{time.time()-t_start:.0f}s", flush=True)
            last = time.time()

    out = {
        "dram": f"{G.DRAM['impl']}_ch{G.DRAM['channel']}",
        "n_layers": n,
        "speedup_diff_only": tot_itc / tot_diff if tot_diff else 0,
        "speedup_defo": tot_itc / tot_defo if tot_defo else 0,
        "flip_frac": n_flip / n if n else 0,
        "elapsed_sec": time.time() - t_start,
    }
    print("\n=== Full-network SimpleO3 (concurrent memory + overlap) ===")
    print(f"DRAM {out['dram']}, LLC 192MB, {n} layers, "
          f"{out['elapsed_sec']:.0f}s")
    print(f"  ITC baseline:      1.00x")
    print(f"  Ditto diff-only:   {out['speedup_diff_only']:.2f}x")
    print(f"  Ditto + Defo:      {out['speedup_defo']:.2f}x")
    print(f"  Defo flip:         {out['flip_frac']*100:.1f}% "
          f"(paper Fig 17 ~14.4%)")
    print("\nThis is concurrent (not serial): if flip% is far below 100% and")
    print("speedup is a real number (not stuck 1.00x), SimpleO3 fixed the")
    print("degeneration. Absolute value still depends on the (unpublished) DRAM;")
    print("8-channel DDR4 is a roofline-justified choice, documented as such.")

    RESULT.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULT, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {RESULT}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
simpleo3_probe.py - minimal validation that SimpleO3 gives concurrent
memory + compute-overlap behavior (unlike the serial ReadWriteTrace).

SimpleO3 trace format (from trace.cpp):
    bubble_count  load_addr  [store_addr]
  - bubble_count: compute cycles BEFORE this memory op (lets compute overlap mem)
  - load_addr: address read
  - store_addr (optional 3rd col): address written
SimpleO3 has a 128-deep instruction window + 16 MSHRs + LLC, so multiple memory
requests are in flight concurrently and overlap with the bubbles -> real
roofline max(compute, memory), not serial sum.

This probe builds a tiny synthetic trace and sweeps the compute/memory balance:
  - BIG bubbles, few mem ops  -> compute-bound  -> cycles ~ total bubbles
  - tiny bubbles, many mem ops -> memory-bound  -> cycles ~ memory latency sum
If SimpleO3 shows that transition, overlap works and we can build the real
per-layer trace on it.

    cd ~/Ditto && python3 sim/simpleo3_probe.py
"""
import subprocess
import re
import tempfile
from pathlib import Path

RAMULATOR = Path.home() / "ramulator2" / "build" / "ramulator2"
CL = 64


def make_cfg(trace_path, n_insts, llc_mb=2, tmp=None):
    """SimpleO3 + DDR4 8-channel config."""
    return f"""Frontend:
  impl: SimpleO3
  clock_ratio: 8
  num_expected_insts: {n_insts}
  ipc: 4
  inst_window_depth: 128
  llc_capacity_per_core: {llc_mb}MB
  llc_num_mshr_per_core: 16
  traces:
    - {trace_path}
  Translation:
    impl: RandomTranslation
    max_addr: 2147483648
MemorySystem:
  impl: GenericDRAM
  clock_ratio: 3
  DRAM:
    impl: DDR4
    org:
      preset: DDR4_8Gb_x8
      channel: 8
      rank: 2
    timing:
      preset: DDR4_2400R
  Controller:
    impl: Generic
    Scheduler: {{impl: FRFCFS}}
    RefreshManager: {{impl: AllBank}}
    RowPolicy: {{impl: ClosedRowPolicy, cap: 4}}
  AddrMapper:
    impl: RoBaRaCoCh
"""


def run(trace_lines, tmp, llc_mb=2):
    tp = tmp / "s.trace"
    tp.write_text("\n".join(trace_lines) + "\n")
    cp = tmp / "s.yaml"
    cp.write_text(make_cfg(tp, len(trace_lines), llc_mb, tmp))
    out = subprocess.run([str(RAMULATOR), "-f", str(cp)],
                         capture_output=True, text=True)
    txt = out.stdout
    cyc = re.search(r"memory_system_cycles:\s*(\d+)", txt)
    rec = re.search(r"cycles_recorded_core_0:\s*(\d+)", txt)
    memacc = re.search(r"memory_access_cycles_recorded_core_0:\s*(\d+)", txt)
    return {
        "mem_sys_cyc": int(cyc.group(1)) if cyc else None,
        "core_cyc": int(rec.group(1)) if rec else None,
        "mem_acc_cyc": int(memacc.group(1)) if memacc else None,
        "stderr": out.stderr[-300:],
    }


def main():
    if not RAMULATOR.exists():
        raise SystemExit(f"ramulator2 not found at {RAMULATOR}")
    tmp = Path(tempfile.mkdtemp(prefix="o3probe_"))

    N = 2000  # memory ops
    # spread addresses so they hit different channels/banks (concurrency)
    addrs = [(i * CL * 37) % (2 << 30) for i in range(N)]

    print("=== SimpleO3 overlap probe (DDR4 8-channel) ===\n")
    print(f"{'bubble/op':>10} {'core_cyc':>10} {'mem_sys_cyc':>12} "
          f"{'mem_acc_cyc':>12} {'regime':>14}")
    print("-" * 64)
    results = []
    for bubble in [0, 1, 5, 20, 100]:
        # each line: bubble compute cycles, then a load (read)
        lines = [f"{bubble} {a}" for a in addrs]
        r = run(lines, tmp)
        if r["mem_sys_cyc"] is None:
            print(f"{bubble:>10}  FAILED  stderr={r['stderr']}")
            continue
        core = r["core_cyc"] or 0
        macc = r["mem_acc_cyc"] or 0
        regime = "compute-bound" if bubble * N > macc else "memory-bound"
        results.append((bubble, core, r["mem_sys_cyc"], macc))
        print(f"{bubble:>10} {core:>10} {r['mem_sys_cyc']:>12} {macc:>12} {regime:>14}")

    print("\nInterpretation:")
    print("- bubble=0: pure memory, all ops concurrent -> mem-bound floor")
    print("- large bubble: compute dominates, memory hides under compute (overlap)")
    print("- if core_cyc grows ~linearly with bubble at large bubble, compute is")
    print("  exposed; if it stays flat at small bubble, memory is the floor.")
    if len(results) >= 2:
        lo, hi = results[0], results[-1]
        print(f"\nbubble 0 -> {lo[1]} core cyc; bubble {hi[0]} -> {hi[1]} core cyc.")
        if hi[1] > lo[1] * 1.5:
            print("Compute became exposed at high bubble -> overlap WORKS, "
                  "SimpleO3 is usable for real speedup.")
        else:
            print("Core cycles barely changed -> investigate (mem floor very high?).")

    # concurrency check: compare 8-channel vs would-be serial
    # (ReadWriteTrace gave mem_sys_cyc ~ N*latency; SimpleO3 should be much less
    #  for bubble=0 because MSHRs overlap requests)
    if results:
        b0 = results[0]
        naive_serial = N * 50    # ~50 cyc/access if fully serial
        print(f"\nConcurrency: bubble=0 mem_sys_cyc={b0[2]} vs naive-serial "
              f"~{naive_serial}. Lower = requests overlapping across channels/MSHRs.")


if __name__ == "__main__":
    main()

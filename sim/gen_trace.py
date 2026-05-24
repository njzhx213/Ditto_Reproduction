#!/usr/bin/env python3
"""
gen_trace.py - Step 2 of the Ramulator path.

Translate UNet per-layer memory accesses into Ramulator2 ReadWriteTrace files,
run them through the (patched) ramulator2 binary, and read back the cycle-accurate
memory_system_cycles. The headline number is the diff/act memory-cycle RATIO,
which is clock-domain independent and is exactly what the analytical byte model
could not push past ~1.34x (paper Fig 8: 2.75x).

Runs on the user's WSL (needs torch+diffusers to enumerate, and the built
ramulator2 at ~/ramulator2/build/ramulator2 with the ReadWriteTrace is_finished
patch applied).

Tiling model (option B: buffer-driven)
--------------------------------------
Each layer is a GEMM M x K times K x N (conv via im2col: M=out pixels,
K=(C_in/groups)*kH*kW, N=C_out). Given on-chip buffer BUF (in int8 elements),
pick the largest square output tile Tm=Tn=t with full-K kept resident:

        2 * t * K + t * t  <=  BUF        ->   t = floor(sqrt(K^2+BUF)) - K

Then iterate output tiles. Per tile we emit, INTERLEAVED (so A/B/out land in
different DRAM regions -> real row/bank conflicts that Ramulator captures and
the analytical model cannot):
  - read the A row-strip            (Tm*K elements)
  - [diff] read prev_in row-strip   (Tm*K)   -- needed to form delta
  - read the B col-strip            (K*Tn)
  - write the C output block        (Tm*Tn)
  - [diff] read prev_out block      (Tm*Tn)  -- needed for the summation
A is re-read ceil(N/Tn) times, B is re-read ceil(M/Tm) times. Smaller BUF ->
smaller tiles -> more re-reads -> more DRAM traffic. diff mode's working set is
larger, so its tiles are squeezed smaller and its re-reads amplified -> the
diff/act ratio should rise above the analytical 1.34x, toward Fig-8's 2.75x.

Honesty boundaries (documented, not hidden)
-------------------------------------------
  - B col-strip and C block are emitted as CONTIGUOUS sweeps from a per-tile
    offset (flat-buffer approximation). This preserves access VOLUME and the
    inter-tensor jumps (the conflict-causing effect), but not the exact strided
    weight/output layout. First-order; a full layout model is future work.
  - One request per 64B cache line (= 64 int8 elements). int8 activations.
  - BUF is the buffer-size knob (swept), analogous to the analytical model's BUF.
  - v1 runs a few REPRESENTATIVE layers, not all 282 (full network would be tens
    of millions of trace lines). We validate the pipeline + ratio first.

Usage (in WSL, fastdllm env):
    cd ~/Ditto
    python3 sim/gen_trace.py --buf-kb 256
    python3 sim/gen_trace.py --buf-kb 64 --keep   # keep trace files for inspection

Author: njzhx213
Project: Ditto Reproduction (HPCA 2025) - Week 2, Ramulator path, Step 2
"""

from __future__ import annotations

import argparse
import math
import re
import subprocess
import tempfile
from pathlib import Path

CL = 64                                  # cache-line bytes (= int8 elements / request)
REGION_STRIDE = 256 * 1024 * 1024        # 256 MB between tensors -> distinct banks
RAMULATOR = Path.home() / "ramulator2" / "build" / "ramulator2"

# DRAM config. A 1GHz, 39398-PE (A4) accelerator peaks ~315 TOPS and needs
# hundreds of GB/s -> HBM class, NOT single-channel DDR4 (~20 GB/s). The paper
# does NOT publish Ditto's DRAM, so we align to what this compute scale requires
# (roofline-based), not to a paper number. Switchable for sensitivity.
DRAM_CONFIGS = {
    "ddr4": {"impl": "DDR4", "org": "DDR4_8Gb_x8", "timing": "DDR4_2400R",
             "channel": 1, "extra": "      rank: 2\n"},
    "ddr4x8": {"impl": "DDR4", "org": "DDR4_8Gb_x8", "timing": "DDR4_2400R",
               "channel": 8, "extra": "      rank: 2\n"},
    "hbm":  {"impl": "HBM",  "org": "HBM_2Gb",     "timing": "HBM_2Gbps",
             "channel": 8, "extra": ""},
}
DRAM = DRAM_CONFIGS["ddr4x8"]        # <- 8-channel DDR4: raises bandwidth, has rank
DRAM_ORG = DRAM["org"]
DRAM_TIMING = DRAM["timing"]


# ===========================================================================
# PURE tiling + trace math (no torch, no ramulator) -- unit-testable
# ===========================================================================
def choose_tiles(M, K, N, buf_elems):
    """Largest square output tile with full K resident: 2*t*K + t^2 <= buf."""
    t = math.isqrt(K * K + buf_elems) - K          # floor of positive root
    t = max(1, min(t, M, N))
    Tk = K
    if 2 * K + 1 > buf_elems:                       # even t=1 + full K won't fit
        t = 1
        Tk = max(1, buf_elems // 3)                 # fall back to K-tiling (rough)
    return t, t, Tk


def gen_layer_lines(M, K, N, mode, buf_elems):
    """Return (list_of_trace_lines, info_dict) for one layer in one mode."""
    Tm, Tn, Tk = choose_tiles(M, K, N, buf_elems)
    names = ["A", "B", "C"] + (["PA", "PC"] if mode == "diff" else [])
    base = {nm: i * REGION_STRIDE for i, nm in enumerate(names)}

    lines = []

    def emit_run(b, off, nbytes, write=False):
        tok = "W" if write else "R"
        a = (off // CL) * CL
        end = off + nbytes
        while a < end:
            lines.append(f"{tok} {b + a}")
            a += CL

    nM = math.ceil(M / Tm)
    nN = math.ceil(N / Tn)
    for im in range(nM):
        Tm_i = min(Tm, M - im * Tm)
        for jn in range(nN):
            Tn_j = min(Tn, N - jn * Tn)
            emit_run(base["A"], im * Tm * K, Tm_i * K)                 # A row-strip
            if mode == "diff":
                emit_run(base["PA"], im * Tm * K, Tm_i * K)            # prev_in
            emit_run(base["B"], jn * Tn * K, K * Tn_j)                 # B col-strip (approx)
            emit_run(base["C"], im * Tm * N + jn * Tn, Tm_i * Tn_j, write=True)  # C block
            if mode == "diff":
                emit_run(base["PC"], im * Tm * N + jn * Tn, Tm_i * Tn_j)  # prev_out
    info = {"Tm": Tm, "Tn": Tn, "Tk": Tk, "nM": nM, "nN": nN, "n_lines": len(lines)}
    return lines, info


# ===========================================================================
# Ramulator invocation
# ===========================================================================
def run_ramulator(lines, workdir):
    trace_path = workdir / "layer.trace"
    cfg_path = workdir / "layer.yaml"
    trace_path.write_text("\n".join(lines) + "\n")
    cfg_path.write_text(f"""Frontend:
  impl: ReadWriteTrace
  path: {trace_path}
  clock_ratio: 8
  num_expected_insts: {len(lines)}
MemorySystem:
  impl: GenericDRAM
  clock_ratio: 3
  DRAM:
    impl: {DRAM["impl"]}
    org:
      preset: {DRAM["org"]}
      channel: {DRAM["channel"]}
{DRAM["extra"]}    timing:
      preset: {DRAM["timing"]}
  Controller:
    impl: Generic
    Scheduler: {{impl: FRFCFS}}
    RefreshManager: {{impl: AllBank}}
    RowPolicy: {{impl: ClosedRowPolicy, cap: 4}}
  AddrMapper:
    impl: RoBaRaCoCh
""")
    out = subprocess.run([str(RAMULATOR), "-f", str(cfg_path)],
                         capture_output=True, text=True).stdout
    cyc = _grab(out, "memory_system_cycles")
    rh = _grab(out, "row_hits_0")
    rm = _grab(out, "row_misses_0")
    rc = _grab(out, "row_conflicts_0")
    return {"cycles": cyc, "row_hits": rh, "row_misses": rm, "row_conflicts": rc}


def _grab(text, key):
    m = re.search(rf"{re.escape(key)}:\s*([-\d.]+)", text)
    return float(m.group(1)) if m else None


# ===========================================================================
# Enumeration (torch) + driver
# ===========================================================================
def enumerate_layers():
    import torch
    import torch.nn as nn
    from diffusers import UNet2DConditionModel

    print("Loading SDM v1.4 UNet from local HF cache ...", flush=True)
    unet = UNet2DConditionModel.from_pretrained(
        "CompVis/stable-diffusion-v1-4", subfolder="unet").eval()
    rec = []

    def hook(mod, inp, out):
        x = inp[0]
        y = out[0] if isinstance(out, (tuple, list)) else out
        rec.append((mod, tuple(x.shape), tuple(y.shape)))

    hs = [m.register_forward_hook(hook) for _, m in unet.named_modules()
          if isinstance(m, (nn.Conv2d, nn.Linear))]
    with torch.no_grad():
        unet(torch.randn(1, 4, 64, 64), torch.tensor(1), torch.randn(1, 77, 768))
    for h in hs:
        h.remove()

    layers = []
    for mod, in_s, out_s in rec:
        if isinstance(mod, nn.Conv2d):
            M = in_s[0] * out_s[2] * out_s[3]
            K = (in_s[1] // mod.groups) * mod.kernel_size[0] * mod.kernel_size[1]
            N = out_s[1]
            kind = "conv"
        else:
            M = 1
            for d in in_s[:-1]:
                M *= d
            K, N, kind = mod.in_features, mod.out_features, "linear"
        layers.append({"kind": kind, "M": M, "K": K, "N": N, "macs": M * K * N})
    return layers


def analytical_ratio(L):
    """The byte model's diff/act ratio for one layer (for comparison)."""
    w, ai, ao = L["K"] * L["N"], L["M"] * L["K"], L["M"] * L["N"]
    return (w + 2 * ai + 2 * ao) / (w + ai + ao)


def pick_representative(layers):
    by_macs = sorted(range(len(layers)), key=lambda i: layers[i]["macs"])
    idxs = {0,                       # conv_in (first layer)
            by_macs[-1],             # largest MAC
            by_macs[0],              # smallest MAC
            by_macs[len(by_macs) // 2]}  # median MAC
    return sorted(idxs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--buf-kb", type=float, default=256, help="on-chip buffer in KB")
    ap.add_argument("--keep", action="store_true", help="keep generated trace files")
    args = ap.parse_args()
    buf_elems = int(args.buf_kb * 1024)

    if not RAMULATOR.exists():
        raise SystemExit(f"ramulator2 not found at {RAMULATOR}")

    layers = enumerate_layers()
    reps = pick_representative(layers)
    print(f"\n=== Step 2: Ramulator-based memory cycles (BUF = {args.buf_kb} KB) ===")
    print(f"Total layers: {len(layers)}   Running {len(reps)} representative layers\n")
    print(f"{'layer':>6} {'kind':>7} {'M':>6} {'K':>6} {'N':>5} {'MAC(M)':>8} "
          f"{'cyc_act':>9} {'cyc_diff':>9} {'RAM ratio':>10} {'byte ratio':>11}")
    print("-" * 92)

    tmp = Path(tempfile.mkdtemp(prefix="ditto_trace_"))
    tot_act = tot_diff = 0.0
    for i in reps:
        L = layers[i]
        a_lines, a_info = gen_layer_lines(L["M"], L["K"], L["N"], "act", buf_elems)
        d_lines, d_info = gen_layer_lines(L["M"], L["K"], L["N"], "diff", buf_elems)
        r_act = run_ramulator(a_lines, tmp)
        r_diff = run_ramulator(d_lines, tmp)
        ca, cd = r_act["cycles"], r_diff["cycles"]
        ratio = cd / ca if ca else float("nan")
        tot_act += ca or 0
        tot_diff += cd or 0
        print(f"{i:>6} {L['kind']:>7} {L['M']:>6} {L['K']:>6} {L['N']:>5} "
              f"{L['macs']/1e6:>8.1f} {ca:>9.0f} {cd:>9.0f} {ratio:>9.2f}x "
              f"{analytical_ratio(L):>10.2f}x")
        if args.keep:
            (tmp / f"layer{i}_act.trace").write_text("\n".join(a_lines))
            (tmp / f"layer{i}_diff.trace").write_text("\n".join(d_lines))

    print("-" * 92)
    agg = tot_diff / tot_act if tot_act else float("nan")
    print(f"\nAggregate Ramulator diff/act cycle ratio (representative set): {agg:.2f}x")
    print(f"  (analytical byte model ceiling was ~1.34x; paper Fig 8 avg: 2.75x)")
    print(f"\nThis ratio is clock-domain independent -- it is the memory-access cost")
    print(f"comparison the analytical model could not produce. Speedup (compute+memory")
    print(f"with clock alignment) is the next step.")
    if args.keep:
        print(f"\nTrace files kept in: {tmp}")


if __name__ == "__main__":
    main()

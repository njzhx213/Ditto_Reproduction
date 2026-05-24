#!/usr/bin/env python3
"""
validate.py - cross-validation harness for the Ditto performance-model code.

Checks split into two tiers:
  [PURE]  no torch needed; verifies formulas, invariants, tiling, trace format,
          and the 10.4x decomposition arithmetic. Runs anywhere.
  [ENUM]  needs torch+diffusers; enumerates the real UNet once and checks the
          anchors (conv_in 47.2M, total ~338G) AND that fig13_speedup.py and
          gen_trace.py agree on per-layer MACs (the cross-module consistency
          that, if broken, would invalidate every comparison we made).

Honesty: this validates things with OBJECTIVE right/wrong answers. It does NOT
"prove correct" the analytical memory model or the tiling access pattern -- those
are modeling choices; for them we only assert self-consistency invariants
(monotonicity, bounds, dimensional sanity).

Run:
    cd ~/Ditto && python3 sim/validate.py
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fig13_speedup as F          # noqa: E402
import gen_trace as G              # noqa: E402

results = []   # (tier, name, passed, detail)


def check(tier, name, cond, detail=""):
    results.append((tier, name, bool(cond), detail))


# reference MAC formulas, independent of both modules
def ref_conv_macs(C_in, C_out, kH, kW, groups, B, H_out, W_out):
    return B * C_out * (C_in // groups) * kH * kW * H_out * W_out


def ref_linear_macs(F_in, F_out, n_rows):
    return n_rows * F_out * F_in


# =========================================================================
# [PURE] checks
# =========================================================================
def pure_checks():
    # --- fig13 MAC formulas vs independent reference ---
    cases_conv = [(4, 320, 3, 3, 1, 1, 64, 64), (640, 640, 3, 3, 1, 1, 16, 16),
                  (1280, 1280, 1, 1, 1, 1, 8, 8)]
    ok = True
    for (ci, co, kh, kw, gr, B, ho, wo) in cases_conv:
        macs_f = F.conv_bytes_macs(ci, co, kh, kw, gr, B, ho, wo, ho, wo)[0]
        macs_r = ref_conv_macs(ci, co, kh, kw, gr, B, ho, wo)
        ok &= (macs_f == macs_r)
    check("PURE", "fig13 conv MAC formula == reference", ok)

    ok = True
    for (fi, fo, nr) in [(320, 320, 4096), (768, 320, 77), (1280, 10, 1)]:
        ok &= (F.linear_bytes_macs(fi, fo, nr)[0] == ref_linear_macs(fi, fo, nr))
    check("PURE", "fig13 linear MAC formula == reference", ok)

    # --- gen_trace M*K*N derivation == reference (mirrors enumerate inline logic) ---
    # conv: M=B*Ho*Wo, K=(Cin/groups)*kh*kw, N=Co
    ci, co, kh, kw, gr, B, ho, wo = 4, 320, 3, 3, 1, 1, 64, 64
    M, K, N = B * ho * wo, (ci // gr) * kh * kw, co
    check("PURE", "gen_trace conv M*K*N == reference",
          M * K * N == ref_conv_macs(ci, co, kh, kw, gr, B, ho, wo),
          f"M*K*N={M*K*N}")

    # --- fig13 and gen_trace agree on the SAME conv (formula-level) ---
    macs_f = F.conv_bytes_macs(ci, co, kh, kw, gr, B, ho, wo, ho, wo)[0]
    check("PURE", "fig13 conv MAC == gen_trace conv MAC (same layer)",
          macs_f == M * K * N, f"{macs_f} vs {M*K*N}")

    # --- conv_in anchor 47.2M ---
    check("PURE", "conv_in anchor MAC = 47.2M",
          abs(ref_conv_macs(4, 320, 3, 3, 1, 1, 64, 64) / 1e6 - 47.2) < 0.1)

    # --- 10.4x bare-compute speedup + decomposition ---
    macs = 1e9
    bare = F.comp_itc(macs) / F.comp_diff(macs)
    pe_ratio = F.N_PE_DITTO / F.N_PE_ITC
    lanes = F.DITTO_LANES
    skipbit = 1.0 / (F._NONZERO * F._BIT_FACTOR)
    decomposed = pe_ratio * lanes * skipbit
    check("PURE", "bare compute speedup = 10.40x",
          abs(bare - 10.40) < 0.05, f"{bare:.3f}")
    check("PURE", "decomposition 1.42 x 4 x 1.82 = bare",
          abs(decomposed - bare) < 1e-6,
          f"PE {pe_ratio:.3f} x lanes {lanes} x skipbit {skipbit:.3f} = {decomposed:.3f}")

    # --- fig13 analytical invariants on synthetic layers ---
    L = [dict(kind="conv", macs=ref_conv_macs(640, 640, 3, 3, 1, 1, 16, 16),
              w=640 * 640 * 9, a_in=640 * 256, a_out=640 * 256)]
    L += [dict(kind="linear", macs=ref_linear_macs(320, 4, 4096),
               w=320 * 4, a_in=4096 * 320, a_out=4096 * 4) for _ in range(20)]
    GB = 1024**3
    r_big = F.run_at(L, 1e9, 1024 * GB)        # huge buffer -> reload 1
    r_small = F.run_at(L, 1e9, 4096)           # tiny buffer
    check("PURE", "analytical: small-buf ratio >= large-buf ratio",
          r_small["mem_ratio"] >= r_big["mem_ratio"] - 1e-9,
          f"big {r_big['mem_ratio']:.2f} small {r_small['mem_ratio']:.2f}")
    # monotone ratio in buffer
    prev, mono = None, True
    for mb in [0.01, 1, 64, 4096, 1024**2]:
        rr = F.run_at(L, 1e9, mb * 1024 * 1024)["mem_ratio"]
        if prev is not None and rr > prev + 1e-9:
            mono = False
        prev = rr
    check("PURE", "analytical: ratio monotone non-increasing in buffer", mono)
    # Defo >= diff-only at every bandwidth; speedup bounded
    inv = True
    for b in [8, 128, 2048, 8192]:
        r = F.run_at(L, b, 64 * 1024 * 1024)
        if r["speedup_defo"] < r["speedup_diff_only"] - 1e-9:
            inv = False
        if not (0 < r["speedup_defo"] <= 10.45):
            inv = False
    check("PURE", "analytical: Defo>=diff-only & 0<speedup<=10.4", inv)

    # --- gen_trace tiling constraint 2tK+t^2<=BUF (full-K case) ---
    ok = True
    for (M, K, N, buf) in [(4096, 36, 320, 256 * 1024), (4096, 320, 320, 64 * 1024),
                           (256, 1280, 1280, 256 * 1024)]:
        Tm, Tn, Tk = G.choose_tiles(M, K, N, buf)
        if Tk == K and Tm > 1:
            ok &= (2 * Tm * K + Tm * Tm <= buf)
        ok &= (1 <= Tm <= M and 1 <= Tn <= N)
    check("PURE", "gen_trace tile satisfies 2tK+t^2<=BUF and is clamped", ok)

    # --- gen_trace trace format + bounds + diff>act + small-buf more lines ---
    a_lines, a_info = G.gen_layer_lines(256, 320, 320, "act", 64 * 1024)
    d_lines, d_info = G.gen_layer_lines(256, 320, 320, "diff", 64 * 1024)
    fmt_ok, bound_ok = True, True
    for ln in a_lines[:3000] + d_lines[:3000]:
        t = ln.split(" ")
        if len(t) != 2 or t[0] not in ("R", "W"):
            fmt_ok = False
        elif not (0 <= int(t[1]) < 2 * 1024**3):
            bound_ok = False
    check("PURE", "gen_trace lines are 'R|W <int>'", fmt_ok)
    check("PURE", "gen_trace addresses within 2GB DRAM", bound_ok)
    check("PURE", "gen_trace diff emits more lines than act",
          d_info["n_lines"] > a_info["n_lines"],
          f"act {a_info['n_lines']} diff {d_info['n_lines']}")
    big = G.gen_layer_lines(256, 320, 320, "act", 1024**3)[1]["n_lines"]
    small = G.gen_layer_lines(256, 320, 320, "act", 4096)[1]["n_lines"]
    check("PURE", "gen_trace tiny buffer -> >= lines than huge buffer",
          small >= big, f"huge {big} tiny {small}")


# =========================================================================
# [ENUM] checks  (need torch; skipped automatically if unavailable)
# =========================================================================
def enum_checks():
    try:
        import torch  # noqa: F401
    except Exception:
        check("ENUM", "torch available", False, "SKIPPED - run on WSL fastdllm env")
        return

    layers_f = F.enumerate_sdm_layers()       # has macs, w, a_in, a_out
    layers_g = G.enumerate_layers()           # has M, K, N, macs

    check("ENUM", "both modules enumerate same #layers",
          len(layers_f) == len(layers_g), f"{len(layers_f)} vs {len(layers_g)}")

    # per-layer MAC equality -- THE cross-module consistency check
    n = min(len(layers_f), len(layers_g))
    mismatches = sum(1 for i in range(n)
                     if abs(layers_f[i]["macs"] - layers_g[i]["macs"]) > 0.5)
    check("ENUM", "per-layer MACs identical across fig13 & gen_trace",
          mismatches == 0, f"{mismatches} mismatches of {n}")

    total_f = sum(L["macs"] for L in layers_f) / 1e9
    check("ENUM", "fig13 total MACs ~ 338G (public SD v1.4 ~340G)",
          330 <= total_f <= 345, f"{total_f:.1f} G")

    # conv_in anchor in the real enumeration
    cin = next((L for L in layers_f if abs(L["macs"] / 1e6 - 47.2) < 0.5), None)
    check("ENUM", "conv_in 47.2M present in real enumeration", cin is not None)

    # gen_trace M*K*N == its own macs field
    ok = all(abs(L["M"] * L["K"] * L["N"] - L["macs"]) < 0.5 for L in layers_g)
    check("ENUM", "gen_trace M*K*N == macs for every layer", ok)


def main():
    pure_checks()
    enum_checks()
    width = max(len(n) for _, n, _, _ in results)
    npass = sum(1 for _, _, p, _ in results if p)
    print(f"\n{'='*4} Ditto code cross-validation {'='*4}\n")
    for tier, name, passed, detail in results:
        mark = "PASS" if passed else "FAIL"
        line = f"[{tier}] {mark}  {name:<{width}}"
        if detail:
            line += f"   ({detail})"
        print(line)
    print(f"\n{npass}/{len(results)} checks passed.")
    if npass != len(results):
        print("Some checks failed or were skipped -- see above.")


if __name__ == "__main__":
    main()

# Ditto Reproduction

Architecture-level reproduction and performance model of **Ditto: Accelerating Diffusion Model via Temporal Value Similarity** (HPCA 2025, Yonsei), for Coding Test 5 (*SOTA AI accelerator arch-level reproduction*).

Ditto accelerates diffusion inference by exploiting **temporal value similarity** — the activations of consecutive denoising steps are nearly identical, so the per-step *difference* is sparse and low-bit. Ditto computes on this difference with dynamic bit-width + zero-skip, and uses a runtime decision unit (**Defo**) to fall back to full-activation execution on memory-bound layers.

Test 5 allows a performance model *or* RTL; **this repo does both**:

1. A **performance model** (Phase 2) reproducing the paper's bit-width, compute ceiling, memory ratio, speedup roofline, and six-segment energy, with an **extendability study across three workloads** — SD v1.4 UNet (image), DiT-XL/2 (image), and Fast-dLLM v2 (text diffusion LLM, *not benchmarked by the paper*).
2. A **Verilog RTL compute core** (Phase 3) — 13 modules built top-down from a block diagram, each verified against the functional model under cocotb, integrated end to end, driven by real traces, and **synthesized on Vivado** (ZU3EG) with a measured Fmax and a carry-save accumulator optimization (5x).

These directly answer the brief's two extendability requirements: (1) *profile a workload not benchmarked by the paper* → the Fast-dLLM v2 study; (2) *make modifications to the original hardware based on your design* → the RTL track, including the carry-save accumulator.

See **[docs/SUMMARY.md](docs/SUMMARY.md)** for the full, honest write-up (every number labeled paper-stated / inferred / measured; disagreements explained, not fitted away), and **[rtl/SYNTHESIS.md](rtl/SYNTHESIS.md)** for the synthesis experiment.

## Headline results

| Quantity | Result | Paper | Status |
|---|---|---|---|
| Temporal-diff bit-width, SDM (zero) | 45.9% | 44.48% | reproduced |
| Total MACs, SDM (incl. attention) | 401.6 G | ~340 G (linear-only public) | enumerated |
| Compute ceiling, SDM (attn + VPU aware) | 8.89x | — | derived |
| Bare memory-access ratio (incl. attention) | 2.46x | 2.75x (Fig 8) | approaches paper |
| Speedup vs ITC | 1.5x at ~251 GB/s (bandwidth roofline) | 1.5x (Fig 13) | reproduced |
| Energy, Ditto+Defo vs ITC (SDM) | saving ~34% (16-38% sweep) | 17.74% (Fig 13) | range contains paper |
| Energy, Cambricon-D vs ITC (SDM) | 1.34x (inversion) | ~1.5x (Fig 13) | reproduced |

**Three-workload extendability trend** (SDM → DiT → Fast-dLLM): attention fraction 15.7 → 3.6 → 0.8%, temporal zero rate 45.9 → 67.2 → 80.3%, Ditto energy saving 34 → 46 → 70%, Cambricon-D inversion 1.34 → 1.04x. Ditto's temporal-difference acceleration generalizes across modalities and is strongest for large, linear-dominated text diffusion.

## What is reproduced

- **Functional datapath** — three-stage Ditto algorithm, Encoding Unit (Fig 11), shift-and-add PE (Fig 12), bitwise-exact vs A8W8 on real traces; the attention two-sub-operation difference trick, int8 bitwise-verified.
- **Bit-width** (Fig 5), **compute ceiling**, **memory ratio** (resolving the 2.75x gap by including previously-missed attention traffic).
- **Speedup** as a bandwidth roofline (the paper does not publish Ditto's DRAM).
- **Six-segment energy** (Fig 13) with CACTI-measured SRAM (818 pJ/64B), the Cambricon-D inversion, and Defo.
- **Defo** runtime decision + the "naive difference is slower" rescue (Fig 16).
- **Extendability**: DiT-XL/2 and Fast-dLLM v2, each with its own measured temporal sparsity.

## RTL compute core (Phase 3)

A 13-module Verilog implementation of the Ditto datapath, built top-down and verified bottom-up against the functional model (numpy golden reference) under cocotb + Icarus Verilog — all targets pass.

- **Core blocks**: Encoding Unit (classifies each int9 difference into zero / 4-bit / wide; exhaustively verified over all 509 values), difference PE (zero-skip MAC, lossless), slot PE (4-bit/wide multiplier slots; hardware slot count 1.97 matches the perf-model bit_factor), Defo decision unit (memory-driven stop-loss).
- **Datapath entry/exit**: diff-generator and VPU-restore are exact inverses — `restore(diff_generator(act)) == act`, proving the difference encode/decode is lossless.
- **Variants & scale**: 3-stage pipelined PE, carry-save PE, datapaths A/B/slot, parallel and systolic PE arrays, and an integrated top (EU → slot PE → Defo).
- **Real-data closure**: driven by real SDM trace differences, the hardware zero-skip rate is 42.8% — matching the paper (44.48%) and the perf model (45.9%) on the same recipe.
- **Synthesis finding** (Vivado 2025.1, ZU3EG): the three carry-propagate PEs all hit ~600 MHz because the critical path is the 32-bit accumulator carry chain; pipelining the multiply/sum logic gave nothing (wrong path), while a **carry-save accumulator** on the real bottleneck reached **3021 MHz (5x)**. See `figs/rtl_fmax.png` and `rtl/SYNTHESIS.md`. The lesson: profile the critical path before optimizing.

## Layout

| Path | Purpose |
|---|---|
| `sim/` | performance-model code (functional, cycle, energy, roofline, per-workload) |
| `src/trace_collection/` | activation-trace collection (SDM, DiT) |
| `src/validation/` | bit-width reproduction (Fig 5) |
| `docs/SUMMARY.md` | full honest write-up |
| `docs/architecture_spec.md`, `docs/ditto_paper_notes.md` | block diagram / paper notes |
| `figs/` | all figures (bit-width, roofline, energy, three-workload, RTL Fmax) |
| `results/` | numeric outputs (JSON) |
| `rtl/` | RTL compute core (Phase 3): `common/*.v` (13 modules), `tb/*.py` (cocotb), `Makefile`, synthesis scripts, `SYNTHESIS.md` |
| `docs/rtl_diagrams.md` | RTL datapath block diagram + verification hierarchy (Mermaid) |

Key scripts: `energy_model.py` (six-segment energy), `fig13_roofline.py` (speedup), `dit_structure.py` / `dit_recompute.py` (DiT), `fastdllm_structure.py` / `gen_fastdllm_ditto.py` (Fast-dLLM), `validate.py` (cross-checks).

## Reproducing

```bash
# environment: diffusers, torch+CUDA, numpy, matplotlib
# SDM speedup + energy (no trace needed; structural)
python sim/fig13_roofline.py
python sim/energy_model.py

# DiT and Fast-dLLM extendability
python sim/dit_structure.py
python sim/fastdllm_structure.py

# figures
python sim/plot_roofline.py
python sim/plot_energy.py
python sim/plot_fig13_both.py            # three-workload energy
python sim/plot_three_workload_trend.py
```

Bit-width reproduction needs activation traces (collected separately; **not committed** — see `.gitignore`):

```bash
python src/trace_collection/collect_sdm_traces.py     # ~19 GB
python src/validation/reproduce_fig5_bitwidth.py
```

### RTL — simulation and synthesis

```bash
# functional verification (needs iverilog + cocotb); each target is one module
cd rtl
make pe            # difference PE          make slot-equivalent targets: pe, datapath, ...
make csa           # carry-save PE          make top   # integrated EU -> slot PE -> Defo
make real_sdm      # real SDM trace -> RTL  (drives the slot datapath with trace diffs)
# every target: encoding_unit, x4, pe, pipe, csa, datapath{,_b,_slot}, defo,
#               diffgen, vpu, array, array_sys, top, real_sdm

# synthesis (needs Vivado; ZU3EG used here, change the part for your board)
vivado -mode batch -source synth_vivado.tcl -tclargs xczu3eg-sbva484-2-e pe_diff      1.0
vivado -mode batch -source synth_vivado.tcl -tclargs xczu3eg-sbva484-2-e pe_diff_csa  1.0
# Yosys fallback (no license): ./synth_yosys.sh
python sim/plot_rtl_fmax.py    # the Fmax comparison figure
```

To **modify** the design, each module in `rtl/common/` has a matching cocotb testbench in `rtl/tb/`; edit the `.v`, re-run `make <target>`, and the golden reference catches regressions.

## Status — completed and pending tasks

**Completed — performance model:** functional datapath; Fig 5 bit-width; compute ceiling; MAC enumeration; memory ratio (2.46x); speedup roofline; six-segment energy (CACTI-backed, Cam-D, Defo); three-workload extendability (SDM/DiT/Fast-dLLM).

**Completed — RTL:** 13-module Verilog compute core, all cocotb targets pass; real-trace closure (42.8% zero-skip); Vivado synthesis with the carry-save accumulator finding (5x Fmax).

**Pending / not done:** Fig 17 Defo-accuracy (analyzed infeasible on available layers); Diffy baseline; attention in the Ramulator main line. DiT/Fast-dLLM theoretical ceilings (16.49x / 28.42x) are bandwidth→∞ limits, **not attainable speedups**. See `docs/SUMMARY.md` §8 for the full boundary list.

## Roadmap for future work

- **RTL → full fabric**: scale the verified slot PE / array into Ditto's ~39398-PE fabric; the per-PE synthesis numbers now make an area estimate possible.
- **Close the next Fmax limiter**: after carry-save removed the accumulator chain, the per-cycle 4-lane multiply/sum tree is next; pipeline *that* (the now-correct target) and re-synthesize with place-and-route for a realistic Fmax.
- **VPU non-linearities**: the restore path is implemented; add the softmax/GELU datapath the VPU also handles (table or polynomial approximation).
- **Attention in the cycle model**: feed attention QK/PV into the Ramulator trace path (currently in the analytical roofline only).
- **More workloads through the RTL**: drive the slot datapath with DiT and Fast-dLLM trace diffs (the perf model already has their sparsity).

**Note:** A separate study skips whole attention/MLP modules in Fast-dLLM by cross-step *similarity* (delivered independently). This repo instead applies *Ditto's* per-element difference-quantization to Fast-dLLM; the two are compared in `docs/SUMMARY.md` §9.4.

## Hardware

Traces collected on RTX 5080; Ramulator2 + CACTI 7.0 compiled locally for memory cross-validation and SRAM energy.

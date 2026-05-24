# Ditto Reproduction

Architecture-level reproduction and performance model of **Ditto: Accelerating Diffusion Model via Temporal Value Similarity** (HPCA 2025, Yonsei), for Coding Test 5 (*SOTA AI accelerator arch-level reproduction*).

Ditto accelerates diffusion inference by exploiting **temporal value similarity** — the activations of consecutive denoising steps are nearly identical, so the per-step *difference* is sparse and low-bit. Ditto computes on this difference with dynamic bit-width + zero-skip, and uses a runtime decision unit (**Defo**) to fall back to full-activation execution on memory-bound layers.

This repo takes the **performance-model path** (Test 5 allows perf-model *or* RTL), and adds an **extendability study across three workloads**: SD v1.4 UNet (image), DiT-XL/2 (image), and Fast-dLLM v2 (text diffusion LLM, *not benchmarked by the paper*).

See **[docs/SUMMARY.md](docs/SUMMARY.md)** for the full, honest write-up (every number labeled paper-stated / inferred / measured; disagreements explained, not fitted away).

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

## Layout

| Path | Purpose |
|---|---|
| `sim/` | performance-model code (functional, cycle, energy, roofline, per-workload) |
| `src/trace_collection/` | activation-trace collection (SDM, DiT) |
| `src/validation/` | bit-width reproduction (Fig 5) |
| `docs/SUMMARY.md` | full honest write-up |
| `docs/architecture_spec.md`, `docs/ditto_paper_notes.md` | block diagram / paper notes |
| `figs/` | all figures (bit-width, roofline, energy, three-workload) |
| `results/` | numeric outputs (JSON) |
| `rtl/` | RTL track (Phase 3, optional/in progress) |

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

## Status

**Done:** functional datapath; Fig 5 bit-width; compute ceiling; MAC enumeration; memory ratio (2.46x); speedup roofline; six-segment energy (CACTI-backed, Cam-D, Defo); three-workload extendability (SDM/DiT/Fast-dLLM).

**Boundaries / not done:** Fig 17 Defo-accuracy (analyzed infeasible on available layers); Diffy baseline; attention in the Ramulator main line; RTL (Phase 3, optional — independent track in `rtl/`). DiT/Fast-dLLM theoretical ceilings (16.49x / 28.42x) are bandwidth→∞ limits, **not attainable speedups**. See `docs/SUMMARY.md` §8 for the full boundary list.

**Note:** A separate study skips whole attention/MLP modules in Fast-dLLM by cross-step *similarity* (delivered independently). This repo instead applies *Ditto's* per-element difference-quantization to Fast-dLLM; the two are compared in `docs/SUMMARY.md` §9.4.

## Hardware

Traces collected on RTX 5080; Ramulator2 + CACTI 7.0 compiled locally for memory cross-validation and SRAM energy.

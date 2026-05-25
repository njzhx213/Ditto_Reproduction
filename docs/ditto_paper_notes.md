# Ditto Paper Notes

Key facts and equations extracted from Ditto (HPCA 2025) so we don't have to re-read the paper during implementation.

## I. Core observation (Section II–III)

- **Temporal cosine similarity** between adjacent denoising steps across 7 diffusion models: average **0.983**, minimum 0.947 (per-model average). Always > 0.94 in the SDM `conv-in` and `up.0.0.skip` layers.
- **Spatial similarity** inside layers (Diffy's approach) is only **0.31** on average. Temporal >> spatial.
- **Value range of temporal differences** is **8.96× narrower** than original activations on average (range 2.44× CHUR to 25.02× DDPM).

## II. Bit-width breakdown — Fig 5 reproduction target

Quantization: A8W8 (8-bit activation, 8-bit weight). Bit-width requirement is the minimum bits needed to represent each data element.

Average over all 7 models (Fig 5):
- Original activations: zero ~18%, 4-bit ~40%, >4-bit ~42%
- Spatial differences: zero ~26%, 4-bit ~48%, >4-bit ~26%
- **Temporal differences: zero 44.48%, 4-bit 51.52% (excl. zero), >4-bit 3.99%**
  - Equivalently: ≤4-bit = 96.01%, > 4-bit = 3.99%.

**Implication for hardware:** the baseline multiplier should be 4-bit × 8-bit (data × weight); 8-bit operands use two multipliers + shift. Outlier PEs are unnecessary because >4-bit data is rare enough that we can shift-and-add through the existing 4-bit MAC.

## III. BOPs reduction (Fig 6)

- Temporal diff vs original: **53.3% BOPs reduction** on average.
- Temporal diff vs spatial diff: 23.1% reduction.
- Consistent across all time steps (Fig 6b) except the last few steps (which need more denoising).

## IV. Algorithm (Section IV)

### Linear layers (Conv, FC) — Fig 7
```
act_t × W = (act_{t+1} + δ_t) × W = act_{t+1} × W + δ_t × W
                                    ────────────   ─────────
                                    use cached    new compute (low-bit + zero-skip)
```
Three stages:
1. **Calculate diff**: δ_t = act_t − act_{t-1}; classify each element as zero / 4-bit / >4-bit.
2. **Execute**: δ × W using mixed-precision MAC (zero-skip + low-bit).
3. **Sum**: out_t = out_{t-1} + (δ × W).

### Attention layers (Q×K, P×V)
```
Q_t × K_t = (Q_{t+1} + ΔQ)(K_{t+1} + ΔK)
          = Q_{t+1} × K_{t+1} + Q_{t+1}×ΔK + ΔQ×K_{t+1} + ΔQ×ΔK
```
Rewrite to **2 sub-ops** instead of 4: `Q_t × ΔK + ΔQ × K_{t+1}` (treat Q_t and K_{t+1} as weights). Same for P × V.

**Cross-attention special case**: in conditional models (IMG, SDM), the context (K, V in cross-attn) is identical across time steps. Treat them as weights → no diff needed.

### Defo (execution flow optimization) — Fig 9
- **Static analysis**: traverse the computational graph, identify non-linear functions, check layer dependencies, bypass diff calc/summation where unnecessary.
- **First time step**: execute all layers with original activations. Record `Cycle_act` per layer in a 512-entry table.
- **Second time step**: execute all layers with diff processing. Record `Cycle_diff` per layer.
- **For each layer**: if `Cycle_act > Cycle_diff`, set `ExeDiff = True` for all subsequent time steps. Otherwise stay on original activation execution.
- **Defo+ extension**: layers tagged "original activation" can still benefit from *spatial* difference processing (Diffy-style) since spatial diff has no memory overhead.
- **Defo accuracy on SDM**: 92% (correctly picks the optimal flow per layer).

## V. Hardware (Section V)

### Configuration (Table III)
- **39398 PEs** (iso-area to ITC's 27648 PEs and Diffy's 39398 PEs)
- **A4W8** multipliers in the PE (4-bit data × 8-bit weight, two used for 8-bit data via shift)
- **Power**: 33.6 W
- **SRAM**: 192 MB on-chip
- **Area**: 64.48 mm²
- **Frequency**: 1 GHz
- **Tech**: FreePDK 45nm (Synopsys DC for core, CACTI for memory)

### Encoding Unit — Fig 11

```
Inputs:  prev_act[i]  (8-bit signed)
         curr_act[i]  (8-bit signed)

Stage 1: Subtract
  diff[i] = curr_act[i] - prev_act[i]   (signed 8-bit, may be -128..127)

Stage 2: Classify into zero / 4-bit / 8-bit
  high_4bit = diff[i][7:4]              (sign-extended upper nibble)
  low_4bit  = diff[i][3:0]              (lower nibble)

  ctrl = {high_part_nonzero, low_part_nonzero}   (2-bit control signal)

  ctrl == 00: zero → skip
  ctrl == 01: low 4-bit only → enqueue low part
  ctrl == 1X: high part present (full 8-bit) → enqueue both parts + metadata

Stage 3: Reorder + enqueue
  Skip zeros.
  For 4-bit: enqueue low_4bit + metadata=0 (no shift needed)
  For 8-bit: enqueue (low_4bit, metadata=0) AND (high_4bit, metadata=1)
             metadata=1 tells the PE to shift left by 4 after multiplication.
  Output: 4×4-bit data per cycle into PE input queue.
```

Latency: subtraction + comparison in 1 cycle; enqueue in 1 cycle. Total Encoding Unit latency ≈ 2 cycles.

**(Implementation note: the [-8,7] = 4-bit definition requires a sign-magnitude datapath, not a literal two's-complement nibble check — see `docs/architecture_spec.md` and the exhaustive RTL test.)**

### Compute Unit (PE) — Fig 12

```
Each PE:
  - 4 multipliers (4-bit data × 8-bit weight)
  - Adder tree (3 levels: 4→2→1)
  - 2 shifters at the first adder stage (one per pair of multipliers)
  - Partial sum register (for 8-bit data, accumulates high+low parts)

Per cycle:
  Inputs: 4 × 4-bit data + 4 × 8-bit weight + 4 × 1-bit metadata flag
  For each multiplier i: prod[i] = data[i] * weight[i]
  Shifter: if metadata[i]: prod[i] <<= 4
  Adder tree: sum = prod[0..3]
  Accumulator: psum_reg += sum
```

Throughput: 4 elements per cycle per PE. With 39398 PEs and 4-bit operands → 157.6 TOPS at 1 GHz.

### VPU
- Non-linear functions (SiLU, GeLU, Softmax, LayerNorm, GroupNorm), quant/dequant, summation of (δ × W) + out_{t-1}. Pipelined with Compute Unit. 2.9% of total energy, ~0.17% of latency.

### Defo Unit
- 512 entries × 33-bit (16-bit Cycle_act + 16-bit Cycle_diff + 1-bit decision). Comparator + control. 0.01% of area.

## VI. Evaluation methodology (Section VI-A)

- **Simulator**: open-source Sparse-DySta cycle-accurate simulator (IROS '23), modified to detect zero / low-bit / full-bit temporal differences from PyTorch hooks.
- **Baselines**: ITC (Tensor-Core-like integer MAC), Diffy (spatial diff), Cambricon-D (temporal diff with outlier PEs).
- **Iso-area**: all hardware at same SRAM size, same frequency.
- **Workloads**: 7 models (Table I). SDM uses Stable-Diffusion COCO2017 PLMS 50-step.
- **Accuracy**: Table II FID/IS/CS for FP32 vs Ditto-quantized — quality preserved.

## VII. What we reproduced (vs the paper's figures)

Reproduced from this paper (see `docs/SUMMARY.md` for evidence and honest boundaries):
1. **Fig 5 SDM bit-width**: zero / ≤4-bit / >4-bit for temporal differences — reproduced (45.9% zero vs paper 44.48%).
2. **Fig 13 SDM speedup/energy**: speedup as a bandwidth roofline reaching the paper's 1.5×; six-segment energy with the Cambricon-D inversion (1.34×) and Ditto saving (34%).
3. **Fig 16 Defo rescue**: difference-only drops below ITC under tight memory; Defo holds ≥1.0×.

Analyzed but found infeasible on our data:
- **Fig 17 Defo accuracy (92%)**: our trace covers only compute-heavy layers, so `diff` wins regardless and Defo never flips (degenerate ~100%); needs memory-bound layers we structurally lack. Recorded as analyzed infeasibility (SUMMARY §8), not a target silently dropped.

Extended **beyond** the paper's scope:
- **DiT-XL/2** (image diffusion transformer) and **Fast-dLLM v2** (7B text diffusion LLM, not benchmarked by the paper) run through the same model — the extendability requirement (SUMMARY §9). So the earlier "SDM-only" plan was extended: DiT and Fast-dLLM are done.
- **RTL compute core** + Vivado synthesis (SUMMARY §10) — the paper's PPA is reproduced in spirit (carry-save accumulator finding), not just modeled.

## VIII. Things the paper does not explicitly specify (we had to choose)

- **Quantization calibration**: paper uses Q-Diffusion's calibration; we use dynamic per-tensor A8W8. The >4-bit fraction is calibration-dependent (dynamic → ~0%, tighter → toward the paper's ~4%); flagged throughout, not fitted.
- **DRAM bandwidth / size**: paper's Table III has **no DRAM row**. We do **not** assume a single value — speedup is a **bandwidth roofline** (the paper's 1.5× is reached at ~251 GB/s; its 14.4% Defo-flip reverse-estimates the DRAM to ~3.2 TB/s). (An earlier draft of these notes assumed 1.5 TB/s; that was superseded by the roofline — see SUMMARY §3.)
- **Encoding Unit queue depth**: not specified; sized to match the PE throughput (4 × 4-bit/cycle).
- **PE clustering**: only the total 39398 PEs is given, not the grouping; the RTL builds a representative 4×4 tile.

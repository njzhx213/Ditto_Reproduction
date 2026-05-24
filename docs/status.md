# Ditto Reproduction — Project Status

Last updated: end of Week 1, Day 2.

This is an **honest** status: it separates what we have rigorously verified from
what remains, and states the degree of paper-faithfulness for each component.
The goal is that anyone (including the advisor) can see the true progress at a
glance without being misled by optimistic phrasing.

---

## 0. Project goal (scope definition)

Per the task brief: *"emphasis on thought process, architecture, and roadmap;
completion is not required."* So the target is **not** a bit-exact reproduction
of every number in the paper. The target is:

1. Faithfully implement the Ditto algorithm (3-stage temporal difference processing).
2. Verify the core claims that make Ditto work (temporal similarity → low bit-width).
3. Reproduce the key figures' **trends** (Fig 5 bit-width split, Fig 13 ~1.5× speedup) within reason.
4. Demonstrate deep understanding, including implementation details the paper leaves implicit.

Everything below is graded against that scope.

---

## 1. What is rigorously verified ✅

### 1.1 Algorithm correctness (paper-faithful)
- The 3-stage Ditto linear-layer algorithm (δ = act_t − act_{t-1}; δ×W; out_{t-1}+δ×W) is implemented exactly as paper Section IV-A / Fig 7.
- The distributive-property identity `out_{t-1} + δ×W == act_t × W` is verified **bitwise-exact** on:
  - synthetic int8 data (256 elements, all-exact)
  - **real SDM v1.4 activations** (6 representative layers, step transitions 2→3 / 25→26 / 48→49, 64 rows each — every row bitwise-exact)
- The Encoding Unit (Fig 11) nibble split + PE (Fig 12) shift-and-add reconstruction verified exact over all 256 int8 values.

### 1.2 Core claim verified (paper-faithful)
- **Temporal cosine similarity** of our real SDM traces: 0.999 (conv_in), 0.994–0.998 (attention) — matches paper Fig 3 (~0.98).
- **Zero-difference ratio** (the headline of Ditto's sparsity gain): our **45.7%** vs paper's **44%** on SDM (step 25→26 average over 6 layers). This is the single most important number and it matches.

### 1.3 Implementation details reverse-engineered (beyond the paper)
Three details the paper does not state explicitly, each backed by our data:

1. **Per-tensor (not per-row) quantization scale.** Per-row scale is unstable on
   activations with outliers and inflates the >4-bit fraction to ~43%. Per-tensor
   scale (one scale per layer activation) is required, matching the paper's
   "dynamic quantization."

2. **Signed-range (sign-magnitude) bit-width classification.** Paper Fig 5 counts
   a value in [−8, 7] as 4-bit (minimum signed representation). A literal
   two's-complement nibble-zero check (Fig 11's wording) misclassifies small
   negatives (−1..−7, high nibble 0xF) as 8-bit, dropping ≤4-bit from 96% to 73%.
   The Encoding Unit must use a sign-magnitude datapath. After this fix, ≤4-bit ≈ 100%.

3. **Calibration tightness controls the >4-bit fraction.** With true-absmax dynamic
   quant, >4-bit ≈ 0%. Tightening the scale (clipping outliers, simulating
   Q-Diffusion calibration) raises >4-bit monotonically — e.g. down2_attn reaches
   5.7% at the 98th-percentile absmax. The paper's ~4% corresponds to an
   intermediate calibration tightness. Our zero ratio already matches; the >4-bit
   gap is purely a calibration choice, not an implementation error.

---

## 2. What remains, and faithfulness grade

| Component | Status | Paper-faithful? | Notes |
|---|---|---|---|
| 3-stage algorithm | ✅ done | ✅ faithful | Verified bitwise-exact on real data |
| Encoding Unit (functional) | ✅ done | ✅ faithful | Fig 11; sign-magnitude classify |
| PE / Compute Unit (functional) | ✅ done | ✅ faithful | Fig 12; shift-and-add verified |
| A8W8 quantization | ✅ done | ⚠️ partial | dynamic per-tensor, not Q-Diffusion calibration |
| zero ratio (Fig 5 core) | ✅ verified | ✅ faithful | 45.7% vs paper 44% |
| >4-bit ratio (Fig 5) | ⏳ pending | ⚠️ partial | 0% vs paper 4%; needs tighter calibration |
| Full-network Fig 5 bar chart | ⏳ pending | ❌ not yet | only 6 layers / 64 rows / 3 steps sampled so far |
| Weights | fake random | ❌ not faithful (OK) | equivalence is W-independent; real W needed only for cycle counts (Week 2) |
| Defo cycle decision (Fig 9/17) | ⏳ not started | — | Week 1 Day 3 |
| Cycle simulator (Fig 13) | ⏳ not started | — | Week 2 |
| Baselines ITC/Diffy/Cambricon-D | ⏳ not started | — | Week 2 |
| RTL (Encoding Unit + 1 PE) | ⏳ not started | — | Week 3 |
| Fast-dLLM extension demo | ⏳ not started | — | Week 4 |

### On the "not faithful (OK)" items
- **fake weight**: This is a *correct* engineering choice, not a compromise. The
  Ditto equivalence is a distributive-law identity, independent of the weight
  values. Real weights matter only when computing actual cycle/FLOP counts
  (Week 2 cycle simulator), where we will read them from the SDM checkpoint.

---

## 3. How much of the collected trace did we actually use?

We collected **20 GB** (20 images × 50 steps × 6 layers × input+output, 6120 .npz).

Used so far for Day 2 validation:
- 1 image (image_000) of 20
- 3 step transitions of 49
- 64 rows per layer (of thousands)
- input activations only (output field untouched)

So we used **< 1%** of the trace. This is appropriate for Day 2 (logic
validation only). Day 4–5 (full Fig 5) and Week 2 (cycle simulation) will
consume the full dataset.

---

## 4. Path to "Fig 5 reproduced" (Day 4–5)

To upgrade Fig 5 from "core verified" to "reproduced":
- [ ] Add Q-Diffusion calibration (or a percentile-clip scale) to replace bare dynamic absmax.
- [ ] Run all 20 images × all 50 steps × all rows × 6 layers.
- [ ] Aggregate into the zero / 4-bit / >4-bit bar chart.
- [ ] Compare against paper Fig 5 SDM column (zero 44%, ≤4-bit 96%, >4-bit 4%).
- Expected: with calibration, >4-bit rises from ~0% toward the paper's ~4%.

## 5. Path to "Fig 13 reproduced" (Week 2)
- [ ] Extract real SDM weights.
- [ ] Build the cycle simulator (PE-array scheduling + memory model).
- [ ] Implement ITC / Diffy / Cambricon-D baselines.
- [ ] Compute speedup; target paper's 1.5× over ITC within ±20%.

## 6. Path to "Fig 17 reproduced" (Week 2)
- [ ] Implement Defo cycle-based execution-flow decision.
- [ ] Run per-layer Cycle_act vs Cycle_diff across all layers.
- [ ] Compute Defo accuracy; target paper's 92%.

---

## 7. One-line honest summary

> **The Ditto algorithm is faithfully implemented and its correctness is verified
> bitwise-exact on real SDM data; the core temporal-similarity claim (zero ratio)
> matches the paper. The quantitative figure reproductions (full Fig 5 with
> calibration, Fig 13 speedup, Fig 17 Defo accuracy) are not yet done — they are
> the work of Day 4–5 and Week 2. We also reverse-engineered three quantization
> implementation details the paper leaves implicit.**

# Ditto Architecture Spec — Our Implementation

Module-level interface spec for our Ditto reproduction. References paper Section V (Fig 10–12) but documents our specific implementation choices and Python/RTL interfaces. (For the as-built RTL — 13 verified modules including VPU-restore and Defo in Verilog — see `rtl/` and `docs/rtl_diagrams.md`; this spec is the design-time interface contract those modules were built against.)

## Top-level data flow

```
                           ┌──────────────┐
                           │  Defo Unit   │  (control)
                           └──────┬───────┘
                                  │ execution_type
                                  ▼
prev_act_t-1 ──┐
               │   ┌───────────┐    reordered    ┌──────────┐    psum    ┌───────┐    out_t
curr_act_t ────┼──▶│ Encoding  │──── 4-bit + ───▶│  PE      │───────────▶│ VPU   │──────────▶
               │   │   Unit    │     metadata    │  array   │            │       │
weights ───────┘   └───────────┘                 └──────────┘            └───────┘
                                                                            ▲
                                                                out_{t-1} ──┘  (summation)
```

## Module specs

### 1. Encoding Unit (`encoding_unit.py`, `rtl/common/encoding_unit.v`)

**Functional Python interface:**
```python
class EncodingUnit:
    def __init__(self, queue_depth: int = 16):
        self.queue = deque(maxlen=queue_depth)

    def process(self,
                prev_act: np.ndarray,    # int8, shape [N]
                curr_act: np.ndarray,    # int8, shape [N]
                weights:  np.ndarray,    # int8, shape [N]
                ) -> EncodedBatch:
        """
        Returns an EncodedBatch with:
          data:     int8 values (4-bit each, packed in int8 lower nibble)
          weights:  int8 (full bit-width)
          metadata: 1-bit per element: 1 = "this is the high nibble, shift << 4"
          control:  per-input-element 2-bit signal {ctrl=00 zero, 01 low4, 1X full8}
          n_valid:  count of non-zero entries actually enqueued
        """
```

**Internal logic (mirrors paper Fig 11):**
1. `diff = curr_act - prev_act`  (signed int8 subtraction with overflow guard)
2. classify into zero / 4-bit (signed range [-8,7]) / >4-bit, emit sign-magnitude
3. zeros are skipped; >4-bit values split into nibbles for the slot PE

**As-built note:** the RTL Encoding Unit uses a **sign-magnitude** classification — a value in signed [-8,7] is 4-bit. A literal two's-complement nibble-zero check misclassifies small negatives as 8-bit; the sign-magnitude datapath is required and is verified exhaustively over all 509 diff values (`rtl/tb/test_encoding_unit.py`).

**RTL interface (Verilog, as built):**
```verilog
module encoding_unit #(
    parameter DIFF_WIDTH = 9
)(
    input  wire signed [DIFF_WIDTH-1:0] diff,
    output wire                          is_zero,
    output wire                          is_wide,    // >4-bit
    output wire                          sign,
    output wire [DIFF_WIDTH-1:0]         mag
);
// 4-lane wrapper: encoding_unit_x4.v
```

Cocotb test plan (as run): exhaustive over all 509 diff values + 2000 random vectors, compare RTL classification against the Python functional model — 0 mismatch.

---

### 2. Compute Unit / PE (`pe_func.py`, `rtl/common/pe_diff*.v`)

**Functional Python interface:**
```python
class PE:
    def __init__(self):
        self.psum = 0

    def cycle(self, data, weight, metadata) -> int:
        """4-multiplier MAC; high-nibble lanes shift <<4; accumulate into psum."""
```

**RTL interface (Verilog, as built — `pe_diff.v` / `pe_diff_slot.v`):**
```verilog
module pe_diff_slot #(
    parameter LANES = 4, DIFF_WIDTH = 9, W_WIDTH = 8, ACC_WIDTH = 32
)(
    input  wire                          clk, rst, valid,
    input  wire [LANES-1:0]              sign_vec, is_zero_vec, is_wide_vec,
    input  wire [LANES*DIFF_WIDTH-1:0]   mag_vec,
    input  wire [LANES*W_WIDTH-1:0]      w_vec,
    output reg  signed [ACC_WIDTH-1:0]   acc,
    output reg  [7:0]                    slots_this_cycle,
    output reg  [31:0]                   slots_total
);
```

Internal: 4 multipliers (4-bit × 8-bit), zero-skip, nibble-split shift-add for wide lanes, accumulator. Variants built and synthesized: single-cycle (`pe_diff`), slot (`pe_diff_slot`), 3-stage pipelined (`pe_diff_pipe`), and carry-save (`pe_diff_csa`). Cocotb: psum == numpy golden; synthesis Fmax in `rtl/SYNTHESIS.md`.

---

### 3. Vector Processing Unit (`vpu_restore.v` built; non-linears modeled)

The **restore** path (accumulate difference back to full activation, `act_curr = act_prev + diff`) is implemented in RTL (`rtl/common/vpu_restore.v`) and verified as the exact inverse of the diff generator (`restore(diff_generator(act)) == act`). The non-linear functions (SiLU/GeLU/Softmax/LayerNorm/GroupNorm) are modeled as analytical cycles in the performance model (not in the difference domain; the VPU runs them after the GEMM). Per the energy model, the VPU is ~5% of energy and its serial cycles move the compute ceiling 9.03x → 8.89x.

---

### 4. Defo Unit (`defo_unit.v` built)

The runtime decision is implemented in RTL (`rtl/common/defo_unit.v`) as a roofline comparison `cost(mode) = max(compute, memory)` with DIFF's memory term = 2× activation bytes (the previous-frame re-read), picking the cheaper mode (stop-loss). Decision rule (Fig 9): use difference if its cost is lower, else fall back to original-activation. The static half (locate non-linear boundaries) is in `defo_static.py`.

---

## Memory hierarchy & dataflow

```
DRAM (bandwidth NOT assumed as a single value — see below)
  ├─ weights (read once per layer; modeled SRAM-resident, amortized)
  ├─ prev activation t-1 (read for diff calc — the forced-DRAM prev-frame traffic)
  └─ output t-1 (read for summation)

On-chip SRAM (192 MB total, paper Table III)
```

**On DRAM bandwidth (important):** the paper does **not** publish Ditto's DRAM bandwidth or size (Table III has no DRAM row). This spec originally assumed a single 1.5 TB/s figure; the final performance model does **not** — speedup is reported as a **bandwidth roofline** (bandwidth is the explicit sweep axis), and the paper's 1.5× is reached at ~251 GB/s while its 14.4% Defo-flip reverse-estimates the unpublished DRAM to ~3.2 TB/s. See `docs/SUMMARY.md` §3.

## Cycle accounting

```
Cycle_encoding_unit = N / 4                       (4 lanes)
Cycle_pe_array      = N_nonzero / (n_pe × 4)      (zero-skip; zero_ratio ≈ 0.46 SDM)
Cycle_vpu           = N / vpu_throughput          (serial, after GEMM)
Total (pipelined)   ≈ max(EU, PE) + serial VPU + fill
```

The pure-compute per-layer ratio is ~4 × (1 − zero_ratio)⁻¹; memory stalls and the attention two-sub-op penalty reduce the end-to-end ceiling to 8.89× (attention+VPU-aware), and realistic bandwidth brings the attainable speedup to the paper's ~1.5×.

## Validation strategy (as executed)

- **Functional**: random + real-trace inputs → Encoding Unit + PE bitwise-identical to direct `act × weight` (integer, tolerance 0). Real SDM traces → Fig 5 bit-width reproduced (45.9% zero vs paper 44.48%).
- **Performance**: speedup roofline vs paper 1.5×; six-segment energy vs Fig 13; Defo rescue vs Fig 16.
- **RTL**: cocotb random + exhaustive vectors vs Python golden (all 13 modules pass); real-trace closure (42.8% zero-skip); Vivado synthesis (Fmax + carry-save 5×).
- **Extension**: DiT-XL/2 and Fast-dLLM v2 through the same model.

## Implementation choices the paper leaves implicit

1. **Quantization**: dynamic per-tensor A8W8 (Q-Diffusion calibration is orthogonal to the architecture story; the >4-bit fraction is scale-dependent, flagged throughout).
2. **PE clustering**: paper gives only the total 39398 PEs, not the grouping; the RTL builds a representative 4×4 tile, not the full fabric.
3. **Sign-magnitude classification**: required for the [-8,7] = 4-bit definition (reverse-engineered, verified exhaustively).
4. **Cross-attention K/V**: constant across steps in conditional models → treated as weights, bypass the Encoding Unit.

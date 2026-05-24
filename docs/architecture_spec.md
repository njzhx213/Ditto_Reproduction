# Ditto Architecture Spec — Our Implementation

Module-level interface spec for our Ditto reproduction. References paper Section V (Fig 10–12) but documents our specific implementation choices and Python/RTL interfaces.

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

### 1. Encoding Unit (`encoding_unit.py`, `rtl/encoding_unit.sv`)

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
2. `high_nibble = diff >> 4`, `low_nibble = diff & 0x0F`
3. `ctrl = (high_nibble != 0, low_nibble != 0)`
4. For each element:
   - `ctrl == 00` → skip (zero diff)
   - `ctrl == 01` → enqueue `(low_nibble, weight, metadata=0)`
   - `ctrl == 1X` → enqueue `(low_nibble, weight, metadata=0)` **and** `(high_nibble, weight, metadata=1)`

**Cycle accounting:** 1 cycle for subtract+compare, 1 cycle for enqueue. Latency = 2 cycles, throughput = 4 elements/cycle.

**RTL interface (SystemVerilog):**
```verilog
module encoding_unit #(
    parameter int N_LANES = 4
)(
    input  logic                clk,
    input  logic                rst_n,
    input  logic [N_LANES-1:0][7:0] prev_act,
    input  logic [N_LANES-1:0][7:0] curr_act,
    input  logic [N_LANES-1:0][7:0] weights,
    input  logic                in_valid,
    output logic                in_ready,
    
    output logic [N_LANES-1:0][3:0] data_out,
    output logic [N_LANES-1:0][7:0] weight_out,
    output logic [N_LANES-1:0]      metadata_out,
    output logic [N_LANES-1:0]      valid_out,
    input  logic                out_ready
);
```

Cocotb test plan: random input streams (zero / 4-bit / 8-bit ratios matching SDM stats from Fig 5), compare RTL output queue against Python functional model element by element.

---

### 2. Compute Unit / PE (`pe_func.py`, `rtl/pe.sv`)

**Functional Python interface:**
```python
class PE:
    def __init__(self):
        self.psum = 0
    
    def cycle(self,
              data:     np.ndarray,    # int4 (stored in int8), shape [4]
              weight:   np.ndarray,    # int8, shape [4]
              metadata: np.ndarray,    # bool, shape [4]
              ) -> int:
        """
        4-multiplier MAC.
        For each lane i:
          prod[i] = data[i] * weight[i]
          if metadata[i]: prod[i] <<= 4   (this was the high nibble of an 8-bit value)
        Returns sum(prod) and accumulates into self.psum.
        """
```

**RTL interface (SystemVerilog):**
```verilog
module pe #(
    parameter int N_MULT = 4
)(
    input  logic                  clk,
    input  logic                  rst_n,
    input  logic                  clear_psum,         // reset accumulator at layer boundary
    
    input  logic [N_MULT-1:0][3:0]  data_in,           // 4-bit operands
    input  logic [N_MULT-1:0][7:0]  weight_in,         // 8-bit operands
    input  logic [N_MULT-1:0]       metadata_in,       // shift << 4 flag
    input  logic                    valid_in,
    
    output logic signed [31:0]    psum_out,           // accumulated result
    output logic                  psum_valid
);
```

Internal:
- 4 multipliers (4-bit × 8-bit → 12-bit)
- Shifters on lane 0 and lane 2 (the high-nibble lanes per the Encoding Unit's reorder)
- 3-stage adder tree (4 → 2 → 1)
- 32-bit signed psum register

Cocotb test plan: random `(data, weight, metadata)` vectors, compare RTL psum against Python golden.

---

### 3. Vector Processing Unit (Python only, no RTL)

**Functional Python interface:**
```python
class VPU:
    def silu(self, x): ...
    def gelu(self, x): ...
    def softmax(self, x, dim): ...
    def layernorm(self, x, weight, bias, eps): ...
    def groupnorm(self, x, num_groups, weight, bias, eps): ...
    def quantize(self, x, scale, zero_point, bits=8): ...
    def dequantize(self, x_int, scale, zero_point): ...
    def add(self, a, b): ...   # for δ×W + out_{t-1}
```

We model the VPU as analytical cycles (BW-bound + non-linear FUs). Per paper Section VI-B, VPU accounts for 0.17% latency and 2.9% energy.

---

### 4. Defo Unit (Python only, no RTL)

**Functional Python interface:**
```python
class DefoUnit:
    def __init__(self, n_entries: int = 512):
        self.table = np.zeros((n_entries, 3), dtype=np.int32)
        # columns: [cycle_act, cycle_diff, decision_bit]
        self.layer_idx = 0
    
    def record_step1_cycle(self, layer_id: int, cycle_count: int):
        """At t=1 (first time step), record original-activation cycle count."""
    
    def record_step2_cycle_and_decide(self, layer_id: int, cycle_count: int):
        """At t=2 (second time step), record diff-processing cycle count
           AND decide which path to use for t≥3."""
    
    def get_decision(self, layer_id: int) -> str:
        """Returns 'original' or 'difference' for t≥3."""
```

Decision rule (Fig 9): `if Cycle_act > Cycle_diff: use difference else: use original`.

---

## Memory hierarchy & dataflow

```
DRAM (HBM2e, 1.5 TB/s assumed)
  ├─ weights (read once per layer)
  ├─ prev activation t-1 (read for diff calc)
  └─ output t-1 (read for summation)
  
On-chip SRAM (192 MB total, partitioned)
  ├─ Previous Act Buffer  (~32 MB)
  ├─ Current Act Buffer   (~32 MB)
  ├─ Weight Quant/Dequant (~32 MB)
  ├─ Softmax Buffer       (~16 MB)
  └─ Accumulation Buffer  (~80 MB)

Interconnect to Compute Unit
  ├─ data: 4 × 4-bit = 16 bits per lane
  ├─ weight: 4 × 8-bit = 32 bits per lane
  ├─ metadata: 4 × 1-bit = 4 bits per lane
  └─ output: 32-bit per PE
```

## Cycle accounting

For a layer of N elements processed with temporal differences:

```
Cycle_encoding_unit = N / 4        (4 lanes, 4 elements/cycle)
Cycle_pe_array      = N_nonzero / (n_pe × 4)
                      where N_nonzero = N × (1 - zero_ratio)
                      and zero_ratio ≈ 0.44 for SDM (Fig 5)
Cycle_vpu           = N / vpu_throughput
                      (varies by op; LN/GN ≈ N/128, softmax ≈ N/64)

Total cycle (pipelined) ≈ max(Cycle_encoding_unit, Cycle_pe_array, Cycle_vpu)
                          + pipeline fill (~10 cycles per layer)
```

For original-activation execution:
```
Cycle_pe_array_act = N / (n_pe × 1)    (full 8-bit, no zero skip → 1 effective element/PE/cycle)
```

So speedup from temporal diff ≈ `Cycle_act / Cycle_diff ≈ 4 × (1 - zero_ratio)^-1 ≈ 7.1` per layer in pure compute terms. Memory stalls and pipeline overhead reduce this to the paper's reported 1.5× end-to-end.

## Validation strategy

**Week 1 (functional):**
- Random tensor inputs → check that Encoding Unit + PE produce same result as `act × weight` computed directly. Tolerance: 0 (bitwise identical for integer arithmetic).
- Real SDM activation traces → reproduce Fig 5 bit-width breakdown within ±5 percentage points.

**Week 2 (cycle):**
- End-to-end SDM run → compare against paper Fig 13 SDM bar (target 1.5× over ITC, accept ±20%).
- Defo accuracy → target 92% (paper number).

**Week 3 (RTL):**
- Cocotb random testing → 1000+ random vectors, compare RTL against Python functional.
- RTL cycle count vs Python cycle model → match within 1-2 cycles.

**Week 4 (extension):**
- Feed Fast-dLLM v2 traces from `~/Fast-dLLM/v2/logs/motivation_100/` into the simulator.
- Report: what speedup does Ditto get on dLLM? Is the temporal similarity assumption still valid?

## Open questions (to resolve as we implement)

1. **Quantization scheme**: Q-Diffusion calibration vs simple dynamic A8W8? Going with dynamic A8W8 for v1 to avoid Q-Diffusion dependency; calibration is orthogonal to the architecture story.
2. **PE clustering**: paper doesn't say how 39398 PEs are organized. Assuming 256 PEs/cluster × ~154 clusters for now.
3. **Encoding Unit queue depth**: paper doesn't specify. Setting to 16 (enough to absorb 2-3 cycles of burst zeros).
4. **Cross-attention K/V**: paper says treat K, V as weights in conditional models. For SDM (which uses CLIP text encoder), this means K/V from text embedding don't go through Encoding Unit. We'll wire a control bypass in the simulator.

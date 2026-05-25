# RTL Synthesis (area & Fmax)

Functional correctness is verified by cocotb + Icarus Verilog (`make`). Synthesis is the
next step: it turns the same RTL into real gates and reports **area** (LUT/FF/DSP) and
**Fmax** (max clock). This is run on a server with the tools; two options are provided.

## Headline experiment

Compare **`pe_diff`** (single-cycle MAC: one long `mul -> sum -> accumulate`
combinational path) against **`pe_diff_pipe`** (the same MAC split into 3 registered
stages). Expectation: the pipelined version reaches a **higher Fmax** (shorter critical
path) at the cost of more flip-flops and 3-cycle latency. This quantifies the pipeline
trade-off the functional sim already proved equivalent in result.

## Option A — Vivado (real FPGA area + Fmax)

On a server with Vivado (use your board's part; default is a Zynq UltraScale+ ZU9EG):

```bash
cd rtl
vivado -mode batch -source synth_vivado.tcl -tclargs xczu9eg-ffvb1156-2-e pe_diff      2.0
vivado -mode batch -source synth_vivado.tcl -tclargs xczu9eg-ffvb1156-2-e pe_diff_pipe 2.0
```

Each run prints utilization (LUT/FF/DSP) and a Fmax derived from the worst-negative-slack
against the 2 ns (500 MHz) target. Try other modules: `encoding_unit`, `pe_diff_slot`,
`defo_unit`, `diff_generator`, `vpu_restore`, `ditto_top`.

## Option B — Yosys (open source, no license)

If Vivado isn't available, Yosys gives cell counts (area proxy) and a longest-path
estimate (critical-path proxy):

```bash
cd rtl
./synth_yosys.sh
```

It reports `stat` (cell counts) and `ltp -noff` (longest combinational path) per module.
`pe_diff` should show a long path; `pe_diff_pipe` a short per-stage path. Yosys does not
give a true FPGA Fmax — use Vivado for that number.

## What to record

For the writeup: per-module LUT/FF/DSP and Fmax, and especially the `pe_diff` vs
`pe_diff_pipe` Fmax delta. Optionally relate the slot PE's area to Ditto's claim of
fitting ~39398 small 4-bit PEs — a single `pe_diff_slot` (4 lanes) times the array size
gives a rough fabric-area estimate to compare against the paper.

## Measured results (Vivado 2025.1, ZU3EG `xczu3eg-sbva484-2-e`)

Out-of-context synthesis, target 1.0 ns (extremes pushed so the tool reports the true
critical path, not slack against a loose target):

| Module | Fmax | Critical path |
|---|---|---|
| `pe_diff` (single-cycle) | **599.9 MHz** | `acc_reg -> acc_reg`, 9 logic levels (5x CARRY8) |
| `pe_diff_pipe` (3-stage) | **575.7 MHz** | same accumulator path |
| `pe_diff_slot` (4-bit/wide) | **612.4 MHz** | same accumulator path |

### The honest finding: pipelining did NOT raise Fmax — and the timing report says why

All three variants top out at ~600 MHz, and the 3-stage pipeline is actually *slightly
slower* (575.7 < 599.9). The critical-path report explains it: for every variant the
worst path is `acc_reg[0] -> acc_reg[31]`, i.e. **the 32-bit accumulator's carry chain**
(9 logic levels, 5 CARRY8 primitives). The pipeline split the `mul -> sum` logic into
stages, but **the real bottleneck is the accumulator add, which the pipeline never
touched** — so Fmax is unchanged, and the extra pipeline registers/routing cost a few
percent. The slot PE, despite its more complex wide-lane nibble-split shift-add, is no
slower (612 MHz) because that logic sits *before* the accumulator and is masked by the
same carry-chain bottleneck.

**Conclusion (a real hardware lesson, not a tuned result):** pipeline benefit depends on
cutting the actual critical path. Here the bottleneck is the wide accumulator's carry
chain, common to all three, so the right optimization is the accumulator itself —
carry-save accumulation (defer the carry-propagate to the end) or segmenting/pipelining
the 32-bit add — not the multiply/sum stages. The takeaway: **bottleneck analysis (read
the timing report's critical path) precedes adding pipeline stages.** This is the
concrete next step for the RTL (see the carry-save experiment).

Area note: the same DSP/LUT/FF counts per `pe_diff_slot` lane, times the array size,
give a rough fabric estimate to compare against Ditto's ~39398-PE claim — a useful
forward exercise now that real per-PE resource numbers are available.

## The fix, verified: carry-save accumulator (`pe_diff_csa`)

The hypothesis from the timing report — that the 32-bit accumulator carry chain is the
bottleneck — was tested directly. `pe_diff_csa` keeps the accumulator in carry-save form
(two registers `acc_s`, `acc_c`), folding each new partial sum with a 3:2 compressor
(`s = a^b^c`, `cout = ((a&b)|(b&c)|(a&c))<<1`) so **no carry propagates per cycle**; the
single carry-propagate add is resolved once at the output, off the accumulation loop.

| Module | Fmax (ZU3EG, 1.0 ns target, OOC) |
|---|---|
| `pe_diff` (carry-propagate) | 599.9 MHz |
| `pe_diff_pipe` (3-stage, mul/sum split) | 575.7 MHz |
| `pe_diff_slot` (4-bit/wide) | 612.4 MHz |
| **`pe_diff_csa` (carry-save)** | **3021.1 MHz** |

**A 5x Fmax improvement, confirming the diagnosis.** Removing the per-cycle carry
propagation lifts the PE from ~600 MHz to ~3 GHz, while remaining functionally identical
(cocotb: resolved `acc_s + acc_c` == numpy dot product, zero-skip still lossless). This
is the decisive evidence that the accumulator carry chain — not the multiply/sum logic
the pipeline split — was the bottleneck.

The lesson the experiment teaches end to end: a 3-stage pipeline on the wrong path gave
nothing (even -4%), while a carry-save accumulator on the right path gave 5x. **Profile
first (read the critical path), then optimize where the bottleneck actually is.**

Caveats (honest): 3021 MHz is an out-of-context post-synthesis estimate; place-and-route
would lower it, and the per-cycle 4-lane multiply/sum tree (not removed by CSA) becomes
the next limiter at these frequencies. The comparison is valid as a relative result
under identical OOC conditions. Carry-save costs an extra accumulator register and a
one-time resolve add — the classic area-for-frequency trade.

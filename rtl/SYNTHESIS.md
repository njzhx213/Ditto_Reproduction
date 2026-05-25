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

# Ditto RTL — Architecture & Verification Diagrams

## 1. Full compute path (`ditto_top.v`): the three core blocks integrated

The Encoding Unit classifies each temporal-difference element, the slot PE does the
zero-skipped 4-bit/wide MAC and counts multiplier slots, and Defo uses that real slot
count plus the layer's memory traffic to choose DIFF vs ACT. One consistent definition
threads through: EU's `is_wide` selects the PE's 1-slot/2-slot multiply, and the PE's
`slots_total` is exactly the diff-mode compute cost Defo reasons about.

```mermaid
flowchart LR
    subgraph IN[Layer inputs]
        D[diff_vec<br/>4 lanes, int9]
        W[w_vec<br/>4 lanes, int8]
        META[layer_n_macs<br/>act_bytes, bw]
    end

    subgraph EU[encoding_unit_x4]
        EUC["classify per lane<br/>signed [-8,7] = 4-bit<br/>is_zero / is_wide / sign / mag"]
    end

    subgraph PE[pe_diff_slot]
        PEC["zero-skip + sign-mag MAC<br/>4-bit lane = 1 slot<br/>>4-bit lane = 2 slots (nibble split)"]
    end

    subgraph DEFO[defo_unit]
        DFC["roofline: cost = max(compute, memory)<br/>DIFF mem = 2x act (prev-frame re-read)<br/>pick min -> stop-loss"]
    end

    D --> EUC
    EUC -->|is_zero, is_wide| PEC
    EUC -->|sign, mag| PEC
    W --> PEC
    PEC -->|acc| ACC[acc = sum diff*weight]
    PEC -->|slots_total| DFC
    META --> DFC
    DFC -->|mode_diff| MODE[DIFF or ACT]

    classDef block fill:#e8f0fe,stroke:#4a7fb5,stroke-width:1px;
    classDef io fill:#f5f5f5,stroke:#999,stroke-width:1px;
    class EUC,PEC,DFC block;
    class D,W,META,ACC,MODE io;
```

Verified end to end (`test_ditto_top.py`): `acc` equals the numpy dot product of the
real differences; `slots_total` gives bit_factor 1.97; Defo picks DIFF on a
compute-bound layer and stop-losses to ACT on a memory-bound one — all through the same
hardware.

## 2. Verification hierarchy — every level checked against a golden reference

The RTL track reuses the validated Functional Ditto / numpy as the golden reference at
each level, from a single combinational unit up to the integrated path and the array.

```mermaid
flowchart TB
    subgraph L1[Unit level]
        EU1["encoding_unit<br/>exhaustive 509 diffs, 0 mismatch"]
        EUX["encoding_unit_x4<br/>4-lane parallel"]
        PE1["pe_diff<br/>zero-skip lossless, acc == dot"]
    end

    subgraph L2[Datapath level]
        DA["datapath A (loose)<br/>EU drives zero-skip"]
        DB["datapath B (tight)<br/>PE consumes sign-mag encoding"]
        DS["datapath slot<br/>4-bit/wide slots == bit_factor 1.97"]
    end

    subgraph L3[System level]
        TOP["ditto_top<br/>EU -> slot PE -> Defo<br/>end-to-end acc + Defo decision"]
    end

    subgraph L4[Array level]
        AP["pe_array_parallel<br/>4x4 == numpy matmul"]
        AS["pe_array_systolic<br/>wavefront == parallel == matmul"]
    end

    subgraph REAL[Real-data closure]
        RT["real SDM trace -> RTL<br/>zero-skip 42.8% ~ paper 44.48%<br/>RTL <-> trace <-> perf-model"]
    end

    GOLD["Golden reference<br/>Functional Ditto / numpy<br/>(validated in the perf model)"]

    GOLD -.-> L1
    L1 --> L2
    L2 --> L3
    L1 --> L4
    L3 --> RT
    GOLD -.-> RT

    classDef ref fill:#fff3cd,stroke:#d4a017,stroke-width:1px;
    classDef lvl fill:#e8f5e9,stroke:#4a7a30,stroke-width:1px;
    class GOLD ref;
    class EU1,EUX,PE1,DA,DB,DS,TOP,AP,AS,RT lvl;
```

All targets pass under cocotb + Icarus Verilog: `cd rtl && make` then one of
`encoding_unit` (default), `x4`, `pe`, `pipe`, `csa`, `datapath`, `datapath_b`,
`datapath_slot`, `defo`, `diffgen`, `vpu`, `array`, `array_sys`, `top`, `real_sdm`.

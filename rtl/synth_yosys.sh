#!/usr/bin/env bash
# synth_yosys.sh - run open-source Yosys synthesis on the Ditto RTL modules and report
# cell counts (area proxy) + longest combinational path (critical-path proxy).
#
#   cd rtl && ./synth_yosys.sh
#
# Requires: yosys (open source). Install on the server with e.g.
#   conda install -c conda-forge yosys     # or: apt-get install yosys
#
# The headline comparison is pe_diff (single-cycle, long mul->sum->acc path) vs
# pe_diff_pipe (3 registered stages, short per-stage path).

set -euo pipefail
RTL_DIR="$(cd "$(dirname "$0")/common" && pwd)"
MODULES=(encoding_unit pe_diff pe_diff_slot pe_diff_pipe defo_unit diff_generator vpu_restore ditto_top)

if ! command -v yosys >/dev/null 2>&1; then
    echo "yosys not found. Install: conda install -c conda-forge yosys (or apt-get install yosys)"
    exit 1
fi

for m in "${MODULES[@]}"; do
    echo ""
    echo "================ $m ================"
    yosys -q -p "
        read_verilog ${RTL_DIR}/*.v;
        hierarchy -top ${m};
        proc; opt; fsm; opt; memory; opt; techmap; opt;
        stat;
        ltp -noff
    " 2>&1 | grep -E "Number of cells|Number of wires|\\$|Longest topological|cell |  [A-Z]" | head -40 || true
done

echo ""
echo "=== interpretation ==="
echo "  'Number of cells' is the area proxy; compare across modules."
echo "  'Longest topological path' (ltp) is the combinational critical-path proxy:"
echo "    pe_diff should be long (mul->sum->acc in one cycle),"
echo "    pe_diff_pipe should be short per stage (registers break the path)."
echo "  For a real FPGA Fmax, use synth_vivado.tcl on a server with Vivado."

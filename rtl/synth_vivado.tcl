# synth_vivado.tcl
# Out-of-context synthesis of the Ditto RTL modules with Vivado, reporting area
# (LUT/FF/DSP) and timing (achieved Fmax). Run on a server with Vivado:
#
#   vivado -mode batch -source synth_vivado.tcl -tclargs <part> <module> <period_ns>
#
# Examples:
#   vivado -mode batch -source synth_vivado.tcl -tclargs xczu9eg-ffvb1156-2-e pe_diff      2.0
#   vivado -mode batch -source synth_vivado.tcl -tclargs xczu9eg-ffvb1156-2-e pe_diff_pipe 2.0
#
# The pe_diff vs pe_diff_pipe comparison is the headline: the 3-stage pipeline should
# reach a higher Fmax (shorter critical path) at the cost of FFs/latency.
#
# Default part is a Zynq UltraScale+ (ZU9EG); change to your board's part.

set part   [lindex $argv 0]
set top    [lindex $argv 1]
set period [lindex $argv 2]
if {$part   eq ""} { set part   "xczu9eg-ffvb1156-2-e" }
if {$top    eq ""} { set top    "pe_diff" }
if {$period eq ""} { set period "2.0" }

set rtl_dir "[file dirname [info script]]/common"

# all module sources (synth picks the hierarchy under $top)
set srcs [glob -nocomplain $rtl_dir/*.v]

puts "=== Synthesizing $top on $part, target period ${period} ns ==="

# in-memory project
create_project -in_memory -part $part

# read all sources (Vivado elaborates only what $top needs)
foreach f $srcs { read_verilog $f }

# clock constraint: create a clock on 'clk' with the target period
set xdc_file "[file dirname [info script]]/_synth_clk.xdc"
set fh [open $xdc_file w]
puts $fh "create_clock -name clk -period $period \[get_ports clk\]"
close $fh
read_xdc $xdc_file

# out-of-context synthesis (no I/O buffers, pure logic timing)
synth_design -top $top -part $part -mode out_of_context

# ---- reports ----
puts "\n=== UTILIZATION ($top) ==="
report_utilization

puts "\n=== TIMING SUMMARY ($top) ==="
report_timing_summary -delay_type max -max_paths 1

# achieved Fmax = 1 / (period - WNS)
set wns [get_property SLACK [get_timing_paths -max_paths 1 -nworst 1 -setup]]
if {$wns ne ""} {
    set achieved_period [expr $period - $wns]
    set fmax [expr 1000.0 / $achieved_period]
    puts "\n=== Fmax ($top) ==="
    puts "target period   : $period ns"
    puts "WNS             : $wns ns"
    puts "achieved period : $achieved_period ns"
    puts "achieved Fmax   : [format %.1f $fmax] MHz"
}

puts "\n=== DONE ($top) ==="

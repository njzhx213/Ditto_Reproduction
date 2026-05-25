# Makefile - cocotb + Icarus Verilog for the Ditto RTL modules.
#
#   cd rtl && make            # default: single-element encoding_unit
#   cd rtl && make x4         # 4-lane parallel encoding_unit_x4
#   cd rtl && make clean      # remove sim artifacts
#
# Requires: cocotb, iverilog.

SIM ?= icarus
TOPLEVEL_LANG ?= verilog

# --- defaults: single-element encoding_unit ---
VERILOG_SOURCES ?= $(PWD)/common/encoding_unit.v
TOPLEVEL        ?= encoding_unit
MODULE          ?= test_encoding_unit

export PYTHONPATH := $(PWD)/tb:$(PYTHONPATH)

include $(shell cocotb-config --makefiles)/Makefile.sim

# --- convenience target for the 4-lane wrapper ---
.PHONY: x4
x4:
	$(MAKE) \
	  VERILOG_SOURCES="$(PWD)/common/encoding_unit.v $(PWD)/common/encoding_unit_x4.v" \
	  TOPLEVEL=encoding_unit_x4 \
	  MODULE=test_encoding_unit_x4

# --- PE (difference MAC) target ---
.PHONY: pe
pe:
	$(MAKE) \
	  VERILOG_SOURCES="$(PWD)/common/pe_diff.v" \
	  TOPLEVEL=pe_diff \
	  MODULE=test_pe_diff

# --- full datapath (EU_x4 + PE) target ---
.PHONY: datapath
datapath:
	$(MAKE) \
	  VERILOG_SOURCES="$(PWD)/common/encoding_unit.v $(PWD)/common/encoding_unit_x4.v $(PWD)/common/pe_diff.v $(PWD)/common/ditto_datapath.v" \
	  TOPLEVEL=ditto_datapath \
	  MODULE=test_ditto_datapath

# --- datapath B (PE consumes EU sign-magnitude encoding) ---
.PHONY: datapath_b
datapath_b:
	$(MAKE) \
	  VERILOG_SOURCES="$(PWD)/common/encoding_unit.v $(PWD)/common/encoding_unit_x4.v $(PWD)/common/pe_diff_b.v $(PWD)/common/ditto_datapath_b.v" \
	  TOPLEVEL=ditto_datapath_b \
	  MODULE=test_ditto_datapath_b

# --- slot-based datapath (4-bit/wide multiplier slots + slot accounting) ---
.PHONY: datapath_slot
datapath_slot:
	$(MAKE) \
	  VERILOG_SOURCES="$(PWD)/common/encoding_unit.v $(PWD)/common/encoding_unit_x4.v $(PWD)/common/pe_diff_slot.v $(PWD)/common/ditto_datapath_slot.v" \
	  TOPLEVEL=ditto_datapath_slot \
	  MODULE=test_ditto_datapath_slot

# --- real SDM trace driving the slot datapath (direction 4) ---
.PHONY: real_sdm
real_sdm:
	$(MAKE) \
	  VERILOG_SOURCES="$(PWD)/common/encoding_unit.v $(PWD)/common/encoding_unit_x4.v $(PWD)/common/pe_diff_slot.v $(PWD)/common/ditto_datapath_slot.v" \
	  TOPLEVEL=ditto_datapath_slot \
	  MODULE=test_real_trace_sdm

# --- Defo decision unit (the third Ditto core block) ---
.PHONY: defo
defo:
	$(MAKE) \
	  VERILOG_SOURCES="$(PWD)/common/defo_unit.v" \
	  TOPLEVEL=defo_unit \
	  MODULE=test_defo_unit

# --- parallel PE array (version A) ---
.PHONY: array
array:
	$(MAKE) \
	  VERILOG_SOURCES="$(PWD)/common/pe_array_parallel.v" \
	  TOPLEVEL=pe_array_parallel \
	  MODULE=test_pe_array_parallel

# --- systolic PE array (version B) ---
.PHONY: array_sys
array_sys:
	$(MAKE) \
	  VERILOG_SOURCES="$(PWD)/common/pe_array_systolic.v" \
	  TOPLEVEL=pe_array_systolic \
	  MODULE=test_pe_array_systolic

# --- full integration: EU -> slot PE -> Defo ---
.PHONY: top
top:
	$(MAKE) \
	  VERILOG_SOURCES="$(PWD)/common/encoding_unit.v $(PWD)/common/encoding_unit_x4.v $(PWD)/common/pe_diff_slot.v $(PWD)/common/defo_unit.v $(PWD)/common/ditto_top.v" \
	  TOPLEVEL=ditto_top \
	  MODULE=test_ditto_top

# --- diff generator (datapath entry: activations -> differences) ---
.PHONY: diffgen
diffgen:
	$(MAKE) \
	  VERILOG_SOURCES="$(PWD)/common/diff_generator.v" \
	  TOPLEVEL=diff_generator \
	  MODULE=test_diff_generator

# --- VPU restore (difference -> full-precision activation, inverse of diff_generator) ---
.PHONY: vpu
vpu:
	$(MAKE) \
	  VERILOG_SOURCES="$(PWD)/common/vpu_restore.v" \
	  TOPLEVEL=vpu_restore \
	  MODULE=test_vpu_restore

# --- 3-stage pipelined PE (higher Fmax) ---
.PHONY: pipe
pipe:
	$(MAKE) \
	  VERILOG_SOURCES="$(PWD)/common/pe_diff_pipe.v" \
	  TOPLEVEL=pe_diff_pipe \
	  MODULE=test_pe_diff_pipe

# --- carry-save accumulator PE (tests the accumulator-bottleneck hypothesis) ---
.PHONY: csa
csa:
	$(MAKE) \
	  VERILOG_SOURCES="$(PWD)/common/pe_diff_csa.v" \
	  TOPLEVEL=pe_diff_csa \
	  MODULE=test_pe_diff_csa

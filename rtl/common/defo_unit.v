// defo_unit.v
// Ditto's Defo runtime decision unit (the third core block alongside EU and PE).
//
// Per layer, Defo chooses between two execution modes and picks the cheaper:
//   ACT  mode: full-activation execution (no difference). cost = max(compute, memory)
//              compute = n_macs / LANES_ITC ; memory = act_bytes / BW
//   DIFF mode: Ditto difference execution.            cost = max(compute, memory)
//              compute = slots_used / LANES_DITTO ; memory = 2*act_bytes / BW
//
// The DIFF memory term is ~2x activation bytes because difference mode must re-read
// the PREVIOUS frame's activations (cross-step, cannot stay resident) -- the same
// previous-frame DRAM traffic modeled in the energy analysis. This is what makes the
// decision real: on a memory-bound layer the extra read can exceed the compute saving,
// so Defo falls back to ACT (the "naive difference is slower" stop-loss, paper Fig 16).
//
// slots_used comes from the slot PE (actual multiplier slots = nonzero lanes weighted
// by 1 for 4-bit / 2 for >4-bit), tying Defo to the real datapath.
//
// To avoid division, both costs are scaled by (LANES_ITC * LANES_DITTO * BW) and
// compared as integer products (a roofline is max of two terms, so we compare the
// scaled max of each mode).

module defo_unit #(
    parameter LANES_ITC   = 1,
    parameter LANES_DITTO = 4,
    parameter W           = 64    // wide enough for scaled products
)(
    input  wire [W-1:0] n_macs,
    input  wire [W-1:0] slots_used,   // from slot PE
    input  wire [W-1:0] act_bytes,
    input  wire [W-1:0] bw,           // bytes per cycle
    output wire         mode_diff,    // 1 = DIFF chosen, 0 = ACT (stop-loss)
    output wire [W-1:0] cost_act_s,   // scaled costs (for observability / test)
    output wire [W-1:0] cost_diff_s
);
    // Scale everything by (LANES_ITC * LANES_DITTO * BW) to clear all denominators:
    //   compute_act  = n_macs/LANES_ITC      -> * (LI*LD*BW) = n_macs*LD*BW
    //   memory_act   = act_bytes/BW          -> * (LI*LD*BW) = act_bytes*LI*LD
    //   compute_diff = slots/LANES_DITTO     -> * (LI*LD*BW) = slots*LI*BW
    //   memory_diff  = 2*act_bytes/BW        -> * (LI*LD*BW) = 2*act_bytes*LI*LD
    wire [W-1:0] comp_act_s = n_macs    * LANES_DITTO * bw;
    wire [W-1:0] mem_act_s  = act_bytes * LANES_ITC   * LANES_DITTO;
    wire [W-1:0] comp_diff_s= slots_used* LANES_ITC   * bw;
    wire [W-1:0] mem_diff_s = act_bytes * LANES_ITC   * LANES_DITTO * 2;

    // roofline: cost = max(compute, memory)
    assign cost_act_s  = (comp_act_s  > mem_act_s ) ? comp_act_s  : mem_act_s;
    assign cost_diff_s = (comp_diff_s > mem_diff_s) ? comp_diff_s : mem_diff_s;

    // choose DIFF iff its (scaled) cost is strictly lower
    assign mode_diff = (cost_diff_s < cost_act_s);
endmodule

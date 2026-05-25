// ditto_top.v
// Full Ditto compute path integrating all three core blocks for one layer:
//
//   diff_vec --> encoding_unit_x4 --> {is_zero,is_wide,sign,mag} --> pe_diff_slot --> acc
//                                              |                          |
//                                              +-- is_wide drives slots   +-- slots_total
//                                                                              |
//   layer_n_macs, act_bytes, bw ------------------------------------------> defo_unit --> mode_diff
//
// One consistent set of definitions threads through: the EU's signed-[-8,7] is_wide
// classification selects the PE's 1-slot vs 2-slot multiply, the PE counts the actual
// multiplier slots, and Defo uses that real slot count (the true diff-mode compute
// cost) plus the layer's memory traffic to decide DIFF vs ACT (stop-loss). This is the
// three core blocks working together, not in isolation.
//
// Streaming: feed LANES diffs+weights per cycle while valid; acc and slots_total
// accumulate. Defo evaluates continuously from the running slots_total (the host reads
// mode_diff after the layer completes, when slots_total is final).

module ditto_top #(
    parameter LANES      = 4,
    parameter DIFF_WIDTH = 9,
    parameter W_WIDTH    = 8,
    parameter ACC_WIDTH  = 32,
    parameter LANES_ITC  = 1,
    parameter LANES_DITTO= 4,
    parameter DW64       = 64
)(
    input  wire                          clk,
    input  wire                          rst,
    input  wire                          valid,
    input  wire [LANES*DIFF_WIDTH-1:0]   diff_vec,
    input  wire [LANES*W_WIDTH-1:0]      w_vec,
    // layer metadata for Defo
    input  wire [DW64-1:0]               layer_n_macs,
    input  wire [DW64-1:0]               act_bytes,
    input  wire [DW64-1:0]               bw,
    // outputs
    output wire signed [ACC_WIDTH-1:0]   acc,
    output wire        [31:0]            slots_total,
    output wire [LANES-1:0]              is_zero_vec,
    output wire [LANES-1:0]              is_wide_vec,
    output wire                          mode_diff      // Defo: 1=DIFF, 0=ACT
);
    wire [LANES-1:0]            sign_vec;
    wire [LANES*DIFF_WIDTH-1:0] mag_vec;
    wire [7:0]                  slots_this_cycle;

    // 1) Encoding Unit (x4)
    encoding_unit_x4 #(.DIFF_WIDTH(DIFF_WIDTH), .LANES(LANES)) eu (
        .diff_vec    (diff_vec),
        .is_zero_vec (is_zero_vec),
        .is_wide_vec (is_wide_vec),
        .sign_vec    (sign_vec),
        .mag_vec     (mag_vec)
    );

    // 2) slot PE: consumes EU encoding, counts slots
    pe_diff_slot #(.LANES(LANES), .DIFF_WIDTH(DIFF_WIDTH),
                   .W_WIDTH(W_WIDTH), .ACC_WIDTH(ACC_WIDTH)) pe (
        .clk              (clk),
        .rst              (rst),
        .valid            (valid),
        .sign_vec         (sign_vec),
        .mag_vec          (mag_vec),
        .is_zero_vec      (is_zero_vec),
        .is_wide_vec      (is_wide_vec),
        .w_vec            (w_vec),
        .acc              (acc),
        .slots_this_cycle (slots_this_cycle),
        .slots_total      (slots_total)
    );

    // 3) Defo: uses the PE's running slot count as diff-mode compute cost
    defo_unit #(.LANES_ITC(LANES_ITC), .LANES_DITTO(LANES_DITTO), .W(DW64)) defo (
        .n_macs      (layer_n_macs),
        .slots_used  ({32'b0, slots_total}),   // zero-extend to 64-bit
        .act_bytes   (act_bytes),
        .bw          (bw),
        .mode_diff   (mode_diff),
        .cost_act_s  (),
        .cost_diff_s ()
    );
endmodule

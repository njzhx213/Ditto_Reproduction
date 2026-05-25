// ditto_datapath_slot.v
// Full datapath with the slot-based PE: the EU's complete output (sign, magnitude,
// is_zero, is_wide) drives the 4-bit/>4-bit multiplier-slot PE. This is the most
// faithful version of the Ditto compute path: zero-skip + 4-bit-dominant + 2-slot
// wide handling, with slot accounting that ties back to the performance model's
// bit_factor.

module ditto_datapath_slot #(
    parameter LANES      = 4,
    parameter DIFF_WIDTH = 9,
    parameter W_WIDTH    = 8,
    parameter ACC_WIDTH  = 32
)(
    input  wire                          clk,
    input  wire                          rst,
    input  wire                          valid,
    input  wire [LANES*DIFF_WIDTH-1:0]   diff_vec,
    input  wire [LANES*W_WIDTH-1:0]      w_vec,
    output wire signed [ACC_WIDTH-1:0]   acc,
    output wire        [7:0]             slots_this_cycle,
    output wire        [31:0]            slots_total,
    output wire [LANES-1:0]              is_zero_vec,
    output wire [LANES-1:0]              is_wide_vec
);
    wire [LANES-1:0]            sign_vec;
    wire [LANES*DIFF_WIDTH-1:0] mag_vec;

    encoding_unit_x4 #(.DIFF_WIDTH(DIFF_WIDTH), .LANES(LANES)) eu (
        .diff_vec    (diff_vec),
        .is_zero_vec (is_zero_vec),
        .is_wide_vec (is_wide_vec),
        .sign_vec    (sign_vec),
        .mag_vec     (mag_vec)
    );

    pe_diff_slot #(.LANES(LANES), .DIFF_WIDTH(DIFF_WIDTH),
                   .W_WIDTH(W_WIDTH), .ACC_WIDTH(ACC_WIDTH)) pe (
        .clk              (clk),
        .rst              (rst),
        .valid            (valid),
        .sign_vec         (sign_vec),
        .mag_vec          (mag_vec),
        .is_zero_vec      (is_zero_vec),
        .is_wide_vec      (is_wide_vec),    // is_wide now consumed -> slot selection
        .w_vec            (w_vec),
        .acc              (acc),
        .slots_this_cycle (slots_this_cycle),
        .slots_total      (slots_total)
    );
endmodule

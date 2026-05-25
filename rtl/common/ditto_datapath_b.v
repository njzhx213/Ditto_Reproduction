// ditto_datapath_b.v  (version B: PE consumes EU's sign-magnitude encoding)
//
//   diff_vec --> encoding_unit_x4 --> {sign_vec, mag_vec, is_zero_vec} --> pe_diff_b --> acc
//                                       w_vec ----------------------------->
//
// This is the tighter coupling: the PE no longer re-reads the raw diff; it works
// entirely from the EU's encoded output (sign, magnitude, zero flag), as in Fig 12.
// is_wide is exposed for observability (a full version would route wide lanes to a
// 2-slot multiplier path; here the magnitude product is full-width regardless, so
// the arithmetic is already correct and wide-routing is a micro-arch refinement).

module ditto_datapath_b #(
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

    pe_diff_b #(.LANES(LANES), .DIFF_WIDTH(DIFF_WIDTH),
                .W_WIDTH(W_WIDTH), .ACC_WIDTH(ACC_WIDTH)) pe (
        .clk         (clk),
        .rst         (rst),
        .valid       (valid),
        .sign_vec    (sign_vec),      // EU encoding ->
        .mag_vec     (mag_vec),       // EU encoding ->
        .w_vec       (w_vec),
        .is_zero_vec (is_zero_vec),   // EU encoding ->
        .acc         (acc)
    );
endmodule

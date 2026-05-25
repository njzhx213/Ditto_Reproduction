// ditto_datapath.v  (version A: loose coupling)
// Connects the verified Encoding Unit (x4) and the difference PE into one datapath.
//
//   diff_vec ---+--> encoding_unit_x4 --> is_zero_vec --+
//               |                                        |
//               +--------------------------------------> pe_diff --> acc
//                                  w_vec ---------------->
//
// The EU classifies each lane's temporal difference and emits the zero-skip flags;
// the PE uses those flags to skip zero lanes while accumulating diff*weight. This is
// the EU's role in Ditto: produce the control (which lanes to skip) that the PE acts
// on. diff_vec feeds both (EU to classify, PE to compute) -- physically they receive
// the difference in parallel. (Version B will instead have the PE consume the EU's
// sign-magnitude encoding directly.)
//
// Golden reference (testbench): acc == sum over cycles/lanes of diff*weight; the
// EU-driven zero-skip must not change the result (lossless), proving the two modules
// compose correctly.

module ditto_datapath #(
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
    // expose EU classification for observability
    output wire [LANES-1:0]              is_zero_vec,
    output wire [LANES-1:0]              is_wide_vec
);
    wire [LANES-1:0]            sign_vec_u;
    wire [LANES*DIFF_WIDTH-1:0] mag_vec_u;

    encoding_unit_x4 #(.DIFF_WIDTH(DIFF_WIDTH), .LANES(LANES)) eu (
        .diff_vec    (diff_vec),
        .is_zero_vec (is_zero_vec),
        .is_wide_vec (is_wide_vec),
        .sign_vec    (sign_vec_u),
        .mag_vec     (mag_vec_u)
    );

    pe_diff #(.LANES(LANES), .DIFF_WIDTH(DIFF_WIDTH),
              .W_WIDTH(W_WIDTH), .ACC_WIDTH(ACC_WIDTH)) pe (
        .clk         (clk),
        .rst         (rst),
        .valid       (valid),
        .diff_vec    (diff_vec),
        .w_vec       (w_vec),
        .is_zero_vec (is_zero_vec),   // EU drives the PE's zero-skip
        .acc         (acc)
    );
endmodule

// pe_diff.v
// Ditto difference-processing PE (paper Fig 12), first functional version.
//
// Each cycle it receives LANES=4 difference elements (already encoded by the
// Encoding Unit) and their weights, and accumulates the difference-MAC:
//
//     acc <- acc + sum_lane( is_zero ? 0 : diff_lane * weight_lane )
//
// Key points vs a plain MAC:
//  * ZERO-SKIP: a lane flagged is_zero contributes nothing and its multiplier
//    input is gated to 0 (Ditto's main compute saving). Because diff==0 anyway,
//    gating does NOT change the arithmetic result -- the optimization is exactly
//    that: free, lossless skipping. The testbench checks acc == naive dot product.
//  * diff is 9-bit signed (curr_int8 - prev_int8, range [-254,254]); weight is
//    8-bit signed (W8). 4-bit vs >4-bit lanes use different multiplier widths in
//    real hardware; this functional version computes the full product per lane
//    (result identical) and leaves the 4-bit/wide datapath split as a noted
//    micro-architecture refinement.
//
// Control: synchronous, active-high reset clears acc; `valid` gates accumulation;
// acc is exposed continuously.

module pe_diff #(
    parameter LANES      = 4,
    parameter DIFF_WIDTH = 9,    // signed diff
    parameter W_WIDTH    = 8,    // signed weight
    parameter ACC_WIDTH  = 32    // signed accumulator
)(
    input  wire                          clk,
    input  wire                          rst,        // sync, active-high: clears acc
    input  wire                          valid,      // accumulate this cycle when 1
    input  wire [LANES*DIFF_WIDTH-1:0]   diff_vec,   // packed signed diffs
    input  wire [LANES*W_WIDTH-1:0]      w_vec,      // packed signed weights
    input  wire [LANES-1:0]              is_zero_vec,// zero-skip flags from EU
    output reg  signed [ACC_WIDTH-1:0]   acc
);
    integer i;
    reg signed [ACC_WIDTH-1:0] sum_comb;
    reg signed [DIFF_WIDTH-1:0] d;
    reg signed [W_WIDTH-1:0]    w;
    reg signed [DIFF_WIDTH+W_WIDTH-1:0] prod;

    // combinational per-cycle sum over lanes (with zero-skip gating)
    always @(*) begin
        sum_comb = 0;
        for (i = 0; i < LANES; i = i + 1) begin
            d = diff_vec[i*DIFF_WIDTH +: DIFF_WIDTH];
            w = w_vec[i*W_WIDTH +: W_WIDTH];
            if (is_zero_vec[i])
                prod = 0;                 // zero-skip: gate multiplier input
            else
                prod = d * w;             // full signed product
            sum_comb = sum_comb + prod;
        end
    end

    // synchronous accumulate
    always @(posedge clk) begin
        if (rst)
            acc <= 0;
        else if (valid)
            acc <= acc + sum_comb;
    end
endmodule

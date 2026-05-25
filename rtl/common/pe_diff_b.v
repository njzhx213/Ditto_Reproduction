// pe_diff_b.v  (version B: consumes the EU's sign-magnitude encoding)
//
// Unlike pe_diff (version A, which re-read the raw diff), this PE consumes the
// Encoding Unit's output directly -- sign + magnitude per lane -- and forms the
// product as a sign-magnitude MAC, which is how the Fig 12 shift-add PE actually
// works on the encoded difference:
//
//     diff = sign ? -magnitude : magnitude
//     diff * weight = sign ? -(magnitude * weight) : (magnitude * weight)
//
// magnitude is unsigned (|diff|, fits in DIFF_WIDTH); weight is signed W8. The
// product (unsigned-mag x signed-weight) is then negated when sign=1. Zero-skip
// gates the lane to 0. The arithmetic result is identical to A's direct diff*weight
// (sign-magnitude is just another encoding of the same diff), which the testbench
// verifies against the naive dot product.

module pe_diff_b #(
    parameter LANES      = 4,
    parameter DIFF_WIDTH = 9,    // magnitude width (|diff| in [0,254])
    parameter W_WIDTH    = 8,    // signed weight
    parameter ACC_WIDTH  = 32
)(
    input  wire                          clk,
    input  wire                          rst,
    input  wire                          valid,
    input  wire [LANES-1:0]              sign_vec,    // from EU
    input  wire [LANES*DIFF_WIDTH-1:0]   mag_vec,     // from EU (|diff|)
    input  wire [LANES*W_WIDTH-1:0]      w_vec,
    input  wire [LANES-1:0]              is_zero_vec, // from EU (zero-skip)
    output reg  signed [ACC_WIDTH-1:0]   acc
);
    integer i;
    reg signed [ACC_WIDTH-1:0]          sum_comb;
    reg        [DIFF_WIDTH-1:0]         mag;          // unsigned magnitude
    reg signed [W_WIDTH-1:0]            w;
    // magnitude (unsigned, zero-extended) x weight (signed): use a signed product
    // with magnitude zero-extended by one bit so it stays non-negative as signed.
    reg signed [DIFF_WIDTH+W_WIDTH:0]   magw;         // signed product (mag>=0)
    reg signed [DIFF_WIDTH+W_WIDTH:0]   prod;

    always @(*) begin
        sum_comb = 0;
        for (i = 0; i < LANES; i = i + 1) begin
            mag = mag_vec[i*DIFF_WIDTH +: DIFF_WIDTH];
            w   = w_vec[i*W_WIDTH +: W_WIDTH];
            // {1'b0, mag} keeps magnitude non-negative when read as signed
            magw = $signed({1'b0, mag}) * w;
            if (is_zero_vec[i])
                prod = 0;                       // zero-skip
            else if (sign_vec[i])
                prod = -magw;                   // diff was negative
            else
                prod = magw;
            sum_comb = sum_comb + prod;
        end
    end

    always @(posedge clk) begin
        if (rst)        acc <= 0;
        else if (valid) acc <= acc + sum_comb;
    end
endmodule

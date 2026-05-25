// pe_diff_csa.v
// Difference PE with a CARRY-SAVE accumulator, to test (and fix) the bottleneck the
// synthesis timing report identified: in pe_diff / _pipe / _slot the critical path is
// the 32-bit accumulator's carry-propagate chain (acc_reg -> acc_reg, 5x CARRY8).
//
// Carry-save accumulation avoids propagating the carry every cycle. The accumulator is
// kept as two registers (acc_s, acc_c); each cycle a 3:2 compressor folds the new
// partial sum in with only XOR / AND-OR logic (no long carry chain):
//     s    = a ^ b ^ c
//     cout = ((a&b)|(b&c)|(a&c)) << 1     and  a+b+c == s + cout
// The single carry-propagate add (acc_s + acc_c) is done ONCE at the end (resolve),
// off the per-cycle critical path. If this raises Fmax above the ~600 MHz the
// carry-propagate versions hit, it confirms the accumulator was the bottleneck.
//
// Functionally identical: resolved acc == sum of (diff*weight), verified vs numpy.

module pe_diff_csa #(
    parameter LANES      = 4,
    parameter DIFF_WIDTH = 9,
    parameter W_WIDTH    = 8,
    parameter ACC_WIDTH  = 40    // headroom; resolved result read out
)(
    input  wire                          clk,
    input  wire                          rst,
    input  wire                          valid,
    input  wire [LANES*DIFF_WIDTH-1:0]   diff_vec,
    input  wire [LANES*W_WIDTH-1:0]      w_vec,
    input  wire [LANES-1:0]              is_zero_vec,
    output wire signed [ACC_WIDTH-1:0]   acc          // resolved (acc_s + acc_c)
);
    integer i;
    reg signed [DIFF_WIDTH-1:0] d;
    reg signed [W_WIDTH-1:0]    w;
    reg signed [ACC_WIDTH-1:0]  prod;
    reg signed [ACC_WIDTH-1:0]  new_sum;

    // carry-save accumulator registers
    reg [ACC_WIDTH-1:0] acc_s;
    reg [ACC_WIDTH-1:0] acc_c;

    // per-cycle partial sum over lanes (this part still has a small adder tree, but it
    // is the 4-lane sum, not the 500-deep accumulation; the deep accumulation is what
    // the CSA removes from the critical path)
    always @(*) begin
        new_sum = 0;
        for (i = 0; i < LANES; i = i + 1) begin
            d = diff_vec[i*DIFF_WIDTH +: DIFF_WIDTH];
            w = w_vec[i*W_WIDTH +: W_WIDTH];
            if (is_zero_vec[i]) prod = 0;
            else                prod = d * w;
            new_sum = new_sum + prod;
        end
    end

    // 3:2 carry-save fold: (acc_s, acc_c, new_sum) -> (acc_s', acc_c')
    wire [ACC_WIDTH-1:0] a = acc_s;
    wire [ACC_WIDTH-1:0] b = acc_c;
    wire [ACC_WIDTH-1:0] c = new_sum;
    wire [ACC_WIDTH-1:0] csa_s = a ^ b ^ c;
    wire [ACC_WIDTH-1:0] csa_c = ((a & b) | (b & c) | (a & c)) << 1;

    always @(posedge clk) begin
        if (rst) begin
            acc_s <= 0;
            acc_c <= 0;
        end else if (valid) begin
            acc_s <= csa_s;          // no carry propagation here -> short path
            acc_c <= csa_c;
        end
    end

    // resolve once (this add has the carry chain, but it is NOT in the accumulation
    // loop -- it is a single combinational output, off the per-cycle timing path)
    assign acc = $signed(acc_s) + $signed(acc_c);
endmodule

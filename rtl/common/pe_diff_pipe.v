// pe_diff_pipe.v
// 3-stage pipelined version of the difference PE, for higher clock frequency.
//
//   Stage 1 (mul) : per-lane prod = is_zero ? 0 : diff * weight     -> registers
//   Stage 2 (sum) : s = prod[0]+prod[1]+prod[2]+prod[3]             -> register
//   Stage 3 (acc) : acc += s
//
// Splitting the long combinational mul->sum->accumulate path into three registered
// stages shortens the critical path (higher Fmax) at the cost of 3-cycle latency: an
// input applied on cycle t reaches the accumulator on cycle t+3. A `valid` bit flows
// down the pipeline alongside the data so only real inputs accumulate (bubbles add 0).
// Throughput is still one input/cycle once the pipe is full. The final accumulator
// (after a 3-cycle drain) equals the single-cycle PE and the numpy dot product --
// pipelining is an equivalence transform (same result, higher Fmax).
//
// zero-skip is applied in stage 1 (the multiplier input is gated to 0), so the
// power/area saving is preserved through the pipeline.

module pe_diff_pipe #(
    parameter LANES      = 4,
    parameter DIFF_WIDTH = 9,
    parameter W_WIDTH    = 8,
    parameter ACC_WIDTH  = 32
)(
    input  wire                          clk,
    input  wire                          rst,
    input  wire                          valid_in,
    input  wire [LANES*DIFF_WIDTH-1:0]   diff_vec,
    input  wire [LANES*W_WIDTH-1:0]      w_vec,
    input  wire [LANES-1:0]              is_zero_vec,
    output reg  signed [ACC_WIDTH-1:0]   acc,
    output reg                           valid_out    // 1 when acc just took a real input
);
    integer i;

    // ---- Stage 1 registers: per-lane products + valid ----
    reg signed [DIFF_WIDTH+W_WIDTH-1:0] s1_prod [0:LANES-1];
    reg                                  s1_valid;

    // ---- Stage 2 registers: summed partial + valid ----
    reg signed [ACC_WIDTH-1:0]          s2_sum;
    reg                                  s2_valid;

    // combinational helpers
    reg signed [DIFF_WIDTH-1:0]         d;
    reg signed [W_WIDTH-1:0]            w;
    reg signed [ACC_WIDTH-1:0]          sum_comb;

    always @(posedge clk) begin
        if (rst) begin
            for (i = 0; i < LANES; i = i + 1)
                s1_prod[i] <= 0;
            s1_valid  <= 1'b0;
            s2_sum    <= 0;
            s2_valid  <= 1'b0;
            acc       <= 0;
            valid_out <= 1'b0;
        end else begin
            // ---- Stage 3 (acc): consume stage-2 result ----
            if (s2_valid)
                acc <= acc + s2_sum;
            valid_out <= s2_valid;

            // ---- Stage 2 (sum): sum stage-1 products ----
            sum_comb = 0;
            for (i = 0; i < LANES; i = i + 1)
                sum_comb = sum_comb + s1_prod[i];
            s2_sum   <= sum_comb;
            s2_valid <= s1_valid;

            // ---- Stage 1 (mul): multiply current inputs (zero-skip gated) ----
            for (i = 0; i < LANES; i = i + 1) begin
                d = diff_vec[i*DIFF_WIDTH +: DIFF_WIDTH];
                w = w_vec[i*W_WIDTH +: W_WIDTH];
                if (is_zero_vec[i])
                    s1_prod[i] <= 0;            // zero-skip
                else
                    s1_prod[i] <= d * w;
            end
            s1_valid <= valid_in;
        end
    end
endmodule

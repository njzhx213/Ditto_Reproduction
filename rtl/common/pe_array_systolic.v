// pe_array_systolic.v  (version B: output-stationary systolic array)
// Same matmul as version A -- C[M][N] = diff[M][K] @ weight[K][N], M=N=4, K=8 -- but
// with a systolic dataflow:
//   * diff[i][k] enters row i from the LEFT and propagates one PE to the right each cycle
//   * weight[k][j] enters col j from the TOP and propagates one PE down each cycle
//   * PE[i][j] accumulates diff*weight when the two operands arrive (output-stationary)
//
// The boundary feed is staggered (row i delayed by i, col j delayed by j) so operands
// meet on the diagonal wavefront. The host drives the staggered boundary inputs; the
// array shifts internally. After M+N+K-2(+1) cycles the accumulators hold C.
//
// Zero-skip: a PE gates its multiply to 0 when its incoming diff is 0 (power saving);
// the diff value still propagates (dataflow unchanged), so the result is identical to
// the parallel array A and to numpy matmul -- which the testbench checks.

module pe_array_systolic #(
    parameter M = 4,
    parameter N = 4,
    parameter DIFF_WIDTH = 9,
    parameter W_WIDTH    = 8,
    parameter ACC_WIDTH  = 32
)(
    input  wire                          clk,
    input  wire                          rst,
    input  wire                          en,                  // advance the array
    input  wire [M*DIFF_WIDTH-1:0]       a_left,              // boundary: diff into col 0, per row
    input  wire [N*W_WIDTH-1:0]          w_top,               // boundary: weight into row 0, per col
    output wire [M*N*ACC_WIDTH-1:0]      c_flat
);
    genvar gi, gj;
    integer i, j;

    // per-PE registers: held A (flows right), held W (flows down), accumulator
    reg signed [DIFF_WIDTH-1:0] a_reg [0:M-1][0:N-1];
    reg signed [W_WIDTH-1:0]    w_reg [0:M-1][0:N-1];
    reg signed [ACC_WIDTH-1:0]  acc   [0:M-1][0:N-1];

    // incoming operands to each PE (from neighbor or boundary)
    reg signed [DIFF_WIDTH-1:0] a_in;
    reg signed [W_WIDTH-1:0]    w_in;
    reg signed [ACC_WIDTH-1:0]  prod;

    always @(posedge clk) begin
        if (rst) begin
            for (i = 0; i < M; i = i + 1)
                for (j = 0; j < N; j = j + 1) begin
                    a_reg[i][j] <= 0;
                    w_reg[i][j] <= 0;
                    acc[i][j]   <= 0;
                end
        end else if (en) begin
            for (i = 0; i < M; i = i + 1) begin
                for (j = 0; j < N; j = j + 1) begin
                    // incoming A: left neighbor, or boundary at col 0
                    if (j == 0)
                        a_in = a_left[i*DIFF_WIDTH +: DIFF_WIDTH];
                    else
                        a_in = a_reg[i][j-1];
                    // incoming W: top neighbor, or boundary at row 0
                    if (i == 0)
                        w_in = w_top[j*W_WIDTH +: W_WIDTH];
                    else
                        w_in = w_reg[i-1][j];
                    // accumulate (zero-skip gates the product, dataflow unaffected)
                    if (a_in == 0)
                        prod = 0;
                    else
                        prod = a_in * w_in;
                    acc[i][j]   <= acc[i][j] + prod;
                    a_reg[i][j] <= a_in;     // latch -> flows right next cycle
                    w_reg[i][j] <= w_in;     // latch -> flows down next cycle
                end
            end
        end
    end

    // flatten accumulators
    generate
        for (gi = 0; gi < M; gi = gi + 1)
            for (gj = 0; gj < N; gj = gj + 1)
                assign c_flat[(gi*N+gj)*ACC_WIDTH +: ACC_WIDTH] = acc[gi][gj];
    endgenerate
endmodule

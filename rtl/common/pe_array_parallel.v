// pe_array_parallel.v  (version A: fully-parallel PE array)
// A 4x4 grid of difference-MAC PEs computing one matmul tile:
//     C[M][N] = diff[M][K] @ weight[K][N]      (M=N=4, K=8)
//
// Dataflow: K cycles. On cycle k, the array receives diff column k (one element per
// row, M values) and weight row k (one element per column, N values). Every PE[i][j]
// does C[i][j] += diff[i][k] * weight[k][j], with zero-skip when diff[i][k]==0. All 16
// PEs operate in parallel (no systolic delay); this is the simple reference array.
// (Version B implements the same matmul as a systolic array and must match this.)
//
// Golden reference (testbench): numpy diff @ weight. Also reports total MAC-ops
// skipped by zero-skip across the array.

module pe_array_parallel #(
    parameter M = 4,
    parameter N = 4,
    parameter DIFF_WIDTH = 9,
    parameter W_WIDTH    = 8,
    parameter ACC_WIDTH  = 32
)(
    input  wire                          clk,
    input  wire                          rst,
    input  wire                          valid,           // accumulate this k-step
    input  wire [M*DIFF_WIDTH-1:0]       diff_col,        // diff[:,k], M elements
    input  wire [N*W_WIDTH-1:0]          w_row,           // weight[k,:], N elements
    output reg  signed [M*N*ACC_WIDTH-1:0] c_flat,        // M*N accumulators, packed
    output reg         [31:0]            skips_total      // MAC-ops skipped (zero-skip)
);
    integer i, j;
    reg signed [DIFF_WIDTH-1:0] d;
    reg signed [W_WIDTH-1:0]    w;
    reg signed [ACC_WIDTH-1:0]  cur;
    reg signed [ACC_WIDTH-1:0]  prod;
    reg        [31:0]           skip_inc;

    always @(posedge clk) begin
        if (rst) begin
            c_flat      <= 0;
            skips_total <= 0;
        end else if (valid) begin
            skip_inc = 0;
            for (i = 0; i < M; i = i + 1) begin
                d = diff_col[i*DIFF_WIDTH +: DIFF_WIDTH];
                for (j = 0; j < N; j = j + 1) begin
                    w   = w_row[j*W_WIDTH +: W_WIDTH];
                    cur = c_flat[(i*N+j)*ACC_WIDTH +: ACC_WIDTH];
                    if (d == 0) begin
                        prod = 0;                  // zero-skip
                        skip_inc = skip_inc + 1;
                    end else begin
                        prod = d * w;
                    end
                    c_flat[(i*N+j)*ACC_WIDTH +: ACC_WIDTH] <= cur + prod;
                end
            end
            skips_total <= skips_total + skip_inc;
        end
    end
endmodule

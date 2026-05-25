// vpu_restore.v
// VPU restore unit: reconstructs the full-precision activation from the difference
// result, i.e. the inverse of diff_generator.
//
//   act_curr[l] = act_prev[l] + diff_result[l]      (per lane)
//
// The PE computes in the difference domain (sparse, low-bit); before a non-linear op
// or before handing off to the next layer, the activation must be restored to full
// precision. restore is the exact inverse of the diff_generator's encode
// (diff = curr - prev), so restore(diff_generator(act)) == act -- differencing then
// restoring recovers the original activation, which is why difference-domain compute
// is lossless on linear layers (Ditto's premise). The testbench checks this identity.
//
// (The VPU also evaluates non-linearities like softmax/GELU on the restored full-
// precision activation, since those cannot be done in the difference domain; that
// table/polynomial datapath is noted as a further extension. Here we implement the
// restore accumulation, which is the part on Ditto's critical difference path.)
//
// An internal running accumulator holds act_prev; it starts at 0 (matching the
// diff_generator's prev=0 on the first step) and accumulates each step's diff.

module vpu_restore #(
    parameter LANES      = 4,
    parameter DIFF_WIDTH = 9,    // signed diff result per lane
    parameter ACT_WIDTH  = 16    // restored activation (headroom over int8)
)(
    input  wire                          clk,
    input  wire                          rst,        // sync: clears the running activation
    input  wire                          valid,      // a new diff result this step
    input  wire [LANES*DIFF_WIDTH-1:0]   diff_result,// from the PE (difference domain)
    output reg  [LANES*ACT_WIDTH-1:0]    act_curr    // restored full-precision activation
);
    integer i;
    reg signed [ACT_WIDTH-1:0]  acc [0:LANES-1];   // running activation per lane
    reg signed [DIFF_WIDTH-1:0] d;
    reg signed [ACT_WIDTH-1:0]  nxt;

    always @(posedge clk) begin
        if (rst) begin
            for (i = 0; i < LANES; i = i + 1) begin
                acc[i] <= 0;
                act_curr[i*ACT_WIDTH +: ACT_WIDTH] <= 0;
            end
        end else if (valid) begin
            for (i = 0; i < LANES; i = i + 1) begin
                d   = diff_result[i*DIFF_WIDTH +: DIFF_WIDTH];
                nxt = acc[i] + $signed(d);          // act_curr = act_prev + diff
                acc[i] <= nxt;
                act_curr[i*ACT_WIDTH +: ACT_WIDTH] <= nxt;
            end
        end
    end
endmodule

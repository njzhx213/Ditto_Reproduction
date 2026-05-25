// diff_generator.v
// The entry of the Ditto datapath: turns a stream of quantized activations into the
// temporal differences the Encoding Unit consumes.
//
//   diff[l] = curr_int8[l] - prev_int8[l]      (per lane)
//
// A prev-frame register holds the previous denoising step's activation; each step's
// curr is latched to become the next step's prev. On the first step there is no prev
// (register = 0), so diff = curr (the reference frame), flagged by `first_step`.
//
// This prev-frame register is the physical origin of the previous-frame DRAM traffic
// that Defo accounts for: difference mode must keep/re-fetch the previous activation,
// which is exactly the 2x activation cost in defo_unit. So the datapath is now
// complete end to end: activations -> diff_generator -> EU -> slot PE -> Defo.
//
// curr/prev are int8 (signed [-127,127]); diff is signed [-254,254], i.e. 9 bits,
// matching the EU's DIFF_WIDTH.

module diff_generator #(
    parameter LANES      = 4,
    parameter IN_WIDTH   = 8,    // signed int8 activation
    parameter DIFF_WIDTH = 9     // signed diff [-254,254]
)(
    input  wire                          clk,
    input  wire                          rst,        // sync: clears prev, arms first step
    input  wire                          valid,      // a new activation vector this cycle
    input  wire [LANES*IN_WIDTH-1:0]     curr_vec,   // current step activations (int8)
    output reg  [LANES*DIFF_WIDTH-1:0]   diff_vec,   // to the Encoding Unit
    output reg                           first_step  // 1 on the reference frame (no prev)
);
    integer i;
    reg signed [IN_WIDTH-1:0]   prev [0:LANES-1];
    reg                         have_prev;
    reg signed [IN_WIDTH-1:0]   c;
    reg signed [DIFF_WIDTH-1:0] d;

    always @(posedge clk) begin
        if (rst) begin
            for (i = 0; i < LANES; i = i + 1)
                prev[i] <= 0;
            have_prev  <= 1'b0;
            diff_vec   <= 0;
            first_step <= 1'b0;
        end else if (valid) begin
            for (i = 0; i < LANES; i = i + 1) begin
                c = curr_vec[i*IN_WIDTH +: IN_WIDTH];
                // first step: prev=0 -> diff = curr (baseline)
                d = $signed(c) - $signed(prev[i]);
                diff_vec[i*DIFF_WIDTH +: DIFF_WIDTH] <= d;
                prev[i] <= c;                // latch curr -> prev for next step
            end
            first_step <= ~have_prev;        // first valid after reset is the reference
            have_prev  <= 1'b1;
        end
    end
endmodule

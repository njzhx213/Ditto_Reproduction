// pe_diff_slot.v
// Ditto difference PE with the REAL 4-bit / >4-bit multiplier-slot micro-architecture
// (Fig 12). Consumes the full Encoding Unit output: sign, magnitude, is_zero, is_wide.
//
// Per lane:
//   is_zero -> 0 slots (zero-skip)
//   is_wide=0 (signed 4-bit, |diff|<=8) -> 1 slot: one 4-bit x 8-bit multiply
//   is_wide=1 (>4-bit)                  -> 2 slots: split magnitude into low/high
//        nibbles, two 4-bit x 8-bit multiplies, shift-add: mag*w = (hi*w)<<4 + lo*w
//
// This is why Ditto fits ~39398 small 4-bit PEs: most lanes are 1-slot. The PE also
// COUNTS the multiplier slots used per cycle, so the testbench can confirm the average
// slots/nonzero-lane matches the performance model's bit_factor (the RTL <-> perf-model
// quantitative tie-in). The arithmetic result is identical to a full multiply
// (verified against the naive dot product).
//
// is_wide MUST come from the EU (signed [-8,7] boundary) so the EU's classification
// and the PE's slot selection agree exactly.

module pe_diff_slot #(
    parameter LANES      = 4,
    parameter DIFF_WIDTH = 9,    // magnitude width (|diff| in [0,254])
    parameter W_WIDTH    = 8,
    parameter ACC_WIDTH  = 32,
    parameter SLOT_WIDTH = 8     // per-cycle slot count (max 2*LANES)
)(
    input  wire                          clk,
    input  wire                          rst,
    input  wire                          valid,
    input  wire [LANES-1:0]              sign_vec,
    input  wire [LANES*DIFF_WIDTH-1:0]   mag_vec,
    input  wire [LANES-1:0]              is_zero_vec,
    input  wire [LANES-1:0]              is_wide_vec,
    input  wire [LANES*W_WIDTH-1:0]      w_vec,
    output reg  signed [ACC_WIDTH-1:0]   acc,
    output reg         [SLOT_WIDTH-1:0]  slots_this_cycle,   // multiplier slots used
    output reg         [31:0]            slots_total         // running slot count
);
    integer i;
    reg signed [ACC_WIDTH-1:0]          sum_comb;
    reg        [SLOT_WIDTH-1:0]         slot_comb;
    reg        [DIFF_WIDTH-1:0]         mag;
    reg signed [W_WIDTH-1:0]            w;
    reg        [3:0]                    nib_lo, nib_hi;     // 4-bit nibbles
    reg signed [W_WIDTH+4:0]            p_lo, p_hi;         // nibble*weight (signed)
    reg signed [ACC_WIDTH-1:0]          magw;
    reg signed [ACC_WIDTH-1:0]          prod;

    always @(*) begin
        sum_comb  = 0;
        slot_comb = 0;
        for (i = 0; i < LANES; i = i + 1) begin
            mag = mag_vec[i*DIFF_WIDTH +: DIFF_WIDTH];
            w   = w_vec[i*W_WIDTH +: W_WIDTH];
            if (is_zero_vec[i]) begin
                magw = 0;                      // zero-skip: 0 slots
            end else if (!is_wide_vec[i]) begin
                // 1 slot: |diff|<=8 fits one 4-bit-unsigned x 8-bit multiplier
                magw = $signed({1'b0, mag}) * w;
                slot_comb = slot_comb + 8'd1;
            end else begin
                // 2 slots: split magnitude into low/high nibble, shift-add
                nib_lo = mag[3:0];
                nib_hi = mag[7:4];             // mag<=254 -> hi fits 4 bits
                p_lo = $signed({1'b0, nib_lo}) * w;
                p_hi = $signed({1'b0, nib_hi}) * w;
                magw = (p_hi <<< 4) + p_lo;    // recombine: hi*16 + lo
                slot_comb = slot_comb + 8'd2;
            end
            if (is_zero_vec[i])
                prod = 0;
            else if (sign_vec[i])
                prod = -magw;
            else
                prod = magw;
            sum_comb = sum_comb + prod;
        end
    end

    always @(posedge clk) begin
        if (rst) begin
            acc              <= 0;
            slots_total      <= 0;
            slots_this_cycle <= 0;
        end else if (valid) begin
            acc              <= acc + sum_comb;
            slots_this_cycle <= slot_comb;
            slots_total      <= slots_total + slot_comb;
        end
    end
endmodule

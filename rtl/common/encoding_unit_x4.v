// encoding_unit_x4.v
// 4-lane parallel Ditto Encoding Unit. Real Ditto processes LANES=4 4-bit elements
// per PE per cycle (Table III / Fig 12), so the EU feeding the PE encodes 4 diffs in
// parallel each cycle. This wraps four single-element encoding_unit instances; the
// per-lane logic is the already-verified classification (encoding_unit.v).
//
// Packed I/O so it connects cleanly to a PE row:
//   diff_vec    : 4 x DIFF_WIDTH signed diffs, packed
//   is_zero_vec : 4-bit, one zero-skip flag per lane
//   is_wide_vec : 4-bit, one >4-bit flag per lane
//   sign_vec    : 4-bit, sign per lane
//   mag_vec     : 4 x DIFF_WIDTH magnitudes, packed

module encoding_unit_x4 #(
    parameter DIFF_WIDTH = 9,
    parameter LANES      = 4
)(
    input  wire [LANES*DIFF_WIDTH-1:0] diff_vec,
    output wire [LANES-1:0]            is_zero_vec,
    output wire [LANES-1:0]            is_wide_vec,
    output wire [LANES-1:0]            sign_vec,
    output wire [LANES*DIFF_WIDTH-1:0] mag_vec
);
    genvar i;
    generate
        for (i = 0; i < LANES; i = i + 1) begin : lane
            encoding_unit #(.DIFF_WIDTH(DIFF_WIDTH)) eu (
                .diff      (diff_vec[i*DIFF_WIDTH +: DIFF_WIDTH]),
                .is_zero   (is_zero_vec[i]),
                .is_wide   (is_wide_vec[i]),
                .sign      (sign_vec[i]),
                .magnitude (mag_vec[i*DIFF_WIDTH +: DIFF_WIDTH])
            );
        end
    endgenerate
endmodule

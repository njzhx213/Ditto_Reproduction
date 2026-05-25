// encoding_unit.v
// Ditto Encoding Unit (paper Fig 11): classify a temporal-difference element and
// emit the encoding the PE array consumes.
//
// Golden reference (Python, bitwise-verified in the perf model):
//   diff == 0                     -> ZERO  (skipped by zero-skip)
//   -8 <= diff <= 7  (nonzero)    -> 4-bit (one multiplier slot)
//   otherwise                     -> >4-bit (two multiplier slots)
// where diff = curr_int8 - prev_int8, so diff is in [-254, 254] (needs >8 bits).
//
// Outputs per element:
//   is_zero   : 1 if diff == 0           (PE skips it entirely)
//   is_wide   : 1 if |range| > 4-bit     (PE allocates 2 slots / full precision)
//   sign      : sign bit of diff         (sign-magnitude for the shift-add PE)
//   magnitude : |diff|                   (full magnitude; PE uses low 4 bits when
//                                          !is_wide, full 8-bit window when is_wide)
//
// Pure combinational: the EU is the pipeline stage feeding the PE; one element per
// call here (a real array instantiates LANES of these in parallel).

module encoding_unit #(
    parameter DIFF_WIDTH = 9    // signed diff in [-254,254] fits in 9-bit signed
)(
    input  wire signed [DIFF_WIDTH-1:0] diff,      // curr_int8 - prev_int8
    output wire                          is_zero,
    output wire                          is_wide,   // >4-bit (outside [-8,7])
    output wire                          sign,      // 1 = negative
    output wire        [DIFF_WIDTH-1:0]  magnitude  // |diff|
);
    // sign-magnitude
    assign sign      = diff[DIFF_WIDTH-1];
    assign magnitude = sign ? (~diff + 1'b1) : diff;   // two's-complement abs

    // classification (matches Python classify_signed_range, signed range [-8,7])
    assign is_zero = (diff == 0);
    // 4-bit if -8 <= diff <= 7 ; wide otherwise
    assign is_wide = (diff > 9'sd7) || (diff < -9'sd8);
endmodule

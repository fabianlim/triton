// RUN: spyre-triton-opt %s --lower-spyre-ops -split-input-file -verify-diagnostics | FileCheck %s

// Tests for the rule that turns `arith.cmpf` + `arith.select` into spyreop ops.
//
// Background, in the order the tests need it.
//
// `arith.select %c, %p, %q` takes %p where %c is true. Its condition %c is an
// `i1`, and the Spyre device has no `i1`: `spyreop.select` instead takes a
// condition of the SAME FLOAT TYPE as the values it is choosing between, and
// takes the first one wherever that float is not zero. So a `1.0` condition
// chooses %p and a `0.0` condition chooses %q.
//
// `spyreop.compare` produces exactly those floats: 1.0 where its predicate holds
// and 0.0 where it does not, in the width it compared. So the usual lowering is a
// pair -- compare producing the float, select consuming it:
//
//     %c = spyreop.compare <greaterthan> %x, %y : f32     // 1.0 or 0.0
//     %s = spyreop.select %c, %p, %q : f32                // picks on non-zero
//
// One case does better. When the kernel's own comparison is already "is this
// value non-zero", `spyreop.select` performs that same test on its condition, so
// the comparison is doing work the select repeats. The rule then drops the
// comparison and hands the select the value that was being compared -- leaving a
// select with NO compare beside it. `tl.where(mask != 0, p, q)` is that case, and
// it is the shape a kernel reads a stored mask back in.
//
// Each test below states which of those two outputs it expects and why.

#map = affine_map<(d0) -> (d0)>

// PURPOSE: the ordinary lowering, where the kernel compares two data values.
//
// `x > y` is not a test against zero, so nothing can be dropped: the comparison
// has to run, and its float answer feeds the select. Asserts BOTH ops appear, and
// that the select reads the compare's result rather than one of the inputs -- a
// rule that wired the wrong operand in would still emit two ops and pass a weaker
// check.
// CHECK-LABEL:   func.func @compare_two_values_keeps_both_ops(
// CHECK:           ^bb0(%[[X:.*]]: f32, %[[Y:.*]]: f32, %[[P:.*]]: f32, %[[Q:.*]]: f32, %{{.*}}: f32):
// CHECK-NEXT:        %[[C:.*]] = spyreop.compare <greaterthan> %[[X]], %[[Y]] : f32
// CHECK-NEXT:        %[[S:.*]] = spyreop.select %[[C]], %[[P]], %[[Q]] : f32
// CHECK-NEXT:        linalg.yield %[[S]] : f32
func.func @compare_two_values_keeps_both_ops(%a: tensor<8xf32>, %b: tensor<8xf32>,
                                             %p: tensor<8xf32>, %q: tensor<8xf32>) -> tensor<8xf32> {
  %init = tensor.empty() : tensor<8xf32>
  %0 = linalg.generic {indexing_maps = [#map, #map, #map, #map, #map],
                       iterator_types = ["parallel"]}
      ins(%a, %b, %p, %q : tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>)
      outs(%init : tensor<8xf32>) {
  ^bb0(%x: f32, %y: f32, %t: f32, %f: f32, %o: f32):
    %c = arith.cmpf ogt, %x, %y : f32
    %s = arith.select %c, %t, %f : f32
    linalg.yield %s : f32
  } -> tensor<8xf32>
  return %0 : tensor<8xf32>
}

// -----

// PURPOSE: the shape a kernel reading a stored mask writes, and the one case that
// lowers to a select with no compare at all.
//
// `mask != 0` asks whether the value is non-zero, which is what `spyreop.select`
// does to its condition. So the comparison is redundant: the rule deletes it and
// gives the select the mask directly. `CHECK-NOT: spyreop.compare` is the whole
// point of the test -- the emitted body is one op, and the operand list shows the
// mask arriving as the condition unchanged.
//
// `one` is arith's ORDERED not-equal. The unordered spelling is covered below.
#map = affine_map<(d0) -> (d0)>
// CHECK-LABEL:   func.func @mask_not_equal_zero_needs_no_compare(
// CHECK:           ^bb0(%[[M:.*]]: f32, %[[P:.*]]: f32, %[[Q:.*]]: f32, %{{.*}}: f32):
// CHECK-NEXT:        %[[S:.*]] = spyreop.select %[[M]], %[[P]], %[[Q]] : f32
// CHECK-NEXT:        linalg.yield %[[S]] : f32
// CHECK-NOT:       spyreop.compare
func.func @mask_not_equal_zero_needs_no_compare(%m: tensor<8xf32>, %p: tensor<8xf32>,
                                                %q: tensor<8xf32>) -> tensor<8xf32> {
  %zero = arith.constant 0.0 : f32
  %init = tensor.empty() : tensor<8xf32>
  %0 = linalg.generic {indexing_maps = [#map, #map, #map, #map],
                       iterator_types = ["parallel"]}
      ins(%m, %p, %q : tensor<8xf32>, tensor<8xf32>, tensor<8xf32>)
      outs(%init : tensor<8xf32>) {
  ^bb0(%mask: f32, %t: f32, %f: f32, %o: f32):
    %c = arith.cmpf one, %mask, %zero : f32
    %s = arith.select %c, %t, %f : f32
    linalg.yield %s : f32
  } -> tensor<8xf32>
  return %0 : tensor<8xf32>
}

// -----

// PURPOSE: pin that `m != 0` written in Triton reaches the same one-op form.
//
// Triton's `!=` emits `une`, arith's UNORDERED not-equal, which differs from the
// ordered `one` above only for a NaN input. The device's comparison is ordered, so
// dropping `une` assumes no NaN reaches the mask -- which holds for a Spyre tensor
// and is recorded at the rule. Without this case the previous test would pass
// while every kernel actually written `m != 0` took the two-op path.
#map = affine_map<(d0) -> (d0)>
// CHECK-LABEL:   func.func @triton_not_equal_reaches_the_same_form(
// CHECK:           ^bb0(%[[M:.*]]: f32, %[[P:.*]]: f32, %[[Q:.*]]: f32, %{{.*}}: f32):
// CHECK-NEXT:        spyreop.select %[[M]], %[[P]], %[[Q]] : f32
// CHECK-NOT:       spyreop.compare
func.func @triton_not_equal_reaches_the_same_form(%m: tensor<8xf32>, %p: tensor<8xf32>,
                                                  %q: tensor<8xf32>) -> tensor<8xf32> {
  %zero = arith.constant 0.0 : f32
  %init = tensor.empty() : tensor<8xf32>
  %0 = linalg.generic {indexing_maps = [#map, #map, #map, #map],
                       iterator_types = ["parallel"]}
      ins(%m, %p, %q : tensor<8xf32>, tensor<8xf32>, tensor<8xf32>)
      outs(%init : tensor<8xf32>) {
  ^bb0(%mask: f32, %t: f32, %f: f32, %o: f32):
    %c = arith.cmpf une, %mask, %zero : f32
    %s = arith.select %c, %t, %f : f32
    linalg.yield %s : f32
  } -> tensor<8xf32>
  return %0 : tensor<8xf32>
}

// -----

// PURPOSE: catch an inverted condition, which is the one way this rule can be
// wrong while still looking right.
//
// `m == 0` is true exactly where `spyreop.select` would choose the SECOND value,
// so handing the select the mask directly picks the opposite arm. The rule
// compensates by swapping the two values as it builds the op, and the check below
// asserts that order: the kernel writes `select (m == 0) ? P : Q` and the emitted
// op must be `spyreop.select %mask, Q, P`.
//
// Why this test carries its weight: a version of the rule that forgot the swap
// would emit one clean op, pass a `CHECK-NOT: spyreop.compare`, and silently
// return the wrong value for every lane. The operand ORDER is the only thing that
// distinguishes correct from inverted.
#map = affine_map<(d0) -> (d0)>
// CHECK-LABEL:   func.func @mask_equal_zero_swaps_the_values(
// CHECK:           ^bb0(%[[M:.*]]: f32, %[[P:.*]]: f32, %[[Q:.*]]: f32, %{{.*}}: f32):
// CHECK-NEXT:        spyreop.select %[[M]], %[[Q]], %[[P]] : f32
// CHECK-NOT:       spyreop.compare
func.func @mask_equal_zero_swaps_the_values(%m: tensor<8xf32>, %p: tensor<8xf32>,
                                            %q: tensor<8xf32>) -> tensor<8xf32> {
  %zero = arith.constant 0.0 : f32
  %init = tensor.empty() : tensor<8xf32>
  %0 = linalg.generic {indexing_maps = [#map, #map, #map, #map],
                       iterator_types = ["parallel"]}
      ins(%m, %p, %q : tensor<8xf32>, tensor<8xf32>, tensor<8xf32>)
      outs(%init : tensor<8xf32>) {
  ^bb0(%mask: f32, %t: f32, %f: f32, %o: f32):
    %c = arith.cmpf oeq, %mask, %zero : f32
    %s = arith.select %c, %t, %f : f32
    linalg.yield %s : f32
  } -> tensor<8xf32>
  return %0 : tensor<8xf32>
}

// -----

// PURPOSE: pin the boundary of the one-op form, so it is not widened to a
// comparison that looks similar but computes something else.
//
// `m > 0` also mentions zero, but it is FALSE for a negative value where "not
// zero" is TRUE -- so dropping it would change the result wherever a mask holds a
// negative number. A mask arriving from memory could hold anything, so the rule
// keeps the comparison and emits the two-op form. `oge`, `olt` and `ole` are
// excluded for the same reason and are not repeated here.
#map = affine_map<(d0) -> (d0)>
// CHECK-LABEL:   func.func @greater_than_zero_keeps_its_compare(
// CHECK:           spyreop.compare <greaterthan> {{.*}} : f32
// CHECK:           spyreop.select {{.*}} : f32
func.func @greater_than_zero_keeps_its_compare(%m: tensor<8xf32>, %p: tensor<8xf32>,
                                               %q: tensor<8xf32>) -> tensor<8xf32> {
  %zero = arith.constant 0.0 : f32
  %init = tensor.empty() : tensor<8xf32>
  %0 = linalg.generic {indexing_maps = [#map, #map, #map, #map],
                       iterator_types = ["parallel"]}
      ins(%m, %p, %q : tensor<8xf32>, tensor<8xf32>, tensor<8xf32>)
      outs(%init : tensor<8xf32>) {
  ^bb0(%mask: f32, %t: f32, %f: f32, %o: f32):
    %c = arith.cmpf ogt, %mask, %zero : f32
    %s = arith.select %c, %t, %f : f32
    linalg.yield %s : f32
  } -> tensor<8xf32>
  return %0 : tensor<8xf32>
}

// -----

// PURPOSE: pin that the one-op form needs the comparison to be against ZERO, not
// merely against a constant.
//
// `m != 2.0` is a real question the device does not answer for free, so the
// comparison stays. Without this case the rule could be keyed on "compares
// against a constant" and still pass everything above.
#map = affine_map<(d0) -> (d0)>
// CHECK-LABEL:   func.func @nonzero_constant_keeps_its_compare(
// CHECK:           spyreop.compare <notequal> {{.*}} : f32
// CHECK:           spyreop.select {{.*}} : f32
func.func @nonzero_constant_keeps_its_compare(%m: tensor<8xf32>, %p: tensor<8xf32>,
                                              %q: tensor<8xf32>) -> tensor<8xf32> {
  %two = arith.constant 2.0 : f32
  %init = tensor.empty() : tensor<8xf32>
  %0 = linalg.generic {indexing_maps = [#map, #map, #map, #map],
                       iterator_types = ["parallel"]}
      ins(%m, %p, %q : tensor<8xf32>, tensor<8xf32>, tensor<8xf32>)
      outs(%init : tensor<8xf32>) {
  ^bb0(%mask: f32, %t: f32, %f: f32, %o: f32):
    %c = arith.cmpf one, %mask, %two : f32
    %s = arith.select %c, %t, %f : f32
    linalg.yield %s : f32
  } -> tensor<8xf32>
  return %0 : tensor<8xf32>
}

// -----

// PURPOSE: show that one comparison serving TWO consumers is fully lowered, and
// record what that costs.
//
// The comparison here feeds both a cast and a select, and the two have separate
// rules. Each rule rewrites its own pair, and since neither can know the other
// fired, each BUILDS ITS OWN `spyreop.compare` -- so the body comes out with two
// identical compares rather than one shared. That is what the count below
// pins. CSE merges them immediately afterwards (the pipeline runs one), so the
// duplication does not reach the device; asserting it here rather than asserting
// the post-CSE shape keeps this test about the rules and not about CSE.
//
// What matters either way is that no `arith` op survives: both pairs are consumed,
// so no `i1` is left. This case previously lived in compare-invalid.mlir as an
// expected FAILURE -- with no select rule the select kept the `i1` alive and the
// kernel was refused -- and it is here now because that is no longer true.
#map = affine_map<(d0) -> (d0)>
// CHECK-LABEL:   func.func @one_compare_two_consumers(
// CHECK-COUNT-2:   spyreop.compare <equal> {{.*}} : f16
// CHECK:           spyreop.select {{.*}} : f16
// CHECK-NOT:       arith.cmpf
// CHECK-NOT:       arith.select
func.func @one_compare_two_consumers(%x: tensor<4xf16>, %y: tensor<4xf16>) -> tensor<4xf16> {
  %init = tensor.empty() : tensor<4xf16>
  %0 = linalg.generic {indexing_maps = [#map, #map, #map],
                       iterator_types = ["parallel"]}
      ins(%x, %y : tensor<4xf16>, tensor<4xf16>) outs(%init : tensor<4xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %c = arith.cmpf oeq, %a, %b : f16
    %f = arith.uitofp %c : i1 to f16
    %s = arith.select %c, %f, %b : f16
    linalg.yield %s : f16
  } -> tensor<4xf16>
  return %0 : tensor<4xf16>
}

// -----

// PURPOSE: show what the rule does when it cannot apply, and that it declines
// QUIETLY here rather than reporting.
//
// The condition is an `i1` the kernel was handed as an operand, not one a
// comparison in this body produced, so there is no float for `spyreop.select` to
// use and nothing this rule can build. The select is left exactly as it was.
//
// And no diagnostic, deliberately: an `i1` arriving as a block argument is the
// TENSOR form crossing into a body, which an earlier pass is responsible for
// removing, so reporting it here would blame this pass for IR it did not shape.
// The pass reports only an `i1` made and read inside one body -- the next case.
#map = affine_map<(d0) -> (d0)>
// CHECK-LABEL:   func.func @condition_from_outside_is_left_alone(
// CHECK:           arith.select
// CHECK-NOT:       spyreop.select
func.func @condition_from_outside_is_left_alone(%c: tensor<8xi1>, %p: tensor<8xf32>,
                                              %q: tensor<8xf32>) -> tensor<8xf32> {
  %init = tensor.empty() : tensor<8xf32>
  %0 = linalg.generic {indexing_maps = [#map, #map, #map, #map],
                       iterator_types = ["parallel"]}
      ins(%c, %p, %q : tensor<8xi1>, tensor<8xf32>, tensor<8xf32>)
      outs(%init : tensor<8xf32>) {
  ^bb0(%cond: i1, %t: f32, %f: f32, %o: f32):
    %s = arith.select %cond, %t, %f : f32
    linalg.yield %s : f32
  } -> tensor<8xf32>
  return %0 : tensor<8xf32>
}

// -----

// PURPOSE: pin that the values being selected must be a type the device has a
// select for, and that an integer ternary is refused rather than mislowered.
//
// `spyreop.select` exists only for the float widths, so an i32 ternary has no
// device form and this rule declines. The comparison feeding it is then left with
// no selectable consumer, so its `i1` survives and is reported -- the same
// mechanism as the previous case, reached a different way.
#map = affine_map<(d0) -> (d0)>
// No CHECK lines: the module does not reach FileCheck when the pass reports, so
// `-verify-diagnostics` and the annotations below are the whole assertion.
func.func @integer_values_are_refused(%a: tensor<8xf32>, %b: tensor<8xf32>,
                                      %p: tensor<8xi32>, %q: tensor<8xi32>) -> tensor<8xi32> {
  %init = tensor.empty() : tensor<8xi32>
  %0 = linalg.generic {indexing_maps = [#map, #map, #map, #map, #map],
                       iterator_types = ["parallel"]}
      ins(%a, %b, %p, %q : tensor<8xf32>, tensor<8xf32>, tensor<8xi32>, tensor<8xi32>)
      outs(%init : tensor<8xi32>) {
  ^bb0(%x: f32, %y: f32, %t: i32, %f: i32, %o: i32):
    // expected-error @below {{an i1 value survives inside a compute body}}
    // expected-note @below {{the predicate 'ogt' does have a spyreop.compare counterpart}}
    %c = arith.cmpf ogt, %x, %y : f32
    // expected-note @below {{read here, by 'arith.select'}}
    %s = arith.select %c, %t, %f : i32
    linalg.yield %s : i32
  } -> tensor<8xi32>
  return %0 : tensor<8xi32>
}

// RUN: spyre-triton-opt %s --combine-spyre-ops -split-input-file | FileCheck %s

// The one tenant of the spyreop combiner: a `spyreop.realdiv` whose NUMERATOR is
// a constant one becomes the unary `spyreop.reciprocal`, taking the float
// immediate out of the program entirely. Every other numerator keeps the binary
// op, and most of this file holds that line -- plus the three cases that are
// about the pass being a REWRITE rather than a conversion and about where its
// input comes from: the orphaned constant goes by ordinary dead-op elimination, a
// numerator someone else still reads survives with no guard asking about it, and
// the block-argument form is declined.
//
// `func.func`, not `tt.func`, because that is what this pass actually sees: it
// runs at the tail of the `spyrecode` stage, long after ConvertFunctions.

// A scalar f16 `1.0 / x` -> spyreop.reciprocal, and the constant goes with it:
// nothing reads it once the divide is gone, and the greedy driver erases a
// trivially dead op. No pattern here asks about uses.
// CHECK-LABEL:   func.func @recip_f16(
// CHECK-SAME:  %[[X:.*]]: f16) -> f16 {
// CHECK-NOT:       arith.constant
// CHECK-NOT:       spyreop.realdiv
// CHECK:           %[[R:.*]] = spyreop.reciprocal %[[X]] : f16
// CHECK:           return %[[R]] : f16
func.func @recip_f16(%x: f16) -> f16 {
  %one = arith.constant 1.0 : f16
  %0 = spyreop.realdiv %one, %x : f16
  return %0 : f16
}

// -----

// Same at f32: the rewrite does not branch on the float width.
// CHECK-LABEL:   func.func @recip_f32(
// CHECK-SAME:  %[[X:.*]]: f32) -> f32 {
// CHECK-NOT:       arith.constant
// CHECK-NOT:       spyreop.realdiv
// CHECK:           %[[R:.*]] = spyreop.reciprocal %[[X]] : f32
// CHECK:           return %[[R]] : f32
func.func @recip_f32(%x: f32) -> f32 {
  %one = arith.constant 1.0 : f32
  %0 = spyreop.realdiv %one, %x : f32
  return %0 : f32
}

// -----

// `reduce/softmax_on_stick`'s own shape, and the reason the match is on the
// VALUE rather than on a shape: the divide is scalar, inside a linalg.generic
// body, and its `1.0` is a scalar `arith.constant` HOISTED ABOVE the generic.
// Nothing about the enclosing tensor shape is visible from the op, and nothing
// here looks.
// CHECK-LABEL:   func.func @recip_f16_hoisted_constant(
// CHECK-SAME:  %[[T:.*]]: tensor<4x1xf16>) -> tensor<4x1xf16> {
// CHECK-NOT:       arith.constant 1
// CHECK-NOT:       spyreop.realdiv
// CHECK:           linalg.generic
// CHECK:           ^bb0(%[[IN:.*]]: f16, %[[OUT:.*]]: f16):
// CHECK:             %[[R:.*]] = spyreop.reciprocal %[[IN]] : f16
// CHECK:             linalg.yield %[[R]] : f16
func.func @recip_f16_hoisted_constant(%t: tensor<4x1xf16>) -> tensor<4x1xf16> {
  %one = arith.constant 1.0 : f16
  %init = tensor.empty() : tensor<4x1xf16>
  %0 = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                       affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]}
      ins(%t : tensor<4x1xf16>) outs(%init : tensor<4x1xf16>) {
  ^bb0(%in: f16, %out: f16):
    %1 = spyreop.realdiv %one, %in : f16
    linalg.yield %1 : f16
  } -> tensor<4x1xf16>
  return %0 : tensor<4x1xf16>
}

// -----

// THE PRECONDITION, as a test rather than only as prose. The same kernel before
// FoldDataMovementGenerics has folded the splat's producer into the body: the
// `1.0` arrives as an `ins` OPERAND and the numerator in the body is a BLOCK
// ARGUMENT. A block argument is not a constant, so this pass declines and the
// realdiv survives with its immediate. Converting this form would mean dropping
// an operand, its block argument and its indexing map from the generic, which is
// a rewrite of the generic and not of a spyreop op -- see rule 1 in Passes.td.
// This case is what makes the ordering constraint against
// FoldDataMovementGenerics observable.
// CHECK-LABEL:   func.func @realdiv_splat_ins_numerator_declined(
// CHECK-NOT:       spyreop.reciprocal
// CHECK:           %[[SPLAT:.*]] = arith.constant dense<1.000000e+00> : tensor<4x1xf16>
// CHECK:           linalg.generic
// CHECK:           ^bb0(%[[ONE:.*]]: f16, %[[IN:.*]]: f16, %[[OUT:.*]]: f16):
// CHECK:             %[[R:.*]] = spyreop.realdiv %[[ONE]], %[[IN]] : f16
func.func @realdiv_splat_ins_numerator_declined(%t: tensor<4x1xf16>) -> tensor<4x1xf16> {
  %splat = arith.constant dense<1.0> : tensor<4x1xf16>
  %init = tensor.empty() : tensor<4x1xf16>
  %0 = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                       affine_map<(d0, d1) -> (d0, d1)>,
                       affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]}
      ins(%splat, %t : tensor<4x1xf16>, tensor<4x1xf16>) outs(%init : tensor<4x1xf16>) {
  ^bb0(%one: f16, %in: f16, %out: f16):
    %1 = spyreop.realdiv %one, %in : f16
    linalg.yield %1 : f16
  } -> tensor<4x1xf16>
  return %0 : tensor<4x1xf16>
}

// -----

// A numerator with ANOTHER reader is rewritten just the same, and the constant
// stays because something still needs it. This is the case the version of this
// peephole inside LowerSpyreOps carried a hand-rolled `hasOneUse()` guard and an
// explicit eraseOp for; as a rewrite pattern nothing asks, and dead-op
// elimination answers correctly either way.
// CHECK-LABEL:   func.func @recip_shared_numerator(
// CHECK-SAME:  %[[X:.*]]: f16) -> (f16, f16) {
// CHECK-NOT:       spyreop.realdiv
// CHECK:           %[[ONE:.*]] = arith.constant 1.000000e+00 : f16
// CHECK:           %[[R:.*]] = spyreop.reciprocal %[[X]] : f16
// The other reader's operands come back swapped: the greedy driver folds as well
// as rewrites, and the folder sorts a commutative op's constant operand last.
// Incidental to this pass, and checked in the order it actually prints so the
// case is not asserting something it does not control.
// CHECK:           %[[S:.*]] = arith.addf %[[X]], %[[ONE]] : f16
// CHECK:           return %[[R]], %[[S]] : f16, f16
func.func @recip_shared_numerator(%x: f16) -> (f16, f16) {
  %one = arith.constant 1.0 : f16
  %0 = spyreop.realdiv %one, %x : f16
  %1 = arith.addf %one, %x : f16
  return %0, %1 : f16, f16
}

// -----

// A constant numerator that is NOT one keeps spyreop.realdiv, constant and all.
// CHECK-LABEL:   func.func @realdiv_two_over_x_f16(
// CHECK-SAME:  %[[X:.*]]: f16) -> f16 {
// CHECK-NOT:       spyreop.reciprocal
// CHECK:           %[[TWO:.*]] = arith.constant 2.000000e+00 : f16
// CHECK:           %[[R:.*]] = spyreop.realdiv %[[TWO]], %[[X]] : f16
func.func @realdiv_two_over_x_f16(%x: f16) -> f16 {
  %two = arith.constant 2.0 : f16
  %0 = spyreop.realdiv %two, %x : f16
  return %0 : f16
}

// -----

// A non-constant numerator keeps spyreop.realdiv too. Nothing to match, so
// nothing to remove.
// CHECK-LABEL:   func.func @realdiv_var_over_x_f16(
// CHECK-SAME:  %[[A:.*]]: f16, %[[B:.*]]: f16) -> f16 {
// CHECK-NOT:       spyreop.reciprocal
// CHECK:           %[[R:.*]] = spyreop.realdiv %[[A]], %[[B]] : f16
func.func @realdiv_var_over_x_f16(%a: f16, %b: f16) -> f16 {
  %0 = spyreop.realdiv %a, %b : f16
  return %0 : f16
}

// -----

// The match is on the numerator specifically, not on "an operand is constant": a
// constant DENOMINATOR still needs a binary op, and gets one. A denominator of
// exactly 1.0 is the sharpest form of that -- there is no folder on a spyreop op,
// so unlike the arith.divf version of this test the constant can be the very
// value the pattern looks for and the position is all that decides.
// CHECK-LABEL:   func.func @realdiv_x_over_one_f16(
// CHECK-SAME:  %[[X:.*]]: f16) -> f16 {
// CHECK-NOT:       spyreop.reciprocal
// CHECK:           %[[ONE:.*]] = arith.constant 1.000000e+00 : f16
// CHECK:           %[[R:.*]] = spyreop.realdiv %[[X]], %[[ONE]] : f16
func.func @realdiv_x_over_one_f16(%x: f16) -> f16 {
  %one = arith.constant 1.0 : f16
  %0 = spyreop.realdiv %x, %one : f16
  return %0 : f16
}

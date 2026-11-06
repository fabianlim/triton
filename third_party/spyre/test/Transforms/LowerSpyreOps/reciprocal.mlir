// RUN: spyre-triton-opt %s --lower-spyre-ops -split-input-file | FileCheck %s
// RUN: spyre-triton-opt %s --canonicalize-generic-operands --lower-spyre-ops -split-input-file | FileCheck %s --check-prefix=CANON

// `arith.divf` with a constant 1.0 numerator -> `spyreop.reciprocal`; any other
// numerator -> `spyreop.realdiv`. Two rules on `arith.divf`, the reciprocal
// tried first.
//
// The rule reads the numerator as the body holds it: a scalar 1.0 constant. A
// splat `ins` operand reaches the body as a block argument, which the rule does
// not look through; CanonicalizeGenericOperands' SplatOperandToScalar is what
// turns it into a scalar in the body, so the splat 1.0 cases are checked on a
// RUN line of their own that runs that pass first. Every other case is checked
// with --lower-spyre-ops alone.
//
// `func.func`, not `tt.func`, because that is what this pass sees: it runs in the
// `spyrecode` stage, long after ConvertFunctions.

// THE SPLAT `ins` FORM -- `tl.full([M, S], 1.0)` as it actually arrives. Two `ins`
// go in and one comes out, the constant gone entirely: CanonicalizeGenericOperands
// puts the 1.0 into the body as a scalar, the rule reads it there, and the scalar
// is then dead.
// CANON-LABEL:   func.func @recip_splat_ins(
// CANON-SAME:  %[[T:.*]]: tensor<4x1xf16>) -> tensor<4x1xf16> {
// CANON-NOT:       arith.constant
// CANON-NOT:       arith.divf
// CANON:           %[[E:.*]] = tensor.empty() : tensor<4x1xf16>
// CANON:           linalg.generic {{.*}} ins(%[[T]] : tensor<4x1xf16>) outs(%[[E]] : tensor<4x1xf16>)
// CANON:           ^bb0(%[[IN:.*]]: f16, %[[OUT:.*]]: f16):
// CANON:             %[[R:.*]] = spyreop.reciprocal %[[IN]] : f16
// CANON:             linalg.yield %[[R]] : f16
func.func @recip_splat_ins(%t: tensor<4x1xf16>) -> tensor<4x1xf16> {
  %splat = arith.constant dense<1.0> : tensor<4x1xf16>
  %init = tensor.empty() : tensor<4x1xf16>
  %0 = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                       affine_map<(d0, d1) -> (d0, d1)>,
                       affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]}
      ins(%splat, %t : tensor<4x1xf16>, tensor<4x1xf16>) outs(%init : tensor<4x1xf16>) {
  ^bb0(%one: f16, %in: f16, %out: f16):
    %1 = arith.divf %one, %in : f16
    linalg.yield %1 : f16
  } -> tensor<4x1xf16>
  return %0 : tensor<4x1xf16>
}

// -----

// THE HOISTED-SCALAR FORM -- the same kernel once something has folded the splat
// into the body, which is the form the rule reads. There is no operand to erase
// here.
// CHECK-LABEL:   func.func @recip_hoisted_scalar(
// CHECK-SAME:  %[[T:.*]]: tensor<4x1xf16>) -> tensor<4x1xf16> {
// CHECK-NOT:       arith.constant
// CHECK-NOT:       arith.divf
// CHECK:           ^bb0(%[[IN:.*]]: f16, %[[OUT:.*]]: f16):
// CHECK:             %[[R:.*]] = spyreop.reciprocal %[[IN]] : f16
func.func @recip_hoisted_scalar(%t: tensor<4x1xf16>) -> tensor<4x1xf16> {
  %one = arith.constant 1.0 : f16
  %init = tensor.empty() : tensor<4x1xf16>
  %0 = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                       affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]}
      ins(%t : tensor<4x1xf16>) outs(%init : tensor<4x1xf16>) {
  ^bb0(%in: f16, %out: f16):
    %1 = arith.divf %one, %in : f16
    linalg.yield %1 : f16
  } -> tensor<4x1xf16>
  return %0 : tensor<4x1xf16>
}

// -----

// f32 as well as f16: the rule does not branch on the float width.
// CANON-LABEL:   func.func @recip_f32(
// CANON:             spyreop.reciprocal
func.func @recip_f32(%t: tensor<4xf32>) -> tensor<4xf32> {
  %splat = arith.constant dense<1.0> : tensor<4xf32>
  %init = tensor.empty() : tensor<4xf32>
  %0 = linalg.generic {
      indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>,
                       affine_map<(d0) -> (d0)>],
      iterator_types = ["parallel"]}
      ins(%splat, %t : tensor<4xf32>, tensor<4xf32>) outs(%init : tensor<4xf32>) {
  ^bb0(%one: f32, %in: f32, %out: f32):
    %1 = arith.divf %one, %in : f32
    linalg.yield %1 : f32
  } -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----

// A shared numerator is rewritten the same, and the constant stays because
// something still reads it. Nothing in the rule asks about use counts: the
// numerator goes, or does not go, by ordinary dead-op elimination.
// CHECK-LABEL:   func.func @recip_shared_numerator(
// CHECK-NOT:       arith.divf
// CHECK:           %[[ONE:.*]] = arith.constant 1.000000e+00 : f16
// CHECK:             %[[R:.*]] = spyreop.reciprocal %[[IN:.*]] : f16
// CHECK:             arith.addf
func.func @recip_shared_numerator(%t: tensor<4xf16>) -> tensor<4xf16> {
  %one = arith.constant 1.0 : f16
  %init = tensor.empty() : tensor<4xf16>
  %0 = linalg.generic {
      indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
      iterator_types = ["parallel"]}
      ins(%t : tensor<4xf16>) outs(%init : tensor<4xf16>) {
  ^bb0(%in: f16, %out: f16):
    %1 = arith.divf %one, %in : f16
    %2 = arith.addf %1, %one : f16
    linalg.yield %2 : f16
  } -> tensor<4xf16>
  return %0 : tensor<4xf16>
}

// -----

// The same divide outside any generic body is left as it is: the rule applies
// only to an op whose immediate parent is a linalg.generic.
// CHECK-LABEL:   func.func @recip_outside_a_body(
// CHECK-NOT:       spyreop
// CHECK:           %[[ONE:.*]] = arith.constant 1.000000e+00 : f16
// CHECK:           %[[R:.*]] = arith.divf %[[ONE]], %{{.*}} : f16
func.func @recip_outside_a_body(%x: f16) -> f16 {
  %one = arith.constant 1.0 : f16
  %0 = arith.divf %one, %x : f16
  return %0 : f16
}

// -----

// A numerator that is not one gets the BINARY intrinsic, constant and all: the
// rule matches on the VALUE, not on "there is a constant on the left".
// CHECK-LABEL:   func.func @divf_two_over_x(
// CHECK-NOT:       spyreop.reciprocal
// CHECK:           spyreop.realdiv
func.func @divf_two_over_x(%t: tensor<4xf16>) -> tensor<4xf16> {
  %splat = arith.constant dense<2.0> : tensor<4xf16>
  %init = tensor.empty() : tensor<4xf16>
  %0 = linalg.generic {
      indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>,
                       affine_map<(d0) -> (d0)>],
      iterator_types = ["parallel"]}
      ins(%splat, %t : tensor<4xf16>, tensor<4xf16>) outs(%init : tensor<4xf16>) {
  ^bb0(%two: f16, %in: f16, %out: f16):
    %1 = arith.divf %two, %in : f16
    linalg.yield %1 : f16
  } -> tensor<4xf16>
  return %0 : tensor<4xf16>
}

// -----

// The numerator specifically, not "an operand is constant": a constant
// DENOMINATOR still needs the binary op.
//
// The constant is 2.0 and not 1.0, which would have been the sharper test of
// position: with the splat folded into the body `x / 1.0` folds to `x` by arith's
// own folder before any rule sees it, and the case would pass for the wrong reason.
// CHECK-LABEL:   func.func @divf_x_over_two(
// CHECK-NOT:       spyreop.reciprocal
// CHECK:           spyreop.realdiv
func.func @divf_x_over_two(%t: tensor<4xf16>) -> tensor<4xf16> {
  %splat = arith.constant dense<2.0> : tensor<4xf16>
  %init = tensor.empty() : tensor<4xf16>
  %0 = linalg.generic {
      indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>,
                       affine_map<(d0) -> (d0)>],
      iterator_types = ["parallel"]}
      ins(%t, %splat : tensor<4xf16>, tensor<4xf16>) outs(%init : tensor<4xf16>) {
  ^bb0(%in: f16, %one: f16, %out: f16):
    %1 = arith.divf %in, %one : f16
    linalg.yield %1 : f16
  } -> tensor<4xf16>
  return %0 : tensor<4xf16>
}

// -----

// A NON-SPLAT constant tensor whose element 0 is 1.0 is not a one: it reaches the
// body as a block argument, which the rule does not read through, so element 0
// is never taken for the whole tensor.
// CHECK-LABEL:   func.func @divf_nonsplat_numerator(
// CHECK-NOT:       spyreop.reciprocal
// CHECK:           spyreop.realdiv
func.func @divf_nonsplat_numerator(%t: tensor<2xf16>) -> tensor<2xf16> {
  %mixed = arith.constant dense<[1.0, 3.0]> : tensor<2xf16>
  %init = tensor.empty() : tensor<2xf16>
  %0 = linalg.generic {
      indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>,
                       affine_map<(d0) -> (d0)>],
      iterator_types = ["parallel"]}
      ins(%mixed, %t : tensor<2xf16>, tensor<2xf16>) outs(%init : tensor<2xf16>) {
  ^bb0(%n: f16, %in: f16, %out: f16):
    %1 = arith.divf %n, %in : f16
    linalg.yield %1 : f16
  } -> tensor<2xf16>
  return %0 : tensor<2xf16>
}

// -----

// An unsupported float width has no intrinsic, so BOTH rules decline and the
// divide flows through to the backend as arith. The only case in this file where
// no spyreop op appears at all -- see pass-through.mlir for the rest.
// CHECK-LABEL:   func.func @divf_f64_declined(
// CHECK-NOT:       spyreop
// CHECK:           arith.divf
func.func @divf_f64_declined(%t: tensor<4xf64>) -> tensor<4xf64> {
  %splat = arith.constant dense<1.0> : tensor<4xf64>
  %init = tensor.empty() : tensor<4xf64>
  %0 = linalg.generic {
      indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>,
                       affine_map<(d0) -> (d0)>],
      iterator_types = ["parallel"]}
      ins(%splat, %t : tensor<4xf64>, tensor<4xf64>) outs(%init : tensor<4xf64>) {
  ^bb0(%one: f64, %in: f64, %out: f64):
    %1 = arith.divf %one, %in : f64
    linalg.yield %1 : f64
  } -> tensor<4xf64>
  return %0 : tensor<4xf64>
}

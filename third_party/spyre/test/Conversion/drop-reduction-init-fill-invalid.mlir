// RUN: spyre-triton-opt %s --drop-reduction-init-fill -split-input-file -verify-diagnostics

// A reduction initialised by a NON-ZERO linalg.fill is rejected, not dropped.
// Both alternatives would be wrong: dropping it discards an init the IR states,
// and passing it through hits ConstructThreeStagePipeline's one-compute-op
// assertion in the scheduler. Such a reduction cannot currently be lowered
// correctly at all — the scheduler's accumulator reset is hardcoded to zero
// whatever this pass does — so the diagnostic is the only honest outcome, and it
// is worth more here than a wrong answer several passes later.
//
// The successful rewrites are in drop-reduction-init-fill.mlir.

// A mulf reduction, whose neutral element is 1.0. This is the case the "gate on
// zero, not on the combiner's neutral element" rule exists for: 1.0 IS the
// correct init for a product, and it still cannot be honoured.
module {
func.func @mul_reduction_neutral_is_one(%a: tensor<2x256x64xf16>) -> tensor<2x64xf16> {
  %one = arith.constant 1.000000e+00 : f16
  %empty = tensor.empty() : tensor<2x64xf16>
  %init = linalg.fill ins(%one : f16) outs(%empty : tensor<2x64xf16>) -> tensor<2x64xf16>
  // expected-error @below {{reduction 'outs' operand #1 is initialised by a linalg.fill of a non-zero value}}
  %r = linalg.reduce ins(%a : tensor<2x256x64xf16>) outs(%init : tensor<2x64xf16>) dimensions = [1]
    (%in: f16, %acc: f16) {
      %s = arith.mulf %in, %acc : f16
      linalg.yield %s : f16
    }
  return %r : tensor<2x64xf16>
}
}

// -----

// An accumulation onto a bias: the init is not any combiner's neutral element,
// and dropping it would silently change the result.
module {
func.func @biased_accumulator(%a: tensor<2x256x64xf16>) -> tensor<2x64xf16> {
  %bias = arith.constant 2.500000e+00 : f16
  %empty = tensor.empty() : tensor<2x64xf16>
  %init = linalg.fill ins(%bias : f16) outs(%empty : tensor<2x64xf16>) -> tensor<2x64xf16>
  // expected-error @below {{reduction 'outs' operand #1 is initialised by a linalg.fill of a non-zero value}}
  %r = linalg.reduce ins(%a : tensor<2x256x64xf16>) outs(%init : tensor<2x64xf16>) dimensions = [1]
    (%in: f16, %acc: f16) {
      %s = arith.addf %in, %acc : f16
      linalg.yield %s : f16
    }
  return %r : tensor<2x64xf16>
}
}

// -----

// Integer, so the rejection is not float-only either.
module {
func.func @non_zero_integer_init(%a: tensor<2x256x64xi32>) -> tensor<2x64xi32> {
  %one = arith.constant 1 : i32
  %empty = tensor.empty() : tensor<2x64xi32>
  %init = linalg.fill ins(%one : i32) outs(%empty : tensor<2x64xi32>) -> tensor<2x64xi32>
  // expected-error @below {{reduction 'outs' operand #1 is initialised by a linalg.fill of a non-zero value}}
  %r = linalg.reduce ins(%a : tensor<2x256x64xi32>) outs(%init : tensor<2x64xi32>) dimensions = [1]
    (%in: i32, %acc: i32) {
      %s = arith.muli %in, %acc : i32
      linalg.yield %s : i32
    }
  return %r : tensor<2x64xi32>
}
}

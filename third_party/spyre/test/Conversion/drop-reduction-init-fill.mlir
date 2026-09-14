// RUN: spyre-triton-opt %s --drop-reduction-init-fill -split-input-file | FileCheck %s

// DropReductionInitFill removes the linalg.fill that LowerComputeOps gives a
// tt.reduce for its accumulator, leaving the bare tensor.empty that hand-written
// reference KTIR states directly.
//
// The rewrite is sound ONLY because MapReductionPartials overwrites the accumulator
// before it is read — a reduction payload does read its init, and tensor.empty is
// explicitly unspecified. That reset is not a hardcoded zero: getNeutralAttr derives
// the neutral PER COMBINER (addf/subf 0.0, mulf 1.0, maximumf -inf, minimumf +inf,
// addi/subi 0, muli 1) and the scheduler emits its own fill of it.
//
// So neither the combiner NOR the fill's value is judged here — both are passed
// through, and downstream reports what it cannot handle. The whole gate is
// STRUCTURAL: the op must be a shape that pass actually rewrites (one ins, one
// init, simple body) with the fill writing into a tensor.empty. Anything else is
// left alone, silently. This pass emits no diagnostics of its own, so there is no
// -verify-diagnostics companion file; every case lives here.
//
// Inputs here are written in the already-lowered form the pass actually sees, so
// they do not depend on what the upstream producer happens to emit.

// Test 1: the shape LowerComputeOps produces for tl.sum — linalg.reduce over the
// middle axis, outs initialised by fill(0.0). The fill goes, the reduce takes the
// empty, and the now-dead zero constant is left for canonicalization.
module {
// CHECK-LABEL:   func.func @sum_reduce(
// CHECK-NOT:       linalg.fill
// CHECK:           %[[EMPTY:.*]] = tensor.empty() : tensor<2x64xf16>
// CHECK:           linalg.reduce ins(%{{.*}} : tensor<2x256x64xf16>) outs(%[[EMPTY]] : tensor<2x64xf16>) dimensions = [1]
func.func @sum_reduce(%a: tensor<2x256x64xf16>) -> tensor<2x64xf16> {
  %zero = arith.constant 0.000000e+00 : f16
  %empty = tensor.empty() : tensor<2x64xf16>
  %init = linalg.fill ins(%zero : f16) outs(%empty : tensor<2x64xf16>) -> tensor<2x64xf16>
  %r = linalg.reduce ins(%a : tensor<2x256x64xf16>) outs(%init : tensor<2x64xf16>) dimensions = [1]
    (%in: f16, %acc: f16) {
      %s = arith.addf %in, %acc : f16
      linalg.yield %s : f16
    }
  return %r : tensor<2x64xf16>
}
}

// -----

// Test 2: the same init on a linalg.generic carrying a reduction iterator, which
// is the form the scheduler's own reference KTIR uses. Matching the LinalgOp
// interface rather than the op name is what covers both.
module {
// CHECK-LABEL:   func.func @sum_generic(
// CHECK-NOT:       linalg.fill
// CHECK:           %[[EMPTY:.*]] = tensor.empty() : tensor<2x64xf16>
// CHECK:           linalg.generic {{.*}} outs(%[[EMPTY]] : tensor<2x64xf16>)
func.func @sum_generic(%a: tensor<2x256x64xf16>) -> tensor<2x64xf16> {
  %zero = arith.constant 0.000000e+00 : f16
  %empty = tensor.empty() : tensor<2x64xf16>
  %init = linalg.fill ins(%zero : f16) outs(%empty : tensor<2x64xf16>) -> tensor<2x64xf16>
  %r = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
                       affine_map<(d0, d1, d2) -> (d0, d2)>],
      iterator_types = ["parallel", "reduction", "parallel"]
    } ins(%a : tensor<2x256x64xf16>) outs(%init : tensor<2x64xf16>) {
  ^bb0(%in: f16, %acc: f16):
    %s = arith.addf %in, %acc : f16
    linalg.yield %s : f16
  } -> tensor<2x64xf16>
  return %r : tensor<2x64xf16>
}
}

// -----

// Test 3: -0.0. Nothing inspects the fill's value, and there is nothing to inspect
// it for — -0.0 is numerically the reset the scheduler writes back anyway.
module {
// CHECK-LABEL:   func.func @negative_zero(
// CHECK-NOT:       linalg.fill
// CHECK:           linalg.reduce
func.func @negative_zero(%a: tensor<2x256x64xf16>) -> tensor<2x64xf16> {
  %zero = arith.constant -0.000000e+00 : f16
  %empty = tensor.empty() : tensor<2x64xf16>
  %init = linalg.fill ins(%zero : f16) outs(%empty : tensor<2x64xf16>) -> tensor<2x64xf16>
  %r = linalg.reduce ins(%a : tensor<2x256x64xf16>) outs(%init : tensor<2x64xf16>) dimensions = [1]
    (%in: f16, %acc: f16) {
      %s = arith.addf %in, %acc : f16
      linalg.yield %s : f16
    }
  return %r : tensor<2x64xf16>
}
}

// -----

// Test 4: subf, the other combiner whose derived neutral is zero.
module {
// CHECK-LABEL:   func.func @sub_reduce(
// CHECK-NOT:       linalg.fill
// CHECK:           linalg.reduce
func.func @sub_reduce(%a: tensor<2x256x64xf16>) -> tensor<2x64xf16> {
  %zero = arith.constant 0.000000e+00 : f16
  %empty = tensor.empty() : tensor<2x64xf16>
  %init = linalg.fill ins(%zero : f16) outs(%empty : tensor<2x64xf16>) -> tensor<2x64xf16>
  %r = linalg.reduce ins(%a : tensor<2x256x64xf16>) outs(%init : tensor<2x64xf16>) dimensions = [1]
    (%in: f16, %acc: f16) {
      %s = arith.subf %acc, %in : f16
      linalg.yield %s : f16
    }
  return %r : tensor<2x64xf16>
}
}

// -----

// Test 5: MATMUL — the reason the gate cannot be the fill value alone. A
// contraction is an addf-accumulate reduction whose neutral genuinely IS zero, so
// a zero-only gate would drop this init. But MapReductionPartials never rewrites a
// matmul — its assert is a single `ins` — so nothing resets that accumulator and
// the fill is load-bearing. Skipped silently:
// diagnosing a matmul is not this pass's job, and failing here would stop any
// pipeline that merely contains one.
module {
// CHECK-LABEL:   func.func @matmul_fill_survives(
// CHECK:           linalg.fill
// CHECK:           linalg.matmul
func.func @matmul_fill_survives(%a: tensor<64x128xf16>,
                                %b: tensor<128x64xf16>) -> tensor<64x64xf16> {
  %zero = arith.constant 0.000000e+00 : f16
  %empty = tensor.empty() : tensor<64x64xf16>
  %init = linalg.fill ins(%zero : f16) outs(%empty : tensor<64x64xf16>) -> tensor<64x64xf16>
  %r = linalg.matmul ins(%a, %b : tensor<64x128xf16>, tensor<128x64xf16>)
                     outs(%init : tensor<64x64xf16>) -> tensor<64x64xf16>
  return %r : tensor<64x64xf16>
}
}

// -----

// Test 6: a single-input reduction whose body is more than payload + yield —
// sum-of-squares. LinalgLowering maps one payload op to one vectorchain binary op,
// so this is not a shape the scheduler handles; the fill stays.
module {
// CHECK-LABEL:   func.func @compound_body_fill_survives(
// CHECK:           linalg.fill
// CHECK:           linalg.reduce
func.func @compound_body_fill_survives(%a: tensor<2x256x64xf16>) -> tensor<2x64xf16> {
  %zero = arith.constant 0.000000e+00 : f16
  %empty = tensor.empty() : tensor<2x64xf16>
  %init = linalg.fill ins(%zero : f16) outs(%empty : tensor<2x64xf16>) -> tensor<2x64xf16>
  %r = linalg.reduce ins(%a : tensor<2x256x64xf16>) outs(%init : tensor<2x64xf16>) dimensions = [1]
    (%in: f16, %acc: f16) {
      %sq = arith.mulf %in, %in : f16
      %s = arith.addf %sq, %acc : f16
      linalg.yield %s : f16
    }
  return %r : tensor<2x64xf16>
}
}

// -----

// Test 7: a multi-init reduction. MapReductionPartials asserts exactly one output,
// so this is out of scope however zero the fills are.
module {
// CHECK-LABEL:   func.func @multi_init_fill_survives(
// CHECK:           linalg.fill
// CHECK:           linalg.generic
func.func @multi_init_fill_survives(%a: tensor<2x256x64xf16>)
    -> (tensor<2x64xf16>, tensor<2x64xf16>) {
  %zero = arith.constant 0.000000e+00 : f16
  %e0 = tensor.empty() : tensor<2x64xf16>
  %e1 = tensor.empty() : tensor<2x64xf16>
  %i0 = linalg.fill ins(%zero : f16) outs(%e0 : tensor<2x64xf16>) -> tensor<2x64xf16>
  %i1 = linalg.fill ins(%zero : f16) outs(%e1 : tensor<2x64xf16>) -> tensor<2x64xf16>
  %r:2 = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
                       affine_map<(d0, d1, d2) -> (d0, d2)>,
                       affine_map<(d0, d1, d2) -> (d0, d2)>],
      iterator_types = ["parallel", "reduction", "parallel"]
    } ins(%a : tensor<2x256x64xf16>)
      outs(%i0, %i1 : tensor<2x64xf16>, tensor<2x64xf16>) {
  ^bb0(%in: f16, %acc0: f16, %acc1: f16):
    %s0 = arith.addf %in, %acc0 : f16
    %s1 = arith.addf %in, %acc1 : f16
    linalg.yield %s0, %s1 : f16, f16
  } -> (tensor<2x64xf16>, tensor<2x64xf16>)
  return %r#0, %r#1 : tensor<2x64xf16>, tensor<2x64xf16>
}
}

// -----

// Test 8: an ELEMENTWISE op's fill is out of scope, zero or not. tt.splat lowers
// to linalg.fill, and it is a real initialiser there — no reduction reset covers
// it. The pass must leave it alone rather than treat every fill as redundant.
module {
// CHECK-LABEL:   func.func @elementwise_fill_survives(
// CHECK:           linalg.fill
func.func @elementwise_fill_survives(%a: tensor<2x64xf16>) -> tensor<2x64xf16> {
  %zero = arith.constant 0.000000e+00 : f16
  %empty = tensor.empty() : tensor<2x64xf16>
  %splat = linalg.fill ins(%zero : f16) outs(%empty : tensor<2x64xf16>) -> tensor<2x64xf16>
  %out = tensor.empty() : tensor<2x64xf16>
  %r = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                       affine_map<(d0, d1) -> (d0, d1)>,
                       affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]
    } ins(%a, %splat : tensor<2x64xf16>, tensor<2x64xf16>)
      outs(%out : tensor<2x64xf16>) {
  ^bb0(%x: f16, %y: f16, %o: f16):
    %s = arith.addf %x, %y : f16
    linalg.yield %s : f16
  } -> tensor<2x64xf16>
  return %r : tensor<2x64xf16>
}
}

// -----

// Test 9: a fill writing over live data rather than a tensor.empty is left alone.
// Repointing outs at that value would substitute its contents for the stated
// init, which is a different rewrite from dropping a redundant one.
module {
// CHECK-LABEL:   func.func @fill_over_live_data(
// CHECK:           linalg.fill
// CHECK:           linalg.reduce
func.func @fill_over_live_data(%a: tensor<2x256x64xf16>,
                               %live: tensor<2x64xf16>) -> tensor<2x64xf16> {
  %zero = arith.constant 0.000000e+00 : f16
  %init = linalg.fill ins(%zero : f16) outs(%live : tensor<2x64xf16>) -> tensor<2x64xf16>
  %r = linalg.reduce ins(%a : tensor<2x256x64xf16>) outs(%init : tensor<2x64xf16>) dimensions = [1]
    (%in: f16, %acc: f16) {
      %s = arith.addf %in, %acc : f16
      linalg.yield %s : f16
    }
  return %r : tensor<2x64xf16>
}
}

// -----

// Test 10: a fill with another user is dropped from the reduction's outs but not
// erased, since something else still needs the filled tensor.
module {
// CHECK-LABEL:   func.func @fill_with_another_user(
// CHECK:           %[[EMPTY:.*]] = tensor.empty() : tensor<2x64xf16>
// CHECK:           %[[FILL:.*]] = linalg.fill ins(%{{.*}} : f16) outs(%[[EMPTY]] : tensor<2x64xf16>)
// CHECK:           linalg.reduce ins(%{{.*}} : tensor<2x256x64xf16>) outs(%[[EMPTY]] : tensor<2x64xf16>) dimensions = [1]
// CHECK:           return %[[REDUCED:.*]], %[[FILL]]
func.func @fill_with_another_user(%a: tensor<2x256x64xf16>)
    -> (tensor<2x64xf16>, tensor<2x64xf16>) {
  %zero = arith.constant 0.000000e+00 : f16
  %empty = tensor.empty() : tensor<2x64xf16>
  %init = linalg.fill ins(%zero : f16) outs(%empty : tensor<2x64xf16>) -> tensor<2x64xf16>
  %r = linalg.reduce ins(%a : tensor<2x256x64xf16>) outs(%init : tensor<2x64xf16>) dimensions = [1]
    (%in: f16, %acc: f16) {
      %s = arith.addf %in, %acc : f16
      linalg.yield %s : f16
    }
  return %r, %init : tensor<2x64xf16>, tensor<2x64xf16>
}
}

// -----

// Test 11: a mulf reduction with a 1.0 fill. This used to be rejected on the
// combiner, on the false premise that the scheduler's reset was a hardcoded zero.
// It is not: getNeutralAttr gives mulf a 1.0 neutral and the scheduler restores
// that, so the stated 1.0 is exactly what comes back and the fill says nothing the
// reset does not.
module {
// CHECK-LABEL:   func.func @mul_reduce_neutral_one(
// CHECK:           %[[EMPTY:.*]] = tensor.empty() : tensor<2x64xf16>
// CHECK:           linalg.reduce ins(%{{.*}} : tensor<2x256x64xf16>) outs(%[[EMPTY]] : tensor<2x64xf16>) dimensions = [1]
// CHECK:             arith.mulf
// CHECK-NOT:       linalg.fill
// CHECK:           return
func.func @mul_reduce_neutral_one(%a: tensor<2x256x64xf16>) -> tensor<2x64xf16> {
  %one = arith.constant 1.000000e+00 : f16
  %empty = tensor.empty() : tensor<2x64xf16>
  %init = linalg.fill ins(%one : f16) outs(%empty : tensor<2x64xf16>) -> tensor<2x64xf16>
  %r = linalg.reduce ins(%a : tensor<2x256x64xf16>) outs(%init : tensor<2x64xf16>) dimensions = [1]
    (%in: f16, %acc: f16) {
      %s = arith.mulf %in, %acc : f16
      linalg.yield %s : f16
    }
  return %r : tensor<2x64xf16>
}
}

// -----

// Test 12: a maximumf reduction with a -inf fill — the softmax/layernorm path.
// The scheduler derives -inf as maximumf's neutral and restores it, so dropping a
// fill that already states -inf loses nothing.
module {
// CHECK-LABEL:   func.func @max_reduce_neutral_neg_inf(
// CHECK:           %[[EMPTY:.*]] = tensor.empty() : tensor<2x64xf16>
// CHECK:           linalg.reduce ins(%{{.*}} : tensor<2x256x64xf16>) outs(%[[EMPTY]] : tensor<2x64xf16>) dimensions = [1]
// CHECK:             arith.maximumf
// CHECK-NOT:       linalg.fill
// CHECK:           return
func.func @max_reduce_neutral_neg_inf(%a: tensor<2x256x64xf16>) -> tensor<2x64xf16> {
  %neg_inf = arith.constant 0xFC00 : f16
  %empty = tensor.empty() : tensor<2x64xf16>
  %init = linalg.fill ins(%neg_inf : f16) outs(%empty : tensor<2x64xf16>) -> tensor<2x64xf16>
  %r = linalg.reduce ins(%a : tensor<2x256x64xf16>) outs(%init : tensor<2x64xf16>) dimensions = [1]
    (%in: f16, %acc: f16) {
      %s = arith.maximumf %in, %acc : f16
      linalg.yield %s : f16
    }
  return %r : tensor<2x64xf16>
}
}

// -----

// Test 13: an INTEGER add with an i32 zero fill. The neutral is per combiner AND
// typed — getNeutralAttr builds an integer attribute for addi — so there is nothing
// float-specific to guard against here; the scheduler restores an i32 0 and the
// fill is redundant.
module {
// CHECK-LABEL:   func.func @integer_add_reduce(
// CHECK:           %[[EMPTY:.*]] = tensor.empty() : tensor<2x64xi32>
// CHECK:           linalg.reduce ins(%{{.*}} : tensor<2x256x64xi32>) outs(%[[EMPTY]] : tensor<2x64xi32>) dimensions = [1]
// CHECK:             arith.addi
// CHECK-NOT:       linalg.fill
// CHECK:           return
func.func @integer_add_reduce(%a: tensor<2x256x64xi32>) -> tensor<2x64xi32> {
  %zero = arith.constant 0 : i32
  %empty = tensor.empty() : tensor<2x64xi32>
  %init = linalg.fill ins(%zero : i32) outs(%empty : tensor<2x64xi32>) -> tensor<2x64xi32>
  %r = linalg.reduce ins(%a : tensor<2x256x64xi32>) outs(%init : tensor<2x64xi32>) dimensions = [1]
    (%in: i32, %acc: i32) {
      %s = arith.addi %in, %acc : i32
      linalg.yield %s : i32
    }
  return %r : tensor<2x64xi32>
}
}

// -----

// Test 14: THE ACCEPTED CONSEQUENCE. An addf reduce whose fill states 2.5 — a real
// initial value, not addf's neutral. "Sum, plus 2.5" is what the input says, and the
// 2.5 is dropped: the pass judges no values, so the fill goes like any other, and the
// scheduler will reset the accumulator to addf's neutral 0.0 instead. The constant is
// left dangling for canonicalization, which is the only trace remaining.
//
// This is deliberate, not an oversight — see the header block of
// DropReductionInitFill.cpp under "Why nothing here judges the combiner or the fill's
// value". Nothing in the pipeline emits such a fill today (LowerComputeOps fills the
// combiner's neutral by construction), and the alternative is this pass carrying a
// copy of downstream's reset semantics, which is the thing that was previously wrong.
// The case is pinned here so that the day something DOES emit a biased init, the
// behaviour is documented rather than discovered.
module {
// CHECK-LABEL:   func.func @biased_accumulator(
// CHECK:           %[[EMPTY:.*]] = tensor.empty() : tensor<2x64xf16>
// CHECK:           linalg.reduce ins(%{{.*}} : tensor<2x256x64xf16>) outs(%[[EMPTY]] : tensor<2x64xf16>) dimensions = [1]
// CHECK:             arith.addf
// CHECK-NOT:       linalg.fill
// CHECK:           return
func.func @biased_accumulator(%a: tensor<2x256x64xf16>) -> tensor<2x64xf16> {
  %bias = arith.constant 2.500000e+00 : f16
  %empty = tensor.empty() : tensor<2x64xf16>
  %init = linalg.fill ins(%bias : f16) outs(%empty : tensor<2x64xf16>) -> tensor<2x64xf16>
  %r = linalg.reduce ins(%a : tensor<2x256x64xf16>) outs(%init : tensor<2x64xf16>) dimensions = [1]
    (%in: f16, %acc: f16) {
      %s = arith.addf %in, %acc : f16
      linalg.yield %s : f16
    }
  return %r : tensor<2x64xf16>
}
}

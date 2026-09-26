// RUN: spyre-triton-opt %s --merge-spyre-ops -split-input-file | FileCheck %s

// Rule 2: a comparison whose answer is wanted as a NUMBER rather than as a flag.
//
// `arith.cmpf` + `arith.uitofp` is the mask shape -- `(m != 0)` used
// multiplicatively -- and `spyreop.compare` is one op that does it, its answer
// coming "in the width compared rather than as a boolean".
//
// Why the pair is one choice and not two: `arith.cmpf` alone gives an `i1`, which
// no spyreop op produces and the scheduler will not take in a compute body, so the
// compare is not selectable by itself. What consumes the `i1` is what decides what
// the pair becomes.
//
// The line this file holds is the NaN semantics. `spyreop.compare` is ordered for
// every predicate, `notequal` included, so arith's ordered six map exactly and its
// unordered six do not map at all.
//
// TWO SHAPES OF INPUT, and the first is the one that matters. A rule matches ops
// in ONE body, and ConvertElementwiseToLinalg gives every tensor-level op a body
// of its own -- so this group starts out spread over TWO generics with a
// `tensor<i1>` between them, which no rule can see. The pass's own fusion is what
// brings it together, so the cases below that start from tensor-level arith are
// the ones that prove the rule fires on what the pipeline actually produces. The
// later cases hand-build the single body instead, to isolate a decline from the
// question of whether fusion happened.

//===----------------------------------------------------------------------===//
// From tensor-level arith: the shape the pipeline really produces
//===----------------------------------------------------------------------===//

// RUN: spyre-triton-opt %s --convert-elementwise-to-linalg --merge-spyre-ops -split-input-file | FileCheck %s --check-prefix=FROMTENSOR

// Two tensor ops, and therefore two generics with a `tensor<4xi1>` between them
// before this pass runs. Out comes ONE generic holding one intrinsic, and no `i1`
// of any kind -- neither as a tensor nor in a body. Nothing else in the pipeline
// is needed: this pass fuses what it needs fused.
//
// FROMTENSOR-LABEL: func.func @from_tensor_mask(
// FROMTENSOR-NOT:     tensor<4xi1>
// FROMTENSOR-NOT:     arith.cmpf
// FROMTENSOR-NOT:     arith.uitofp
// FROMTENSOR:         %[[C:.*]] = arith.constant 0.000000e+00 : f16
// FROMTENSOR:         linalg.generic
// FROMTENSOR:           spyreop.compare <notequal> %{{.*}}, %[[C]] : f16
// FROMTENSOR-NOT:     linalg.generic
func.func @from_tensor_mask(%m: tensor<4xf16>) -> tensor<4xf16> {
  %zero = arith.constant dense<0.0> : tensor<4xf16>
  %c = arith.cmpf one, %m, %zero : tensor<4xf16>
  %f = arith.uitofp %c : tensor<4xi1> to tensor<4xf16>
  return %f : tensor<4xf16>
}

// -----

// THE DECLINE, from tensor level. `une` has no counterpart, so no rule fires --
// but the fusion still does, because it is gated on the `i1` and not on the rules.
// The `tensor<4xi1>` is gone, which is always right, and the two arith ops are
// left together in one body for the tier below to deal with.
//
// This is the decline documented in Passes.td under NaN semantics, and it is worth
// seeing what it leaves: an `i1` inside a body, which the scheduler will not take
// either. Nothing here diagnoses that -- see the DECLINES section of the contract.
//
// FROMTENSOR-LABEL: func.func @from_tensor_unordered_declined(
// FROMTENSOR-NOT:     spyreop.compare
// FROMTENSOR-NOT:     tensor<4xi1>
// FROMTENSOR:         arith.cmpf une
// FROMTENSOR:         arith.uitofp
func.func @from_tensor_unordered_declined(%m: tensor<4xf16>) -> tensor<4xf16> {
  %zero = arith.constant dense<0.0> : tensor<4xf16>
  %c = arith.cmpf une, %m, %zero : tensor<4xf16>
  %f = arith.uitofp %c : tensor<4xi1> to tensor<4xf16>
  return %f : tensor<4xf16>
}

//===----------------------------------------------------------------------===//
// Single-body inputs: one decision per case, fusion taken as read
//===----------------------------------------------------------------------===//

// `m != 0` as a float mask, ordered.
//
// Two things happen and only one is the rule. The splat zero is folded out of the
// `ins` list into a SCALAR constant the body reads directly -- upstream's
// splat-constant fold, which rides along with the fusion patterns and is not gated
// by the control function -- and then the compare and the cast become one
// intrinsic. So the generic comes out reading one input, with the zero an operand
// of the intrinsic rather than a tensor.
// CHECK-LABEL:   func.func @mask_notequal_f16(
// CHECK-SAME:  %[[M:.*]]: tensor<4xf16>) -> tensor<4xf16> {
// CHECK-NOT:       arith.cmpf
// CHECK-NOT:       arith.uitofp
// CHECK:           %[[Z:.*]] = arith.constant 0.000000e+00 : f16
// CHECK:           linalg.generic {{.*}} ins(%[[M]] : tensor<4xf16>)
// CHECK:           ^bb0(%[[A:.*]]: f16, %[[OUT:.*]]: f16):
// CHECK:             %[[R:.*]] = spyreop.compare <notequal> %[[A]], %[[Z]] : f16
// CHECK:             linalg.yield %[[R]] : f16
func.func @mask_notequal_f16(%m: tensor<4xf16>) -> tensor<4xf16> {
  %zero = arith.constant dense<0.0> : tensor<4xf16>
  %init = tensor.empty() : tensor<4xf16>
  %0 = linalg.generic {
      indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>,
                       affine_map<(d0) -> (d0)>],
      iterator_types = ["parallel"]}
      ins(%m, %zero : tensor<4xf16>, tensor<4xf16>) outs(%init : tensor<4xf16>) {
  ^bb0(%a: f16, %z: f16, %out: f16):
    %c = arith.cmpf one, %a, %z : f16
    %f = arith.uitofp %c : i1 to f16
    linalg.yield %f : f16
  } -> tensor<4xf16>
  return %0 : tensor<4xf16>
}

// -----

// `a == 0`, the other half of the same mask idiom, at f32.
// CHECK-LABEL:   func.func @mask_equal_f32(
// CHECK-NOT:       arith.cmpf
// CHECK:             spyreop.compare <equal> {{.*}} : f32
func.func @mask_equal_f32(%m: tensor<4xf32>) -> tensor<4xf32> {
  %zero = arith.constant dense<0.0> : tensor<4xf32>
  %init = tensor.empty() : tensor<4xf32>
  %0 = linalg.generic {
      indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>,
                       affine_map<(d0) -> (d0)>],
      iterator_types = ["parallel"]}
      ins(%m, %zero : tensor<4xf32>, tensor<4xf32>) outs(%init : tensor<4xf32>) {
  ^bb0(%a: f32, %z: f32, %out: f32):
    %c = arith.cmpf oeq, %a, %z : f32
    %f = arith.uitofp %c : i1 to f32
    linalg.yield %f : f32
  } -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----

// All four ordered inequalities, in one body, so the predicate table is covered
// rather than sampled.
// CHECK-LABEL:   func.func @all_ordered_inequalities(
// CHECK-NOT:       arith.cmpf
// CHECK:             spyreop.compare <greaterthan>
// CHECK:             spyreop.compare <greaterequal>
// CHECK:             spyreop.compare <lesserthan>
// CHECK:             spyreop.compare <lesserequal>
func.func @all_ordered_inequalities(%x: tensor<4xf16>, %y: tensor<4xf16>) -> tensor<4xf16> {
  %init = tensor.empty() : tensor<4xf16>
  %0 = linalg.generic {
      indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>,
                       affine_map<(d0) -> (d0)>],
      iterator_types = ["parallel"]}
      ins(%x, %y : tensor<4xf16>, tensor<4xf16>) outs(%init : tensor<4xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %c0 = arith.cmpf ogt, %a, %b : f16
    %f0 = arith.uitofp %c0 : i1 to f16
    %c1 = arith.cmpf oge, %a, %b : f16
    %f1 = arith.uitofp %c1 : i1 to f16
    %c2 = arith.cmpf olt, %a, %b : f16
    %f2 = arith.uitofp %c2 : i1 to f16
    %c3 = arith.cmpf ole, %a, %b : f16
    %f3 = arith.uitofp %c3 : i1 to f16
    %s0 = arith.addf %f0, %f1 : f16
    %s1 = arith.addf %f2, %f3 : f16
    %s = arith.addf %s0, %s1 : f16
    linalg.yield %s : f16
  } -> tensor<4xf16>
  return %0 : tensor<4xf16>
}

// -----

// THE NaN LINE. `une` is true where either operand is NaN and
// `spyreop.compare <notequal>` is zero there, so the two are different
// computations and the rule declines. Both ops survive; nothing is silently
// mapped onto the ordered predicate.
// CHECK-LABEL:   func.func @unordered_declined(
// CHECK-NOT:       spyreop.compare
// CHECK:             arith.cmpf une
// CHECK:             arith.uitofp
// What it leaves behind is an `i1` INSIDE a body. The tensor form is gone, which
// the fusion guarantees regardless of any rule, but the scheduler will not take
// this either and nothing here says so -- see DECLINES in the contract.
func.func @unordered_declined(%m: tensor<4xf16>) -> tensor<4xf16> {
  %zero = arith.constant dense<0.0> : tensor<4xf16>
  %init = tensor.empty() : tensor<4xf16>
  %0 = linalg.generic {
      indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>,
                       affine_map<(d0) -> (d0)>],
      iterator_types = ["parallel"]}
      ins(%m, %zero : tensor<4xf16>, tensor<4xf16>) outs(%init : tensor<4xf16>) {
  ^bb0(%a: f16, %z: f16, %out: f16):
    %c = arith.cmpf une, %a, %z : f16
    %f = arith.uitofp %c : i1 to f16
    linalg.yield %f : f16
  } -> tensor<4xf16>
  return %0 : tensor<4xf16>
}

// -----

// `ord` asks about NaN-ness rather than about an ordering, so it has no
// counterpart of any kind and is declined for a different reason than `une` --
// not a NaN disagreement, but nothing to map onto.
// CHECK-LABEL:   func.func @ord_declined(
// CHECK-NOT:       spyreop.compare
// CHECK:             arith.cmpf ord
func.func @ord_declined(%x: tensor<4xf16>, %y: tensor<4xf16>) -> tensor<4xf16> {
  %init = tensor.empty() : tensor<4xf16>
  %0 = linalg.generic {
      indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>,
                       affine_map<(d0) -> (d0)>],
      iterator_types = ["parallel"]}
      ins(%x, %y : tensor<4xf16>, tensor<4xf16>) outs(%init : tensor<4xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %c = arith.cmpf ord, %a, %b : f16
    %f = arith.uitofp %c : i1 to f16
    linalg.yield %f : f16
  } -> tensor<4xf16>
  return %0 : tensor<4xf16>
}

// -----

// `sitofp` is NOT this rule. An `i1` read as signed is 0 or -1, so the cast gives
// -1.0 where the predicate holds -- a different computation, declined rather than
// folded in.
// CHECK-LABEL:   func.func @sitofp_declined(
// CHECK-NOT:       spyreop.compare
// CHECK:             arith.cmpf
// CHECK:             arith.sitofp
func.func @sitofp_declined(%x: tensor<4xf16>, %y: tensor<4xf16>) -> tensor<4xf16> {
  %init = tensor.empty() : tensor<4xf16>
  %0 = linalg.generic {
      indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>,
                       affine_map<(d0) -> (d0)>],
      iterator_types = ["parallel"]}
      ins(%x, %y : tensor<4xf16>, tensor<4xf16>) outs(%init : tensor<4xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %c = arith.cmpf oeq, %a, %b : f16
    %f = arith.sitofp %c : i1 to f16
    linalg.yield %f : f16
  } -> tensor<4xf16>
  return %0 : tensor<4xf16>
}

// -----

// A COMPARE AT ONE WIDTH CAST TO ANOTHER is declined: spyreop.compare has
// SameOperandsAndResultType, so it cannot both compare f32 and give f16. Left for
// the 1:1 lowering, which will report the cmpf rather than mis-select it.
// CHECK-LABEL:   func.func @width_change_declined(
// CHECK-NOT:       spyreop.compare
// CHECK:             arith.cmpf
// CHECK:             arith.uitofp
func.func @width_change_declined(%x: tensor<4xf32>, %y: tensor<4xf32>) -> tensor<4xf16> {
  %init = tensor.empty() : tensor<4xf16>
  %0 = linalg.generic {
      indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>,
                       affine_map<(d0) -> (d0)>],
      iterator_types = ["parallel"]}
      ins(%x, %y : tensor<4xf32>, tensor<4xf32>) outs(%init : tensor<4xf16>) {
  ^bb0(%a: f32, %b: f32, %out: f16):
    %c = arith.cmpf oeq, %a, %b : f32
    %f = arith.uitofp %c : i1 to f16
    linalg.yield %f : f16
  } -> tensor<4xf16>
  return %0 : tensor<4xf16>
}

// -----

// A cmpf with a SECOND reader is still merged, and survives for the other reader.
// Nothing in the rule asks about use counts: the compare goes, or does not go, by
// dead-op elimination.
// CHECK-LABEL:   func.func @cmpf_shared(
// CHECK:             %[[C:.*]] = arith.cmpf oeq
// CHECK:             %[[R:.*]] = spyreop.compare <equal>
// CHECK:             arith.select %[[C]], %[[R]]
func.func @cmpf_shared(%x: tensor<4xf16>, %y: tensor<4xf16>) -> tensor<4xf16> {
  %init = tensor.empty() : tensor<4xf16>
  %0 = linalg.generic {
      indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>,
                       affine_map<(d0) -> (d0)>],
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

// THE SCOPE. The same pair outside any generic body is not matched, for the same
// reason rule 1's is not: the body is where compute is.
// CHECK-LABEL:   func.func @outside_a_body(
// CHECK-NOT:       spyreop.compare
// CHECK:           arith.cmpf
// CHECK:           arith.uitofp
func.func @outside_a_body(%a: f16, %b: f16) -> f16 {
  %c = arith.cmpf oeq, %a, %b : f16
  %f = arith.uitofp %c : i1 to f16
  return %f : f16
}

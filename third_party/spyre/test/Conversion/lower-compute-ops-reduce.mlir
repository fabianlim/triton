// RUN: spyre-triton-opt %s -split-input-file --lower-compute-ops | FileCheck %s

// Tests for --lower-compute-ops on tt.reduce -> linalg.generic, doc = "tt.reduce".
//
// Three things happen in this lowering:
//
//   1. The combiner region is cloned from the tt.reduce body into the
//      linalg.generic body, with tt.reduce.return rewritten to linalg.yield.
//   2. An init value must be materialized, because the generic -- unlike
//      tt.reduce -- takes an explicit accumulator as its outs operand. The pass
//      derives it from the combiner's terminal op via arith::getNeutralElement:
//      0.0 for addf, -inf for maxnumf, -1 for the integer select lane. That
//      arith.constant is the substance of the pattern, so every case below pins
//      its value, not merely that a linalg.generic came out.
//   3. The reduced axis is stated in the indexing maps and the iterator list
//      rather than in a `dimensions = [...]` attribute: there is one loop per
//      *input* dim, the loop over the reduced axis is marked "reduction", and the
//      output map simply omits that axis. The lowering stays rank-reducing -- the
//      result type is unchanged from the named-op era -- so what to look at is
//      *which* position in iterator_types says "reduction", and which dim the
//      output map drops. Those two must agree, and each case pins both.
//
// A combiner whose op has no neutral element cannot be lowered; that rejection
// lives in lower-compute-ops-invalid.mlir.
//
// Note the init constant is printed in whatever form MLIR chooses for the type:
// 0.000000e+00 for f32 zero, but -inf appears as its hex bit pattern 0xFF800000.
// Matching the hex is matching the value.

// -----
// Reduce along axis 0 with an addf combiner. The 4x8 input reduces to 8 elements
// -- the axis-1 extent survives -- and the axis attribute is carried across by the
// *first* iterator being "reduction" while the output map keeps only d1.
//
// This case pairs with reduce_sum_f32 below: identical combiner and input shape,
// different axis, so between them they pin that the axis attribute is read rather
// than assumed.

// CHECK-DAG: #[[$ATTR_0:.+]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-DAG: #[[$ATTR_1:.+]] = affine_map<(d0, d1) -> (d1)>
// CHECK-LABEL:   tt.func @reduce_axis_0(
// CHECK-SAME:  %[[VAL_0:.*]]: tensor<4x8xf32>) -> tensor<8xf32> {
// CHECK:           %[[VAL_1:.*]] = arith.constant 0.000000e+00 : f32
// CHECK:           %[[VAL_2:.*]] = tensor.empty() : tensor<8xf32>
// CHECK:           %[[VAL_3:.*]] = linalg.fill ins(%[[VAL_1]] : f32) outs(%[[VAL_2]] : tensor<8xf32>) -> tensor<8xf32>
// CHECK:           %[[VAL_4:.*]] = linalg.generic {doc = "tt.reduce", indexing_maps = [#[[$ATTR_0]], #[[$ATTR_1]]], iterator_types = ["reduction", "parallel"]} ins(%[[VAL_0]] : tensor<4x8xf32>) outs(%[[VAL_3]] : tensor<8xf32>) {
// CHECK:           ^bb0(%[[VAL_5:.*]]: f32, %[[VAL_6:.*]]: f32):
// CHECK:             %[[VAL_7:.*]] = arith.addf %[[VAL_5]], %[[VAL_6]] : f32
// CHECK:             linalg.yield %[[VAL_7]] : f32
// CHECK:           } -> tensor<8xf32>
// CHECK-NOT:       tt.reduce
// CHECK:           tt.return %[[VAL_4]] : tensor<8xf32>
// CHECK:         }
tt.func @reduce_axis_0(%t: tensor<4x8xf32>) -> tensor<8xf32> {
  %0 = "tt.reduce"(%t) ({
  ^bb0(%a: f32, %b: f32):
    %add = arith.addf %a, %b : f32
    tt.reduce.return %add : f32
  }) {axis = 0 : i32} : (tensor<4x8xf32>) -> tensor<8xf32>
  tt.return %0 : tensor<8xf32>
}

// -----
// Sum-reduce along axis 1: the same addf combiner as above, so the same 0.0
// identity, but the surviving extent is now 4. The accumulator is filled with the
// identity before the reduce, which is why a tensor.empty alone would not do.
//
// Triton source pattern:
//
//   x       = tl.load(x_ptr + offsets)   # tensor<BLOCK_M x BLOCK_N x f32>
//   row_sum = tl.sum(x, axis=1)          # tensor<BLOCK_M x f32>

// CHECK-DAG: #[[$ATTR_2:.+]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-DAG: #[[$ATTR_3:.+]] = affine_map<(d0, d1) -> (d0)>
// CHECK-LABEL:   tt.func @reduce_sum_f32(
// CHECK-SAME:  %[[VAL_0:.*]]: tensor<4x8xf32>) -> tensor<4xf32> {
// CHECK:           %[[VAL_1:.*]] = arith.constant 0.000000e+00 : f32
// CHECK:           %[[VAL_2:.*]] = tensor.empty() : tensor<4xf32>
// CHECK:           %[[VAL_3:.*]] = linalg.fill ins(%[[VAL_1]] : f32) outs(%[[VAL_2]] : tensor<4xf32>) -> tensor<4xf32>
// CHECK:           %[[VAL_4:.*]] = linalg.generic {doc = "tt.reduce", indexing_maps = [#[[$ATTR_2]], #[[$ATTR_3]]], iterator_types = ["parallel", "reduction"]} ins(%[[VAL_0]] : tensor<4x8xf32>) outs(%[[VAL_3]] : tensor<4xf32>) {
// CHECK:           ^bb0(%[[VAL_5:.*]]: f32, %[[VAL_6:.*]]: f32):
// CHECK:             %[[VAL_7:.*]] = arith.addf %[[VAL_5]], %[[VAL_6]] : f32
// CHECK:             linalg.yield %[[VAL_7]] : f32
// CHECK:           } -> tensor<4xf32>
// CHECK-NOT:       tt.reduce
// CHECK:           tt.return %[[VAL_4]] : tensor<4xf32>
// CHECK:         }
tt.func @reduce_sum_f32(%t: tensor<4x8xf32>) -> tensor<4xf32> {
  %0 = "tt.reduce"(%t) ({
  ^bb0(%a: f32, %b: f32):
    %add = arith.addf %a, %b : f32
    tt.reduce.return %add : f32
  }) {axis = 1 : i32} : (tensor<4x8xf32>) -> tensor<4xf32>
  tt.return %0 : tensor<4xf32>
}

// -----
// Max-reduce with a maxnumf combiner. Everything is shaped as in reduce_sum_f32;
// the one difference is the init constant, which becomes -inf (printed as the bit
// pattern 0xFF800000) rather than 0.0. Initializing a max reduction with 0.0
// would silently clamp every negative input to zero, so this constant is the
// case's whole point.
//
// NaN handling follows arith.maxnumf: NaN propagates from the right operand only.
//
// Triton source pattern:
//
//   x       = tl.load(x_ptr + offsets)   # tensor<BLOCK_M x BLOCK_N x f32>
//   row_max = tl.max(x, axis=1)          # tensor<BLOCK_M x f32>

// CHECK-DAG: #[[$ATTR_4:.+]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-DAG: #[[$ATTR_5:.+]] = affine_map<(d0, d1) -> (d0)>
// CHECK-LABEL:   tt.func @reduce_max_f32(
// CHECK-SAME:  %[[VAL_0:.*]]: tensor<4x8xf32>) -> tensor<4xf32> {
// CHECK:           %[[VAL_1:.*]] = arith.constant 0xFF800000 : f32
// CHECK:           %[[VAL_2:.*]] = tensor.empty() : tensor<4xf32>
// CHECK:           %[[VAL_3:.*]] = linalg.fill ins(%[[VAL_1]] : f32) outs(%[[VAL_2]] : tensor<4xf32>) -> tensor<4xf32>
// CHECK:           %[[VAL_4:.*]] = linalg.generic {doc = "tt.reduce", indexing_maps = [#[[$ATTR_4]], #[[$ATTR_5]]], iterator_types = ["parallel", "reduction"]} ins(%[[VAL_0]] : tensor<4x8xf32>) outs(%[[VAL_3]] : tensor<4xf32>) {
// CHECK:           ^bb0(%[[VAL_5:.*]]: f32, %[[VAL_6:.*]]: f32):
// CHECK:             %[[VAL_7:.*]] = arith.maxnumf %[[VAL_5]], %[[VAL_6]] : f32
// CHECK:             linalg.yield %[[VAL_7]] : f32
// CHECK:           } -> tensor<4xf32>
// CHECK-NOT:       tt.reduce
// CHECK:           tt.return %[[VAL_4]] : tensor<4xf32>
// CHECK:         }
tt.func @reduce_max_f32(%t: tensor<4x8xf32>) -> tensor<4xf32> {
  %0 = "tt.reduce"(%t) ({
  ^bb0(%a: f32, %b: f32):
    %max = arith.maxnumf %a, %b : f32
    tt.reduce.return %max : f32
  }) {axis = 1 : i32} : (tensor<4x8xf32>) -> tensor<4xf32>
  tt.return %0 : tensor<4xf32>
}

// -----
// A multi-op combiner body: math.absf feeding arith.addf. Two points.
//
// First, the whole body is cloned, not just its terminal op -- both ops appear in
// the linalg.reduce region, in order, and the absf still consumes the *second*
// block argument as it did in the source.
//
// Second, the identity comes from the op that produces the yielded value (addf,
// giving 0.0), not from the first op in the body. Had the pass looked at math.absf
// it would have found no neutral element and rejected the reduce outright.

// CHECK-DAG: #[[$ATTR_6:.+]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-DAG: #[[$ATTR_7:.+]] = affine_map<(d0, d1) -> (d0)>
// CHECK-LABEL:   tt.func @reduce_math_combiner(
// CHECK-SAME:  %[[VAL_0:.*]]: tensor<4x8xf32>) -> tensor<4xf32> {
// CHECK:           %[[VAL_1:.*]] = arith.constant 0.000000e+00 : f32
// CHECK:           %[[VAL_2:.*]] = tensor.empty() : tensor<4xf32>
// CHECK:           %[[VAL_3:.*]] = linalg.fill ins(%[[VAL_1]] : f32) outs(%[[VAL_2]] : tensor<4xf32>) -> tensor<4xf32>
// CHECK:           %[[VAL_4:.*]] = linalg.generic {doc = "tt.reduce", indexing_maps = [#[[$ATTR_6]], #[[$ATTR_7]]], iterator_types = ["parallel", "reduction"]} ins(%[[VAL_0]] : tensor<4x8xf32>) outs(%[[VAL_3]] : tensor<4xf32>) {
// CHECK:           ^bb0(%[[VAL_5:.*]]: f32, %[[VAL_6:.*]]: f32):
// CHECK:             %[[VAL_7:.*]] = math.absf %[[VAL_6]] : f32
// CHECK:             %[[VAL_8:.*]] = arith.addf %[[VAL_5]], %[[VAL_7]] : f32
// CHECK:             linalg.yield %[[VAL_8]] : f32
// CHECK:           } -> tensor<4xf32>
// CHECK-NOT:       tt.reduce
// CHECK:           tt.return %[[VAL_4]] : tensor<4xf32>
// CHECK:         }
tt.func @reduce_math_combiner(%t: tensor<4x8xf32>) -> tensor<4xf32> {
  %0 = "tt.reduce"(%t) ({
  ^bb0(%a: f32, %b: f32):
    %abs = math.absf %b : f32
    %add = arith.addf %a, %abs : f32
    tt.reduce.return %add : f32
  }) {axis = 1 : i32} : (tensor<4x8xf32>) -> tensor<4xf32>
  tt.return %0 : tensor<4xf32>
}

// -----
// Reducing a 1-D tensor along its only axis yields a *scalar*, and that is the
// rank-0 edge case: linalg.generic cannot produce an f32, only a tensor<f32>, so
// the accumulator is a rank-0 tensor -- its map is the empty one, (d0) -> () --
// and a trailing tensor.extract with an empty index list unwraps it to the scalar
// the tt.func returns.
//
// The extract is the part a lowering that only handled rank >= 1 outputs would
// omit, leaving a type mismatch at the return -- so it is matched explicitly, and
// the empty `[]` index list is matched with it.

// CHECK-DAG: #[[$ATTR_8:.+]] = affine_map<(d0) -> (d0)>
// CHECK-DAG: #[[$ATTR_9:.+]] = affine_map<(d0) -> ()>
// CHECK-LABEL:   tt.func @reduce_to_scalar(
// CHECK-SAME:  %[[VAL_0:.*]]: tensor<8xf32>) -> f32 {
// CHECK:           %[[VAL_1:.*]] = arith.constant 0.000000e+00 : f32
// CHECK:           %[[VAL_2:.*]] = tensor.empty() : tensor<f32>
// CHECK:           %[[VAL_3:.*]] = linalg.fill ins(%[[VAL_1]] : f32) outs(%[[VAL_2]] : tensor<f32>) -> tensor<f32>
// CHECK:           %[[VAL_4:.*]] = linalg.generic {doc = "tt.reduce", indexing_maps = [#[[$ATTR_8]], #[[$ATTR_9]]], iterator_types = ["reduction"]} ins(%[[VAL_0]] : tensor<8xf32>) outs(%[[VAL_3]] : tensor<f32>) {
// CHECK:           ^bb0(%[[VAL_5:.*]]: f32, %[[VAL_6:.*]]: f32):
// CHECK:             %[[VAL_7:.*]] = arith.addf %[[VAL_5]], %[[VAL_6]] : f32
// CHECK:             linalg.yield %[[VAL_7]] : f32
// CHECK:           } -> tensor<f32>
// CHECK:           %[[VAL_8:.*]] = tensor.extract %[[VAL_4]][] : tensor<f32>
// CHECK-NOT:       tt.reduce
// CHECK:           tt.return %[[VAL_8]] : f32
// CHECK:         }
tt.func @reduce_to_scalar(%t: tensor<8xf32>) -> f32 {
  %0 = "tt.reduce"(%t) ({
  ^bb0(%a: f32, %b: f32):
    %add = arith.addf %a, %b : f32
    tt.reduce.return %add : f32
  }) {axis = 0 : i32} : (tensor<8xf32>) -> f32
  tt.return %0 : f32
}

// -----
// A multi-operand tt.reduce -- the argmax shape -- lowers to a *single*
// linalg.generic with two ins and two outs, not to one reduce per lane. Splitting
// them would be wrong as well as slower: the index lane's select reads the value
// lane's comparison, so the two must be reduced in lockstep in one body.
//
// Each lane gets its own init: -inf for the f32 value lane (from maxnumf) and -1
// for the i32 index lane (from select, an invalid-index sentinel). Two different
// neutral elements derived from two different ops in one body is the claim, and
// the guard below rules out a second linalg.generic. The four indexing maps are
// two input maps then two output maps -- the same two distinct maps, each reused
// once per lane -- which is why only two aliases appear.
//
// The combiner body's four block arguments interleave as (value_in, index_in,
// value_init, index_init) -- inputs then inits, not per-lane pairs -- which the
// argument list below pins.
//
// Triton source pattern:
//
//   values, indices = tl.argmax(x, axis=1, return_indices=True)

// CHECK-DAG: #[[$ATTR_10:.+]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-DAG: #[[$ATTR_11:.+]] = affine_map<(d0, d1) -> (d0)>
// CHECK-LABEL:   tt.func @reduce_multi_operand(
// CHECK-SAME:  %[[VAL_0:.*]]: tensor<4x8xf32>, %[[VAL_1:.*]]: tensor<4x8xi32>) -> (tensor<4xf32>, tensor<4xi32>) {
// CHECK:           %[[VAL_2:.*]] = arith.constant 0xFF800000 : f32
// CHECK:           %[[VAL_3:.*]] = tensor.empty() : tensor<4xf32>
// CHECK:           %[[VAL_4:.*]] = linalg.fill ins(%[[VAL_2]] : f32) outs(%[[VAL_3]] : tensor<4xf32>) -> tensor<4xf32>
// CHECK:           %[[VAL_5:.*]] = arith.constant -1 : i32
// CHECK:           %[[VAL_6:.*]] = tensor.empty() : tensor<4xi32>
// CHECK:           %[[VAL_7:.*]] = linalg.fill ins(%[[VAL_5]] : i32) outs(%[[VAL_6]] : tensor<4xi32>) -> tensor<4xi32>
// CHECK:           %[[VAL_8:.*]]:2 = linalg.generic {doc = "tt.reduce", indexing_maps = [#[[$ATTR_10]], #[[$ATTR_10]], #[[$ATTR_11]], #[[$ATTR_11]]], iterator_types = ["parallel", "reduction"]} ins(%[[VAL_0]], %[[VAL_1]] : tensor<4x8xf32>, tensor<4x8xi32>) outs(%[[VAL_4]], %[[VAL_7]] : tensor<4xf32>, tensor<4xi32>) {
// CHECK:           ^bb0(%[[VAL_9:.*]]: f32, %[[VAL_10:.*]]: i32, %[[VAL_11:.*]]: f32, %[[VAL_12:.*]]: i32):
// CHECK:             %[[VAL_13:.*]] = arith.cmpf ogt, %[[VAL_9]], %[[VAL_11]] : f32
// CHECK:             %[[VAL_14:.*]] = arith.maxnumf %[[VAL_9]], %[[VAL_11]] : f32
// CHECK:             %[[VAL_15:.*]] = arith.select %[[VAL_13]], %[[VAL_10]], %[[VAL_12]] : i32
// CHECK:             linalg.yield %[[VAL_14]], %[[VAL_15]] : f32, i32
// CHECK:           } -> (tensor<4xf32>, tensor<4xi32>)
// CHECK-NOT:       linalg.generic
// CHECK-NOT:       tt.reduce
// CHECK:           tt.return %[[VAL_8]]#0, %[[VAL_8]]#1 : tensor<4xf32>, tensor<4xi32>
// CHECK:         }
tt.func @reduce_multi_operand(%vals: tensor<4x8xf32>, %idxs: tensor<4x8xi32>)
    -> (tensor<4xf32>, tensor<4xi32>) {
  %0:2 = "tt.reduce"(%vals, %idxs) ({
  ^bb0(%a: f32, %ai: i32, %b: f32, %bi: i32):
    %cmp = arith.cmpf ogt, %a, %b : f32
    %max = arith.maxnumf %a, %b : f32
    %idx = arith.select %cmp, %ai, %bi : i32
    tt.reduce.return %max, %idx : f32, i32
  }) {axis = 1 : i32} : (tensor<4x8xf32>, tensor<4x8xi32>) -> (tensor<4xf32>, tensor<4xi32>)
  tt.return %0#0, %0#1 : tensor<4xf32>, tensor<4xi32>
}

// RUN: spyre-triton-opt %s -split-input-file --lower-compute-ops | FileCheck %s

// Tests for --lower-compute-ops on tt.dot.
//
// Every rank lowers to one linalg.generic carrying doc = "tt.dot". The named
// linalg.matmul / linalg.batch_matmul forms are gone: the contraction is now
// stated as data the next pass can read -- the three indexing maps plus one
// "reduction" iterator -- rather than implied by an op name. So the rank dispatch
// shows up in the *shape* of that data, not in a choice of op: rank 2 gives three
// loops (m, n, k) and rank 3 gives four (b, m, n, k), with the trailing k the
// reduction in both. Each case pins the maps and the iterator list, since those
// are what carry the contraction.
//
// The body is arith.mulf then arith.addf (muli/addi for integers), and the addf
// reads the running value from the `outs` block argument. That is where the
// accumulator lives: tt.dot's third operand is passed straight through as outs --
// no tensor.empty, no zero fill, because the accumulator already holds the values
// to add into. A lowering that materialized a fresh destination would drop it and
// silently compute a * b instead of a * b + c, so each case pins that outs names
// the third argument.
//
// The f16-input case is not repeated here: it is already pinned by
// Conversion/matmul.mlir, which runs the same pass on the same IR.
//
// Rank 4 and above is rejected, by the upstream Triton verifier rather than by
// this pass -- see lower-compute-ops-invalid.mlir.

// -----
// Rank-2 f32. All three operands and the result share the element type, and the
// contraction is 16x32 by 32x8 into 16x8.
//
// Triton source pattern:
//
//   acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
//   for k in range(k_tiles):
//       a   = a_desc.load([m * BM, k * BK])   # tensor<BM x BK x f32>
//       b   = b_desc.load([k * BK, n * BN])   # tensor<BK x BN x f32>
//       acc = tl.dot(a, b, acc)                # tensor<BM x BN x f32>

// CHECK-DAG: #[[$ATTR_0:.+]] = affine_map<(d0, d1, d2) -> (d0, d2)>
// CHECK-DAG: #[[$ATTR_1:.+]] = affine_map<(d0, d1, d2) -> (d2, d1)>
// CHECK-DAG: #[[$ATTR_2:.+]] = affine_map<(d0, d1, d2) -> (d0, d1)>
// CHECK-LABEL:   tt.func @dot_f32(
// CHECK-SAME:  %[[VAL_0:.*]]: tensor<16x32xf32>, %[[VAL_1:.*]]: tensor<32x8xf32>, %[[VAL_2:.*]]: tensor<16x8xf32>) -> tensor<16x8xf32> {
// CHECK:           %[[VAL_3:.*]] = linalg.generic {doc = "tt.dot", indexing_maps = [#[[$ATTR_0]], #[[$ATTR_1]], #[[$ATTR_2]]], iterator_types = ["parallel", "parallel", "reduction"]} ins(%[[VAL_0]], %[[VAL_1]] : tensor<16x32xf32>, tensor<32x8xf32>) outs(%[[VAL_2]] : tensor<16x8xf32>) {
// CHECK:           ^bb0(%[[VAL_4:.*]]: f32, %[[VAL_5:.*]]: f32, %[[VAL_6:.*]]: f32):
// CHECK:             %[[VAL_7:.*]] = arith.mulf %[[VAL_4]], %[[VAL_5]] : f32
// CHECK:             %[[VAL_8:.*]] = arith.addf %[[VAL_6]], %[[VAL_7]] : f32
// CHECK:             linalg.yield %[[VAL_8]] : f32
// CHECK:           } -> tensor<16x8xf32>
// CHECK-NOT:       tt.dot
// CHECK-NOT:       tensor.empty
// CHECK:           tt.return %[[VAL_3]] : tensor<16x8xf32>
// CHECK:         }
tt.func @dot_f32(%a: tensor<16x32xf32>, %b: tensor<32x8xf32>,
                 %c: tensor<16x8xf32>) -> tensor<16x8xf32> {
  %0 = tt.dot %a, %b, %c : tensor<16x32xf32> * tensor<32x8xf32> -> tensor<16x8xf32>
  tt.return %0 : tensor<16x8xf32>
}

// -----
// Rank-2 at realistic tile sizes, 128x64 by 64x128. Tile extents do not enter the
// lowering -- they are copied out of the operand types -- so this case exists to
// pin that nothing in the pattern is sensitive to magnitude, in particular that no
// size threshold quietly selects a different loop structure: the maps and the
// three-loop iterator list are identical to dot_f32 above.

// CHECK-DAG: #[[$ATTR_3:.+]] = affine_map<(d0, d1, d2) -> (d0, d2)>
// CHECK-DAG: #[[$ATTR_4:.+]] = affine_map<(d0, d1, d2) -> (d2, d1)>
// CHECK-DAG: #[[$ATTR_5:.+]] = affine_map<(d0, d1, d2) -> (d0, d1)>
// CHECK-LABEL:   tt.func @dot_large(
// CHECK-SAME:  %[[VAL_0:.*]]: tensor<128x64xf32>, %[[VAL_1:.*]]: tensor<64x128xf32>, %[[VAL_2:.*]]: tensor<128x128xf32>) -> tensor<128x128xf32> {
// CHECK:           %[[VAL_3:.*]] = linalg.generic {doc = "tt.dot", indexing_maps = [#[[$ATTR_3]], #[[$ATTR_4]], #[[$ATTR_5]]], iterator_types = ["parallel", "parallel", "reduction"]} ins(%[[VAL_0]], %[[VAL_1]] : tensor<128x64xf32>, tensor<64x128xf32>) outs(%[[VAL_2]] : tensor<128x128xf32>) {
// CHECK:           ^bb0(%[[VAL_4:.*]]: f32, %[[VAL_5:.*]]: f32, %[[VAL_6:.*]]: f32):
// CHECK:             %[[VAL_7:.*]] = arith.mulf %[[VAL_4]], %[[VAL_5]] : f32
// CHECK:             %[[VAL_8:.*]] = arith.addf %[[VAL_6]], %[[VAL_7]] : f32
// CHECK:             linalg.yield %[[VAL_8]] : f32
// CHECK:           } -> tensor<128x128xf32>
// CHECK-NOT:       tt.dot
// CHECK-NOT:       linalg.batch_matmul
// CHECK-NOT:       linalg.matmul
// CHECK:           tt.return %[[VAL_3]] : tensor<128x128xf32>
// CHECK:         }
tt.func @dot_large(%a: tensor<128x64xf32>, %b: tensor<64x128xf32>,
                   %c: tensor<128x128xf32>) -> tensor<128x128xf32> {
  %0 = tt.dot %a, %b, %c : tensor<128x64xf32> * tensor<64x128xf32> -> tensor<128x128xf32>
  tt.return %0 : tensor<128x128xf32>
}

// -----
// Rank 3 adds a leading batch loop: four loops (b, m, n, k) instead of three, with
// dim 4 read as the batch and carried by every map. The maps are the real content
// here -- the rank dispatch is the whole pattern, and a fallthrough to the rank-2
// branch would emit three-loop maps that cannot index rank-3 operands. Note the
// batch dim is `parallel`, not a second reduction: k alone is summed over.

// CHECK-DAG: #[[$ATTR_6:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
// CHECK-DAG: #[[$ATTR_7:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>
// CHECK-DAG: #[[$ATTR_8:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
// CHECK-LABEL:   tt.func @dot_batch_matmul(
// CHECK-SAME:  %[[VAL_0:.*]]: tensor<4x16x32xf32>, %[[VAL_1:.*]]: tensor<4x32x8xf32>, %[[VAL_2:.*]]: tensor<4x16x8xf32>) -> tensor<4x16x8xf32> {
// CHECK:           %[[VAL_3:.*]] = linalg.generic {doc = "tt.dot", indexing_maps = [#[[$ATTR_6]], #[[$ATTR_7]], #[[$ATTR_8]]], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%[[VAL_0]], %[[VAL_1]] : tensor<4x16x32xf32>, tensor<4x32x8xf32>) outs(%[[VAL_2]] : tensor<4x16x8xf32>) {
// CHECK:           ^bb0(%[[VAL_4:.*]]: f32, %[[VAL_5:.*]]: f32, %[[VAL_6:.*]]: f32):
// CHECK:             %[[VAL_7:.*]] = arith.mulf %[[VAL_4]], %[[VAL_5]] : f32
// CHECK:             %[[VAL_8:.*]] = arith.addf %[[VAL_6]], %[[VAL_7]] : f32
// CHECK:             linalg.yield %[[VAL_8]] : f32
// CHECK:           } -> tensor<4x16x8xf32>
// CHECK-NOT:       tt.dot
// CHECK-NOT:       linalg.batch_matmul
// CHECK-NOT:       linalg.matmul
// CHECK:           tt.return %[[VAL_3]] : tensor<4x16x8xf32>
// CHECK:         }
tt.func @dot_batch_matmul(%a: tensor<4x16x32xf32>, %b: tensor<4x32x8xf32>,
                          %c: tensor<4x16x8xf32>) -> tensor<4x16x8xf32> {
  %0 = tt.dot %a, %b, %c : tensor<4x16x32xf32> * tensor<4x32x8xf32> -> tensor<4x16x8xf32>
  tt.return %0 : tensor<4x16x8xf32>
}

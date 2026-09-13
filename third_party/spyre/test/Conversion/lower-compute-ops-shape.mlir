// RUN: spyre-triton-opt %s -split-input-file --lower-compute-ops | FileCheck %s

// Tests for --lower-compute-ops on the four pure shape ops: tt.reshape,
// tt.expand_dims, tt.broadcast and tt.trans.
//
// They share this file because they share a question -- how the target shape is
// spelled in the output -- and after the move to linalg.generic three of the four
// answer it the same way: the coordinate change lives in the *operand indexing
// map*, and the body is a bare linalg.yield of the block argument. Nothing is
// computed; the only content is where each output coordinate reads from.
//
//   tt.reshape      -> tensor.reshape (the one op still named). Target shape
//                      passed as a *value*: a tensor.from_elements of index
//                      constants, one per output dim. So the constants are the
//                      shape, and pinning them is pinning the reshape.
//   tt.expand_dims  -> tensor.empty + linalg.generic, doc = "tt.expand_dims".
//                      One loop per *output* dim; the operand map drops the
//                      inserted axis, e.g. (d0, d1) -> (d1) for axis 0. The
//                      dropped dim is the axis attribute, read off the map.
//   tt.broadcast    -> tensor.empty + linalg.generic, doc = "tt.broadcast".
//                      The operand map carries a literal constant 0 in every
//                      expanded dim, e.g. (d0, d1) -> (0, d1): the same source
//                      element is re-read for every value of the broadcast loop.
//                      Those zeros are the claim.
//   tt.trans        -> tensor.empty + linalg.generic, doc = "tt.trans". The order
//                      attribute becomes a permuting operand map, e.g.
//                      (d0, d1) -> (d1, d0) for a 2-D swap.
//
// So the three generics are told apart by their maps and by the doc attribute,
// not by an op name -- which is why every case below matches both. Grouping them
// keeps the contrast visible: all four could plausibly have been lowered to a
// single tensor.reshape, and the cases below pin that they are not.

// -----
// Rank-increasing reshape, 1-D to 2-D. The two index constants 16 and 32 are the
// output shape; they reach tensor.reshape through a tensor.from_elements shape
// operand rather than through the result type alone.
//
// Triton source pattern:
//
//   x = tl.reshape(x, [BLOCK_M, BLOCK_N])   # reinterpret a flat tile as 2-D

// CHECK-LABEL:   tt.func @reshape_1d_to_2d(
// CHECK-SAME:  %[[VAL_0:.*]]: tensor<512xf32>) -> tensor<16x32xf32> {
// CHECK:           %[[VAL_1:.*]] = arith.constant 16 : index
// CHECK:           %[[VAL_2:.*]] = arith.constant 32 : index
// CHECK:           %[[VAL_3:.*]] = tensor.from_elements %[[VAL_1]], %[[VAL_2]] : tensor<2xindex>
// CHECK:           %[[VAL_4:.*]] = tensor.reshape %[[VAL_0]](%[[VAL_3]]) : (tensor<512xf32>, tensor<2xindex>) -> tensor<16x32xf32>
// CHECK-NOT:       tt.reshape
// CHECK:           tt.return %[[VAL_4]] : tensor<16x32xf32>
// CHECK:         }
tt.func @reshape_1d_to_2d(%t: tensor<512xf32>) -> tensor<16x32xf32> {
  %0 = tt.reshape %t : tensor<512xf32> -> tensor<16x32xf32>
  tt.return %0 : tensor<16x32xf32>
}

// -----
// Rank-decreasing reshape, 2-D to 1-D. One output dim means a one-element
// from_elements and a tensor<1xindex> shape operand -- the rank of that shape
// tensor tracks the output rank, which is what the type on the from_elements
// pins.

// CHECK-LABEL:   tt.func @reshape_2d_to_1d(
// CHECK-SAME:  %[[VAL_0:.*]]: tensor<8x16xf16>) -> tensor<128xf16> {
// CHECK:           %[[VAL_1:.*]] = arith.constant 128 : index
// CHECK:           %[[VAL_2:.*]] = tensor.from_elements %[[VAL_1]] : tensor<1xindex>
// CHECK:           %[[VAL_3:.*]] = tensor.reshape %[[VAL_0]](%[[VAL_2]]) : (tensor<8x16xf16>, tensor<1xindex>) -> tensor<128xf16>
// CHECK-NOT:       tt.reshape
// CHECK:           tt.return %[[VAL_3]] : tensor<128xf16>
// CHECK:         }
tt.func @reshape_2d_to_1d(%t: tensor<8x16xf16>) -> tensor<128xf16> {
  %0 = tt.reshape %t : tensor<8x16xf16> -> tensor<128xf16>
  tt.return %0 : tensor<128xf16>
}

// -----
// Rank-increasing reshape, 2-D to 3-D. Three constants, and the first two are
// both 2 -- distinct SSA values, not one CSE'd constant reused, which the two
// separate capture names below pin.

// CHECK-LABEL:   tt.func @reshape_2d_to_3d(
// CHECK-SAME:  %[[VAL_0:.*]]: tensor<4x16xi32>) -> tensor<2x2x16xi32> {
// CHECK:           %[[VAL_1:.*]] = arith.constant 2 : index
// CHECK:           %[[VAL_2:.*]] = arith.constant 2 : index
// CHECK:           %[[VAL_3:.*]] = arith.constant 16 : index
// CHECK:           %[[VAL_4:.*]] = tensor.from_elements %[[VAL_1]], %[[VAL_2]], %[[VAL_3]] : tensor<3xindex>
// CHECK:           %[[VAL_5:.*]] = tensor.reshape %[[VAL_0]](%[[VAL_4]]) : (tensor<4x16xi32>, tensor<3xindex>) -> tensor<2x2x16xi32>
// CHECK-NOT:       tt.reshape
// CHECK:           tt.return %[[VAL_5]] : tensor<2x2x16xi32>
// CHECK:         }
tt.func @reshape_2d_to_3d(%t: tensor<4x16xi32>) -> tensor<2x2x16xi32> {
  %0 = tt.reshape %t : tensor<4x16xi32> -> tensor<2x2x16xi32>
  tt.return %0 : tensor<2x2x16xi32>
}

// -----
// The allow_reorder attribute is accepted and has no effect on the lowering:
// the output is identical to reshape_1d_to_2d above. allow_reorder is a licence
// to permute elements, not an instruction to, so a lowering that honoured it by
// inserting a transpose would be wrong -- hence the guard against linalg.transpose
// between the reshape and the return.

// CHECK-LABEL:   tt.func @reshape_allow_reorder(
// CHECK-SAME:  %[[VAL_0:.*]]: tensor<512xf32>) -> tensor<16x32xf32> {
// CHECK:           %[[VAL_1:.*]] = arith.constant 16 : index
// CHECK:           %[[VAL_2:.*]] = arith.constant 32 : index
// CHECK:           %[[VAL_3:.*]] = tensor.from_elements %[[VAL_1]], %[[VAL_2]] : tensor<2xindex>
// CHECK:           %[[VAL_4:.*]] = tensor.reshape %[[VAL_0]](%[[VAL_3]]) : (tensor<512xf32>, tensor<2xindex>) -> tensor<16x32xf32>
// CHECK-NOT:       tt.reshape
// CHECK-NOT:       linalg.generic
// CHECK:           tt.return %[[VAL_4]] : tensor<16x32xf32>
// CHECK:         }
tt.func @reshape_allow_reorder(%t: tensor<512xf32>) -> tensor<16x32xf32> {
  %0 = tt.reshape %t allow_reorder : tensor<512xf32> -> tensor<16x32xf32>
  tt.return %0 : tensor<16x32xf32>
}

// -----
// Rank 4 to rank 2, collapsing three leading dims into one.
//
// tt.dot accepts only rank-2 (linalg.matmul) and rank-3 (linalg.batch_matmul)
// operands -- see lower-compute-ops-invalid.mlir for the rank-4 rejection -- so
// the rank-4 output of an N-D tt.descriptor_gather must be reshaped before it
// can feed a matmul. The three leading dims [NUM_BLOCKS, NUM_GROUPS, BLOCK_SIZE]
// = [8, 1, 16] collapse into OUT_LEN = 128 while the trailing contiguous dim
// INNER_DIM = 64 is preserved.
//
// The cases above stop at rank 3, so a regression in the rank-decreasing path at
// rank 4 would otherwise pass unnoticed. What pins it here is that exactly two
// constants come out (128 and 64) for a four-dim input -- a lowering that emitted
// one constant per *input* dim would produce four.
//
// Triton source pattern:
//
//   gathered_4d = src_desc.gather(indices, group_idx)   # [8, 1, 16, 64]
//   gathered_2d = tl.reshape(gathered_4d, [OUT_LEN, INNER_DIM])
//   out         = tl.dot(lhs, tl.trans(gathered_2d))    # 2-D dot

// CHECK-LABEL:   tt.func @reshape_4d_to_2d_collapse_three_leading_dims(
// CHECK-SAME:  %[[VAL_0:.*]]: tensor<8x1x16x64xf16>) -> tensor<128x64xf16> {
// CHECK:           %[[VAL_1:.*]] = arith.constant 128 : index
// CHECK:           %[[VAL_2:.*]] = arith.constant 64 : index
// CHECK:           %[[VAL_3:.*]] = tensor.from_elements %[[VAL_1]], %[[VAL_2]] : tensor<2xindex>
// CHECK:           %[[VAL_4:.*]] = tensor.reshape %[[VAL_0]](%[[VAL_3]]) : (tensor<8x1x16x64xf16>, tensor<2xindex>) -> tensor<128x64xf16>
// CHECK-NOT:       tt.reshape
// CHECK-NOT:       arith.constant
// CHECK:           tt.return %[[VAL_4]] : tensor<128x64xf16>
// CHECK:         }
tt.func @reshape_4d_to_2d_collapse_three_leading_dims(%t: tensor<8x1x16x64xf16>) -> tensor<128x64xf16> {
  %0 = tt.reshape %t : tensor<8x1x16x64xf16> -> tensor<128x64xf16>
  tt.return %0 : tensor<128x64xf16>
}

// -----
// tt.expand_dims axis=0, inserting a size-1 dim at the front.
//
// Two output dims means two loops, and the operand map (d0, d1) -> (d1) drops d0:
// the inserted axis is exactly the loop the input is not indexed by. So the axis
// attribute survives as the *position of the dropped dim* in the operand map, and
// that is what separates this case from the axis=1 case below -- the result map
// and the iterator list are identical in both.
//
// Triton source pattern:
//
//   x = tl.load(x_ptr + offsets)     # tensor<BLOCK x f32>
//   x = tl.expand_dims(x, axis=0)    # tensor<1 x BLOCK x f32>

// CHECK-DAG: #[[$ATTR_0:.+]] = affine_map<(d0, d1) -> (d1)>
// CHECK-DAG: #[[$ATTR_1:.+]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-LABEL:   tt.func @expand_dims_axis_0(
// CHECK-SAME:  %[[VAL_0:.*]]: tensor<8xf32>) -> tensor<1x8xf32> {
// CHECK:           %[[VAL_1:.*]] = tensor.empty() : tensor<1x8xf32>
// CHECK:           %[[VAL_2:.*]] = linalg.generic {doc = "tt.expand_dims", indexing_maps = [#[[$ATTR_0]], #[[$ATTR_1]]], iterator_types = ["parallel", "parallel"]} ins(%[[VAL_0]] : tensor<8xf32>) outs(%[[VAL_1]] : tensor<1x8xf32>) {
// CHECK:           ^bb0(%[[VAL_3:.*]]: f32, %[[VAL_4:.*]]: f32):
// CHECK:             linalg.yield %[[VAL_3]] : f32
// CHECK:           } -> tensor<1x8xf32>
// CHECK-NOT:       tt.expand_dims
// CHECK:           tt.return %[[VAL_2]] : tensor<1x8xf32>
// CHECK:         }
tt.func @expand_dims_axis_0(%t: tensor<8xf32>) -> tensor<1x8xf32> {
  %0 = tt.expand_dims %t {axis = 0 : i32} : tensor<8xf32> -> tensor<1x8xf32>
  tt.return %0 : tensor<1x8xf32>
}

// -----
// tt.expand_dims axis=1, inserting the size-1 dim at the end. Everything matches
// axis=0 except the operand map, which now drops d1 rather than d0 -- so this pair
// is the test that the axis attribute is read at all.

// CHECK-DAG: #[[$ATTR_2:.+]] = affine_map<(d0, d1) -> (d0)>
// CHECK-DAG: #[[$ATTR_3:.+]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-LABEL:   tt.func @expand_dims_axis_1(
// CHECK-SAME:  %[[VAL_0:.*]]: tensor<8xf32>) -> tensor<8x1xf32> {
// CHECK:           %[[VAL_1:.*]] = tensor.empty() : tensor<8x1xf32>
// CHECK:           %[[VAL_2:.*]] = linalg.generic {doc = "tt.expand_dims", indexing_maps = [#[[$ATTR_2]], #[[$ATTR_3]]], iterator_types = ["parallel", "parallel"]} ins(%[[VAL_0]] : tensor<8xf32>) outs(%[[VAL_1]] : tensor<8x1xf32>) {
// CHECK:           ^bb0(%[[VAL_3:.*]]: f32, %[[VAL_4:.*]]: f32):
// CHECK:             linalg.yield %[[VAL_3]] : f32
// CHECK:           } -> tensor<8x1xf32>
// CHECK-NOT:       tt.expand_dims
// CHECK:           tt.return %[[VAL_2]] : tensor<8x1xf32>
// CHECK:         }
tt.func @expand_dims_axis_1(%t: tensor<8xf32>) -> tensor<8x1xf32> {
  %0 = tt.expand_dims %t {axis = 1 : i32} : tensor<8xf32> -> tensor<8x1xf32>
  tt.return %0 : tensor<8x1xf32>
}

// -----
// tt.expand_dims into the middle of a 2-D input, so there are three loops and the
// dropped dim is an interior one: (d0, d1, d2) -> (d0, d2) keeps the outer and
// inner dims and omits d1. The surviving dims stay in order, which is what
// separates an insertion from a permutation -- a map of (d0, d2) is a widening,
// (d2, d0) would be a transpose.

// CHECK-DAG: #[[$ATTR_4:.+]] = affine_map<(d0, d1, d2) -> (d0, d2)>
// CHECK-DAG: #[[$ATTR_5:.+]] = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
// CHECK-LABEL:   tt.func @expand_dims_2d_middle_axis(
// CHECK-SAME:  %[[VAL_0:.*]]: tensor<4x8xf16>) -> tensor<4x1x8xf16> {
// CHECK:           %[[VAL_1:.*]] = tensor.empty() : tensor<4x1x8xf16>
// CHECK:           %[[VAL_2:.*]] = linalg.generic {doc = "tt.expand_dims", indexing_maps = [#[[$ATTR_4]], #[[$ATTR_5]]], iterator_types = ["parallel", "parallel", "parallel"]} ins(%[[VAL_0]] : tensor<4x8xf16>) outs(%[[VAL_1]] : tensor<4x1x8xf16>) {
// CHECK:           ^bb0(%[[VAL_3:.*]]: f16, %[[VAL_4:.*]]: f16):
// CHECK:             linalg.yield %[[VAL_3]] : f16
// CHECK:           } -> tensor<4x1x8xf16>
// CHECK-NOT:       tt.expand_dims
// CHECK:           tt.return %[[VAL_2]] : tensor<4x1x8xf16>
// CHECK:         }
tt.func @expand_dims_2d_middle_axis(%t: tensor<4x8xf16>) -> tensor<4x1x8xf16> {
  %0 = tt.expand_dims %t {axis = 1 : i32} : tensor<4x8xf16> -> tensor<4x1x8xf16>
  tt.return %0 : tensor<4x1x8xf16>
}

// -----
// tt.broadcast expanding dim 0, from 1x8 to 4x8.
//
// There is no reshape step at all: the size-1 dim stays in the operand type and
// the expansion is expressed by the map (d0, d1) -> (0, d1), whose literal 0 in
// position 0 reads row 0 of the input for every value of d0. That constant is the
// broadcast -- an operand map of (d0, d1) would be an elementwise copy and would
// not even verify against a 1x8 input.
//
// Triton source pattern:
//
//   row = tl.load(row_ptr + tl.arange(0, N))   # tensor<N x f32>
//   row = tl.expand_dims(row, axis=0)          # tensor<1 x N x f32>
//   mat = tl.broadcast_to(row, [M, N])         # tensor<M x N x f32>

// CHECK-DAG: #[[$ATTR_6:.+]] = affine_map<(d0, d1) -> (0, d1)>
// CHECK-DAG: #[[$ATTR_7:.+]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-LABEL:   tt.func @broadcast_expand_first_dim(
// CHECK-SAME:  %[[VAL_0:.*]]: tensor<1x8xf32>) -> tensor<4x8xf32> {
// CHECK:           %[[VAL_1:.*]] = tensor.empty() : tensor<4x8xf32>
// CHECK:           %[[VAL_2:.*]] = linalg.generic {doc = "tt.broadcast", indexing_maps = [#[[$ATTR_6]], #[[$ATTR_7]]], iterator_types = ["parallel", "parallel"]} ins(%[[VAL_0]] : tensor<1x8xf32>) outs(%[[VAL_1]] : tensor<4x8xf32>) {
// CHECK:           ^bb0(%[[VAL_3:.*]]: f32, %[[VAL_4:.*]]: f32):
// CHECK:             linalg.yield %[[VAL_3]] : f32
// CHECK:           } -> tensor<4x8xf32>
// CHECK-NOT:       tt.broadcast
// CHECK:           tt.return %[[VAL_2]] : tensor<4x8xf32>
// CHECK:         }
tt.func @broadcast_expand_first_dim(%t: tensor<1x8xf32>) -> tensor<4x8xf32> {
  %0 = tt.broadcast %t : tensor<1x8xf32> -> tensor<4x8xf32>
  tt.return %0 : tensor<4x8xf32>
}

// -----
// tt.broadcast expanding dim 1, from 4x1 to 4x8. The constant 0 moves to the
// second position: (d0, d1) -> (d0, 0). Contrast the case above, where it sat in
// the first -- the pair pins that the pattern identifies *which* dim is size 1
// rather than assuming a position.

// CHECK-DAG: #[[$ATTR_8:.+]] = affine_map<(d0, d1) -> (d0, 0)>
// CHECK-DAG: #[[$ATTR_9:.+]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-LABEL:   tt.func @broadcast_expand_last_dim(
// CHECK-SAME:  %[[VAL_0:.*]]: tensor<4x1xf32>) -> tensor<4x8xf32> {
// CHECK:           %[[VAL_1:.*]] = tensor.empty() : tensor<4x8xf32>
// CHECK:           %[[VAL_2:.*]] = linalg.generic {doc = "tt.broadcast", indexing_maps = [#[[$ATTR_8]], #[[$ATTR_9]]], iterator_types = ["parallel", "parallel"]} ins(%[[VAL_0]] : tensor<4x1xf32>) outs(%[[VAL_1]] : tensor<4x8xf32>) {
// CHECK:           ^bb0(%[[VAL_3:.*]]: f32, %[[VAL_4:.*]]: f32):
// CHECK:             linalg.yield %[[VAL_3]] : f32
// CHECK:           } -> tensor<4x8xf32>
// CHECK-NOT:       tt.broadcast
// CHECK:           tt.return %[[VAL_2]] : tensor<4x8xf32>
// CHECK:         }
tt.func @broadcast_expand_last_dim(%t: tensor<4x1xf32>) -> tensor<4x8xf32> {
  %0 = tt.broadcast %t : tensor<4x1xf32> -> tensor<4x8xf32>
  tt.return %0 : tensor<4x8xf32>
}

// -----
// Two non-adjacent dims broadcast at once: 1x8x1 to 4x8x16. Both expanded dims
// get their own constant 0 in the same map, (d0, d1, d2) -> (0, d1, 0), so one
// generic covers both -- not one per dim, which the guard below rules out. Only
// d1, the dim that is genuinely carried through, appears as a loop index.

// CHECK-DAG: #[[$ATTR_10:.+]] = affine_map<(d0, d1, d2) -> (0, d1, 0)>
// CHECK-DAG: #[[$ATTR_11:.+]] = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
// CHECK-LABEL:   tt.func @broadcast_expand_multiple_dims(
// CHECK-SAME:  %[[VAL_0:.*]]: tensor<1x8x1xf16>) -> tensor<4x8x16xf16> {
// CHECK:           %[[VAL_1:.*]] = tensor.empty() : tensor<4x8x16xf16>
// CHECK:           %[[VAL_2:.*]] = linalg.generic {doc = "tt.broadcast", indexing_maps = [#[[$ATTR_10]], #[[$ATTR_11]]], iterator_types = ["parallel", "parallel", "parallel"]} ins(%[[VAL_0]] : tensor<1x8x1xf16>) outs(%[[VAL_1]] : tensor<4x8x16xf16>) {
// CHECK:           ^bb0(%[[VAL_3:.*]]: f16, %[[VAL_4:.*]]: f16):
// CHECK:             linalg.yield %[[VAL_3]] : f16
// CHECK:           } -> tensor<4x8x16xf16>
// CHECK-NOT:       linalg.generic
// CHECK-NOT:       tt.broadcast
// CHECK:           tt.return %[[VAL_2]] : tensor<4x8x16xf16>
// CHECK:         }
tt.func @broadcast_expand_multiple_dims(%t: tensor<1x8x1xf16>) -> tensor<4x8x16xf16> {
  %0 = tt.broadcast %t : tensor<1x8x1xf16> -> tensor<4x8x16xf16>
  tt.return %0 : tensor<4x8x16xf16>
}

// -----
// Every dim is a broadcast dim: 1x1 to 4x8. The operand map degenerates to all
// constants, (d0, d1) -> (0, 0) -- no loop index reaches the input at all, which
// is the edge case a pattern that assumed at least one carried dim would get
// wrong.

// CHECK-DAG: #[[$ATTR_12:.+]] = affine_map<(d0, d1) -> (0, 0)>
// CHECK-DAG: #[[$ATTR_13:.+]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-LABEL:   tt.func @broadcast_all_dims(
// CHECK-SAME:  %[[VAL_0:.*]]: tensor<1x1xf32>) -> tensor<4x8xf32> {
// CHECK:           %[[VAL_1:.*]] = tensor.empty() : tensor<4x8xf32>
// CHECK:           %[[VAL_2:.*]] = linalg.generic {doc = "tt.broadcast", indexing_maps = [#[[$ATTR_12]], #[[$ATTR_13]]], iterator_types = ["parallel", "parallel"]} ins(%[[VAL_0]] : tensor<1x1xf32>) outs(%[[VAL_1]] : tensor<4x8xf32>) {
// CHECK:           ^bb0(%[[VAL_3:.*]]: f32, %[[VAL_4:.*]]: f32):
// CHECK:             linalg.yield %[[VAL_3]] : f32
// CHECK:           } -> tensor<4x8xf32>
// CHECK-NOT:       tt.broadcast
// CHECK:           tt.return %[[VAL_2]] : tensor<4x8xf32>
// CHECK:         }
tt.func @broadcast_all_dims(%t: tensor<1x1xf32>) -> tensor<4x8xf32> {
  %0 = tt.broadcast %t : tensor<1x1xf32> -> tensor<4x8xf32>
  tt.return %0 : tensor<4x8xf32>
}

// -----
// A tt.broadcast whose input and result shapes are equal has no size-1 dim to
// expand, so it is a no-op: the op is replaced by its operand outright and the
// function body reduces to a bare return of the argument. Nothing is built --
// no empty, no linalg.generic -- which the guards pin. This is the one broadcast
// case with no positive op to match, so the return-of-argument line is what
// anchors the guards inside the body.

// CHECK-LABEL:   tt.func @broadcast_noop_same_shape(
// CHECK-SAME:  %[[VAL_0:.*]]: tensor<4x8xf32>) -> tensor<4x8xf32> {
// CHECK-NOT:       tt.broadcast
// CHECK-NOT:       linalg.generic
// CHECK-NOT:       tensor.empty
// CHECK:           tt.return %[[VAL_0]] : tensor<4x8xf32>
// CHECK:         }
tt.func @broadcast_noop_same_shape(%t: tensor<4x8xf32>) -> tensor<4x8xf32> {
  %0 = tt.broadcast %t : tensor<4x8xf32> -> tensor<4x8xf32>
  tt.return %0 : tensor<4x8xf32>
}

// -----
// tt.trans 2-D, swapping rows and columns. tensor.empty materializes the
// transposed destination and the order attribute array<i32: 1, 0> reaches the
// generic as the permuted operand map (d0, d1) -> (d1, d0).
//
// A transpose is a data movement, not a reinterpretation, so the guard against
// tensor.reshape below matters: collapsing this to a reshape of the same element
// count would silently produce the wrong values.
//
// Triton source pattern:
//
//   a   = tl.load(a_desc, [m * BM, k * BK])   # tensor<BM x BK x f32>
//   a_t = tl.trans(a)                          # tensor<BK x BM x f32>

// CHECK-DAG: #[[$ATTR_14:.+]] = affine_map<(d0, d1) -> (d1, d0)>
// CHECK-DAG: #[[$ATTR_15:.+]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-LABEL:   tt.func @trans_2d_transpose(
// CHECK-SAME:  %[[VAL_0:.*]]: tensor<4x8xf32>) -> tensor<8x4xf32> {
// CHECK:           %[[VAL_1:.*]] = tensor.empty() : tensor<8x4xf32>
// CHECK:           %[[VAL_2:.*]] = linalg.generic {doc = "tt.trans", indexing_maps = [#[[$ATTR_14]], #[[$ATTR_15]]], iterator_types = ["parallel", "parallel"]} ins(%[[VAL_0]] : tensor<4x8xf32>) outs(%[[VAL_1]] : tensor<8x4xf32>) {
// CHECK:           ^bb0(%[[VAL_3:.*]]: f32, %[[VAL_4:.*]]: f32):
// CHECK:             linalg.yield %[[VAL_3]] : f32
// CHECK:           } -> tensor<8x4xf32>
// CHECK-NOT:       tt.trans
// CHECK-NOT:       tensor.reshape
// CHECK:           tt.return %[[VAL_2]] : tensor<8x4xf32>
// CHECK:         }
tt.func @trans_2d_transpose(%t: tensor<4x8xf32>) -> tensor<8x4xf32> {
  %0 = tt.trans %t {order = array<i32: 1, 0>} : tensor<4x8xf32> -> tensor<8x4xf32>
  tt.return %0 : tensor<8x4xf32>
}

// -----
// tt.trans 3-D with a genuine 3-cycle permutation [2, 0, 1] -- not a swap, so a
// lowering that only handled pairwise exchange would fail here. The order attribute
// says output dim i reads input dim order[i], so the operand map comes out
// (d0, d1, d2) -> (d1, d2, d0). That exact map is what pins against an inverted-
// permutation bug: the inverse would give (d2, d0, d1), which is still
// shape-correct and still verifies, but moves the data the wrong way.

// CHECK-DAG: #[[$ATTR_16:.+]] = affine_map<(d0, d1, d2) -> (d1, d2, d0)>
// CHECK-DAG: #[[$ATTR_17:.+]] = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
// CHECK-LABEL:   tt.func @trans_3d_permute(
// CHECK-SAME:  %[[VAL_0:.*]]: tensor<2x4x8xf16>) -> tensor<8x2x4xf16> {
// CHECK:           %[[VAL_1:.*]] = tensor.empty() : tensor<8x2x4xf16>
// CHECK:           %[[VAL_2:.*]] = linalg.generic {doc = "tt.trans", indexing_maps = [#[[$ATTR_16]], #[[$ATTR_17]]], iterator_types = ["parallel", "parallel", "parallel"]} ins(%[[VAL_0]] : tensor<2x4x8xf16>) outs(%[[VAL_1]] : tensor<8x2x4xf16>) {
// CHECK:           ^bb0(%[[VAL_3:.*]]: f16, %[[VAL_4:.*]]: f16):
// CHECK:             linalg.yield %[[VAL_3]] : f16
// CHECK:           } -> tensor<8x2x4xf16>
// CHECK-NOT:       tt.trans
// CHECK:           tt.return %[[VAL_2]] : tensor<8x2x4xf16>
// CHECK:         }
tt.func @trans_3d_permute(%t: tensor<2x4x8xf16>) -> tensor<8x2x4xf16> {
  %0 = tt.trans %t {order = array<i32: 2, 0, 1>} : tensor<2x4x8xf16> -> tensor<8x2x4xf16>
  tt.return %0 : tensor<8x2x4xf16>
}

// RUN: spyre-triton-opt %s --lower-descriptor-memory --lower-scalar-load --lower-compute-ops --rewrite-descriptor-layout-generic -verify-diagnostics

// A decline must name the op, not surface as a verifier failure.
//
// A named linalg op is fixed at logical rank and so cannot carry a stick dim.
// This pass rewrites only generics, so it rejects such an op up front, before
// Phase 1 retypes the load underneath it. The ordering is the point: retyping
// first would leave a rank mismatch that MLIR's own verifier reports against an
// indexing map, naming neither this pass nor what it could not restate.
//
// The named op is written out DIRECTLY rather than obtained by lowering a tt.dot.
// LowerComputeOps used to emit a named linalg.matmul for tt.dot and now emits a
// linalg.generic, so routing through tt.dot would no longer produce a named op at
// all and this decline would go untested. Writing it by hand keeps the test about
// the pass's own behaviour, and keeps it valid whatever the frontend emits.
module {
tt.func @named_matmul_declines(%a: !tt.ptr<f16>, %b: !tt.ptr<f16>, %c: !tt.ptr<f16>) {
  %c0_i32 = arith.constant 0 : i32
  %c64_i32 = arith.constant 64 : i32
  %c128_i32 = arith.constant 128 : i32
  %c1_i64 = arith.constant 1 : i64
  %c64_i64 = arith.constant 64 : i64
  %c128_i64 = arith.constant 128 : i64
  %ad = tt.make_tensor_descriptor %a, [%c128_i32, %c64_i32], [%c64_i64, %c1_i64] : !tt.ptr<f16>, !tt.tensordesc<128x64xf16>
  // A[M=128, K=64] stick-on-M(64) -> physical [M/64, K, M%64] = [2, 64, 64]
  tt.spyre_tensor_layout %ad {phys_src = array<i64: 0, 1, 0>, phys_op = array<i64: 1, 0, 2>, phys_arg = array<i64: 64, 0, 64>} : !tt.tensordesc<128x64xf16>
  %av = tt.descriptor_load %ad[%c0_i32, %c0_i32] : !tt.tensordesc<128x64xf16> -> tensor<128x64xf16>
  %bd = tt.make_tensor_descriptor %b, [%c64_i32, %c64_i32], [%c64_i64, %c1_i64] : !tt.ptr<f16>, !tt.tensordesc<64x64xf16>
  %bv = tt.descriptor_load %bd[%c0_i32, %c0_i32] : !tt.tensordesc<64x64xf16> -> tensor<64x64xf16>
  %cd = tt.make_tensor_descriptor %c, [%c128_i32, %c64_i32], [%c64_i64, %c1_i64] : !tt.ptr<f16>, !tt.tensordesc<128x64xf16>
  %cv = tt.descriptor_load %cd[%c0_i32, %c0_i32] : !tt.tensordesc<128x64xf16> -> tensor<128x64xf16>
  // expected-error @below {{rewrite-descriptor-layout-generic: this op reads a value on a physicalized chain, but the rewrite restates only linalg.generic; spell this op as one}}
  %d = linalg.matmul ins(%av, %bv : tensor<128x64xf16>, tensor<64x64xf16>) outs(%cv : tensor<128x64xf16>) -> tensor<128x64xf16>
  tt.descriptor_store %cd[%c0_i32, %c0_i32], %d : !tt.tensordesc<128x64xf16>, tensor<128x64xf16>
  tt.return
}
}

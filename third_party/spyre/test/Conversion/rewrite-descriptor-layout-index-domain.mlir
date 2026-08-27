// RUN: spyre-triton-opt %s --lower-descriptor-memory --lower-scalar-load --lower-compute-ops --rewrite-descriptor-layout -split-input-file | FileCheck %s

// Triton computes offsets in i32, so a pid-derived subscript arrives as
// index_cast(muli(index_cast(pid), c) : i32).  The coordinate split is emitted
// in `index`, and its input is lifted there too: the scheduler's symbolic
// start-address analysis treats a cast as opaque and rejects an address
// computed through one.  Checks are hand-written and minimal on purpose --
// the claim is about which domain the subscript arithmetic lands in, not about
// the whole module.
//
// This pass leaves the original i32 chain in place (dead, once nothing else
// reads it) and emits one index multiply per subscript; the canonicalizer and
// CSE that the frontend runs after DistributeWork collapse both.

// CHECK-LABEL:   tt.func @pid_offset_lifted_to_index(
// CHECK:           %[[PID:.*]] = tt.get_program_id x : i32
// The floordiv subscript: pid cast once, then multiplied in `index`.
// CHECK:           %[[PIDX:.*]] = arith.index_cast %[[PID]] : i32 to index
// CHECK:           %[[C64:.*]] = arith.constant 64 : index
// CHECK:           %[[OFF:.*]] = arith.muli %[[PIDX]], %[[C64]] : index
// CHECK:           arith.divsi %[[OFF]], %{{.*}} : index
// The mod subscript, same shape.
// CHECK:           %[[PIDX2:.*]] = arith.index_cast %[[PID]] : i32 to index
// CHECK:           %[[OFF2:.*]] = arith.muli %[[PIDX2]], %{{.*}} : index
// CHECK:           arith.remsi %[[OFF2]], %{{.*}} : index
tt.func @pid_offset_lifted_to_index(%ptr: !tt.ptr<f16>) {
  %c64_i32 = arith.constant 64 : i32
  %c128_i32 = arith.constant 128 : i32
  %c1_i64 = arith.constant 1 : i64
  %pid = tt.get_program_id x : i32
  %off = arith.muli %pid, %c64_i32 : i32
  // [n=128] stick-on-n with stick_size=64 -> physical [n/64, n%64] = [2, 64]
  %desc = tt.make_tensor_descriptor %ptr, [%c128_i32], [%c1_i64]
      : !tt.ptr<f16>, !tt.tensordesc<64xf16>
  tt.spyre_tensor_layout %desc {phys_src = array<i64: 0, 0>, phys_op = array<i64: 1, 2>, phys_arg = array<i64: 64, 64>} : !tt.tensordesc<64xf16>
  %d = tt.descriptor_load %desc[%off] : !tt.tensordesc<64xf16> -> tensor<64xf16>
  tt.descriptor_store %desc[%off], %d : !tt.tensordesc<64xf16>, tensor<64xf16>
  tt.return
}

// -----

// A run-time i32 scalar is not a grid coordinate, so its arithmetic keeps the
// width Triton gave it -- rebuilding in 64-bit `index` would change what the
// expression means on overflow.  This is the path the dynamic-shape kernels
// take, and it must reach the split exactly as it did before.

// CHECK-LABEL:   tt.func @runtime_scalar_offset_unchanged(
// CHECK-SAME:      %{{.*}}: !tt.ptr<f16>, %[[N:.*]]: i32)
// CHECK:           %[[OFF:.*]] = arith.muli %[[N]], %{{.*}} : i32
// CHECK:           %[[OFFX:.*]] = arith.index_cast %[[OFF]] : i32 to index
// CHECK:           arith.divsi %[[OFFX]], %{{.*}} : index
// CHECK:           arith.remsi %[[OFFX]], %{{.*}} : index
tt.func @runtime_scalar_offset_unchanged(%ptr: !tt.ptr<f16>, %n: i32) {
  %c64_i32 = arith.constant 64 : i32
  %c128_i32 = arith.constant 128 : i32
  %c1_i64 = arith.constant 1 : i64
  %off = arith.muli %n, %c64_i32 : i32
  %desc = tt.make_tensor_descriptor %ptr, [%c128_i32], [%c1_i64]
      : !tt.ptr<f16>, !tt.tensordesc<64xf16>
  tt.spyre_tensor_layout %desc {phys_src = array<i64: 0, 0>, phys_op = array<i64: 1, 2>, phys_arg = array<i64: 64, 64>} : !tt.tensordesc<64xf16>
  %d = tt.descriptor_load %desc[%off] : !tt.tensordesc<64xf16> -> tensor<64xf16>
  tt.descriptor_store %desc[%off], %d : !tt.tensordesc<64xf16>, tensor<64xf16>
  tt.return
}

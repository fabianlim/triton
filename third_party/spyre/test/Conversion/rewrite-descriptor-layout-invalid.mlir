// RUN: spyre-triton-opt %s --lower-descriptor-memory --lower-scalar-load --lower-compute-ops --rewrite-descriptor-layout -split-input-file -verify-diagnostics

// Test 1: block extent of stick dim is smaller than the stick size.
// The descriptor is 64x32 but phys_arg (stick_size) is 64 on dim 1,
// meaning the load's block extent on dim 1 (32) < stick_size (64).
module {
tt.func @block_smaller_than_stick(%ptr: !tt.ptr<f32>, %out: !tt.ptr<f32>) {
  %c0_i32 = arith.constant 0 : i32
  %c64_i32 = arith.constant 64 : i32
  %c32_i32 = arith.constant 32 : i32
  %c32_i64 = arith.constant 32 : i64
  %c1_i64 = arith.constant 1 : i64
  %desc = tt.make_tensor_descriptor %ptr, [%c64_i32, %c32_i32], [%c32_i64, %c1_i64]
      : !tt.ptr<f32>, !tt.tensordesc<64x32xf32>
  // Stick-on-N with stick_size=64, but N=32 < 64
  tt.spyre_tensor_layout %desc {phys_src = array<i64: 1, 0, 1>, phys_op = array<i64: 1, 0, 2>, phys_arg = array<i64: 64, 0, 64>} : !tt.tensordesc<64x32xf32>
  // expected-error @below {{spyre_tensor_layout: block extent of stick dim (32) is smaller than the stick size (64); a stick dim cannot be sub-stick}}
  %d = tt.descriptor_load %desc[%c0_i32, %c0_i32] : !tt.tensordesc<64x32xf32> -> tensor<64x32xf32>
  %out_desc = tt.make_tensor_descriptor %out, [%c64_i32, %c32_i32], [%c32_i64, %c1_i64]
      : !tt.ptr<f32>, !tt.tensordesc<64x32xf32>
  tt.descriptor_store %out_desc[%c0_i32, %c0_i32], %d : !tt.tensordesc<64x32xf32>, tensor<64x32xf32>
  tt.return
}
}

// -----

// Test 2: stick-splitting the indirect (gather) row dim is not supported.
// The layout annotation applies floordiv to the gather's indirect dimension (dim 0).
module {
tt.func @gather_stick_split_indirect_dim(%data_ptr: !tt.ptr<f32>, %idx_ptr: !tt.ptr<i32>, %out: !tt.ptr<f32>) {
  %c0_i32 = arith.constant 0 : i32
  %c512_i32 = arith.constant 512 : i32
  %c128_i32 = arith.constant 128 : i32
  %c128_i64 = arith.constant 128 : i64
  %c1_i64 = arith.constant 1 : i64
  %c32_i32 = arith.constant 32 : i32
  %c32_i64 = arith.constant 32 : i64

  // Index descriptor: 32-element 1-D tensor holding row indices.
  %idx_desc = tt.make_tensor_descriptor %idx_ptr, [%c32_i32], [%c1_i64]
      : !tt.ptr<i32>, !tt.tensordesc<32xi32>
  %x_offsets = tt.descriptor_load %idx_desc[%c0_i32] : !tt.tensordesc<32xi32> -> tensor<32xi32>

  // Data descriptor: [512, 128] with stick-on-M layout (stick=64 on dim 0).
  //   phys_src=[0, 1, 0] phys_op=[1, 0, 2] phys_arg=[64, 0, 64]
  //   This applies floordiv to dim 0, which is the indirect (gather) dim.
  %data_desc = tt.make_tensor_descriptor %data_ptr, [%c512_i32, %c128_i32], [%c128_i64, %c1_i64]
      : !tt.ptr<f32>, !tt.tensordesc<1x128xf32>
  tt.spyre_tensor_layout %data_desc {phys_src = array<i64: 0, 1, 0>, phys_op = array<i64: 1, 0, 2>, phys_arg = array<i64: 64, 0, 64>} : !tt.tensordesc<1x128xf32>

  // Gather 32 non-contiguous rows — error: stick-splitting the indirect row dim.
  // expected-error @below {{spyre_tensor_layout: stick-splitting the indirect (gather) row dim is not supported}}
  %gathered = tt.descriptor_gather %data_desc[%x_offsets, %c0_i32]
      : (!tt.tensordesc<1x128xf32>, tensor<32xi32>, i32) -> tensor<32x128xf32>

  %out_desc = tt.make_tensor_descriptor %out, [%c32_i32, %c128_i32], [%c128_i64, %c1_i64]
      : !tt.ptr<f32>, !tt.tensordesc<32x128xf32>
  tt.descriptor_store %out_desc[%c0_i32, %c0_i32], %gathered : !tt.tensordesc<32x128xf32>, tensor<32x128xf32>
  tt.return
}
}

// RUN: spyre-triton-opt %s --hbm-roundtrip -split-input-file | FileCheck %s

// Two chained computes: the exp result goes to a spill buffer and is read back
// before the sqrt, so the two become separate compute groups. The buffer is a new
// `index` argument (%arg2) and is described on the module for the launcher.
//
// Every store and every load gets its own construct_access_tile, and each
// memory op its own memory view and tile-id arithmetic -- the scheduler moves a
// group's operations into a schedule of their own and asserts on anything two
// groups share.
#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#set_tile = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 + 5 >= 0, d1 >= 0, -d1 + 63 >= 0, d2 >= 0, -d2 + 63 >= 0)>
#set_whole = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 + 11 >= 0, d1 >= 0, -d1 + 63 >= 0, d2 >= 0, -d2 + 63 >= 0)>

// CHECK: module attributes {ktdp.hbm_roundtrip_buffers = [{element_bits = 32 : i64, element_type = f32, shape = array<i64: 12, 64, 64>}]}
// CHECK-LABEL: func.func @two_computes(
// CHECK-SAME:      %[[IN:.*]]: index, %[[OUT:.*]]: index, %[[SPILL:.*]]: index) attributes

// Group 0 reads the input and writes the spill buffer.
// CHECK:         %[[LOADED:.*]] = ktdp.load
// CHECK:         %[[EXP:.*]] = linalg.generic
// CHECK:           spyreop.exp
// CHECK:         %[[SPILL_VIEW_W:.*]] = ktdp.construct_memory_view %[[SPILL]], sizes: [12, 64, 64], strides: [4096, 64, 1]
// CHECK:         %[[SPILL_TILE_W:.*]] = ktdp.construct_access_tile %[[SPILL_VIEW_W]]
// CHECK:         ktdp.store %[[EXP]], %[[SPILL_TILE_W]]

// Group 1 reads it back through a view and access tile of its own.
// CHECK:         %[[SPILL_VIEW_R:.*]] = ktdp.construct_memory_view %[[SPILL]], sizes: [12, 64, 64], strides: [4096, 64, 1]
// CHECK:         %[[SPILL_TILE_R:.*]] = ktdp.construct_access_tile %[[SPILL_VIEW_R]]
// CHECK:         %[[BACK:.*]] = ktdp.load %[[SPILL_TILE_R]]
// CHECK:         %[[SQRT:.*]] = linalg.generic {{.*}} ins(%[[BACK]]
// CHECK:           spyreop.sqrt
// CHECK:         ktdp.store %[[SQRT]]
module {
  func.func @two_computes(%base_in: index, %base_out: index) attributes {grid = [2]} {
    %zero = arith.constant 0 : index
    %tid = ktdp.get_compute_tile_id : index

    %view_in = ktdp.construct_memory_view %base_in, sizes: [12, 64, 64], strides: [4096, 64, 1] {coordinate_set = #set_whole, memory_space = #ktdp.memory_space<global>} : memref<12x64x64xf32>
    %tile_in = ktdp.construct_access_tile %view_in[%tid * 6, %zero, %zero] {access_tile_order = #map, access_tile_set = #set_tile} : memref<12x64x64xf32> -> !ktdp.access_tile<6x64x64xindex>
    %view_out = ktdp.construct_memory_view %base_out, sizes: [12, 64, 64], strides: [4096, 64, 1] {coordinate_set = #set_whole, memory_space = #ktdp.memory_space<global>} : memref<12x64x64xf32>
    %tile_out = ktdp.construct_access_tile %view_out[%tid * 6, %zero, %zero] {access_tile_order = #map, access_tile_set = #set_tile} : memref<12x64x64xf32> -> !ktdp.access_tile<6x64x64xindex>

    %in = ktdp.load %tile_in : <6x64x64xindex> -> tensor<6x64x64xf32>
    %init0 = tensor.empty() : tensor<6x64x64xf32>
    %mid = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%in : tensor<6x64x64xf32>) outs(%init0 : tensor<6x64x64xf32>) {
    ^bb0(%x: f32, %out: f32):
      %e = spyreop.exp %x : f32
      linalg.yield %e : f32
    } -> tensor<6x64x64xf32>
    %init1 = tensor.empty() : tensor<6x64x64xf32>
    %result = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%mid : tensor<6x64x64xf32>) outs(%init1 : tensor<6x64x64xf32>) {
    ^bb0(%x: f32, %out: f32):
      %s = spyreop.sqrt %x : f32
      linalg.yield %s : f32
    } -> tensor<6x64x64xf32>
    ktdp.store %result, %tile_out : tensor<6x64x64xf32>, <6x64x64xindex>
    return
  }
}

// -----

// Three chained computes need only ONE buffer: the middle compute reads the
// first spill and writes the second, so by the time its store lands the earlier
// value has been read and the buffer is free. Reuse is what keeps a chain inside
// the seven base addresses Spyre has.
#map = affine_map<(d0) -> (d0)>
#set = affine_set<(d0) : (d0 >= 0, -d0 + 127 >= 0)>

// One buffer, so one entry in the list and one added argument. The closing paren
// on the signature is what makes that a count rather than a lower bound.
// CHECK: module attributes {ktdp.hbm_roundtrip_buffers = [{element_bits = 32 : i64, element_type = f32, shape = array<i64: 128>}]}
// CHECK-LABEL: func.func @three_computes(
// CHECK-SAME:      %[[IN:.*]]: index, %[[OUT:.*]]: index, %[[SPILL:.*]]: index) attributes
module {
  func.func @three_computes(%base_in: index, %base_out: index) attributes {grid = [1]} {
    %zero = arith.constant 0 : index
    %view_in = ktdp.construct_memory_view %base_in, sizes: [128], strides: [1] {coordinate_set = #set, memory_space = #ktdp.memory_space<global>} : memref<128xf32>
    %tile_in = ktdp.construct_access_tile %view_in[%zero] {access_tile_order = #map, access_tile_set = #set} : memref<128xf32> -> !ktdp.access_tile<128xindex>
    %view_out = ktdp.construct_memory_view %base_out, sizes: [128], strides: [1] {coordinate_set = #set, memory_space = #ktdp.memory_space<global>} : memref<128xf32>
    %tile_out = ktdp.construct_access_tile %view_out[%zero] {access_tile_order = #map, access_tile_set = #set} : memref<128xf32> -> !ktdp.access_tile<128xindex>

    %in = ktdp.load %tile_in : <128xindex> -> tensor<128xf32>
    %init0 = tensor.empty() : tensor<128xf32>
    %a = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%in : tensor<128xf32>) outs(%init0 : tensor<128xf32>) {
    ^bb0(%x: f32, %out: f32):
      %e = spyreop.exp %x : f32
      linalg.yield %e : f32
    } -> tensor<128xf32>
    %init1 = tensor.empty() : tensor<128xf32>
    %b = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%a : tensor<128xf32>) outs(%init1 : tensor<128xf32>) {
    ^bb0(%x: f32, %out: f32):
      %s = spyreop.sqrt %x : f32
      linalg.yield %s : f32
    } -> tensor<128xf32>
    %init2 = tensor.empty() : tensor<128xf32>
    %c = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%b : tensor<128xf32>) outs(%init2 : tensor<128xf32>) {
    ^bb0(%x: f32, %out: f32):
      %e = spyreop.exp %x : f32
      linalg.yield %e : f32
    } -> tensor<128xf32>
    ktdp.store %c, %tile_out : tensor<128xf32>, <128xindex>
    return
  }
}

// -----

// A value the kernel already stores to HBM ahead of its compute consumers needs
// no buffer at all: each consumer reads it back from that buffer. So no new
// argument, and no module attribute.
#map = affine_map<(d0) -> (d0)>
#set = affine_set<(d0) : (d0 >= 0, -d0 + 127 >= 0)>

// CHECK-NOT: ktdp.hbm_roundtrip_buffers
// CHECK-LABEL: func.func @reuses_the_kernels_own_store(
// CHECK-SAME:      %[[IN:.*]]: index, %[[MID:.*]]: index, %[[OUT:.*]]: index) attributes
// CHECK:         %[[FIRST:.*]] = linalg.generic
// CHECK:         ktdp.store %[[FIRST]]
// CHECK:         %[[MID_VIEW:.*]] = ktdp.construct_memory_view %[[MID]]
// CHECK:         %[[MID_TILE:.*]] = ktdp.construct_access_tile %[[MID_VIEW]]
// CHECK:         %[[BACK:.*]] = ktdp.load %[[MID_TILE]]
// CHECK:         linalg.generic {{.*}} ins(%[[BACK]]
module {
  func.func @reuses_the_kernels_own_store(%base_in: index, %base_mid: index, %base_out: index) attributes {grid = [1]} {
    %zero = arith.constant 0 : index
    %view_in = ktdp.construct_memory_view %base_in, sizes: [128], strides: [1] {coordinate_set = #set, memory_space = #ktdp.memory_space<global>} : memref<128xf32>
    %tile_in = ktdp.construct_access_tile %view_in[%zero] {access_tile_order = #map, access_tile_set = #set} : memref<128xf32> -> !ktdp.access_tile<128xindex>
    %view_mid = ktdp.construct_memory_view %base_mid, sizes: [128], strides: [1] {coordinate_set = #set, memory_space = #ktdp.memory_space<global>} : memref<128xf32>
    %tile_mid = ktdp.construct_access_tile %view_mid[%zero] {access_tile_order = #map, access_tile_set = #set} : memref<128xf32> -> !ktdp.access_tile<128xindex>
    %view_out = ktdp.construct_memory_view %base_out, sizes: [128], strides: [1] {coordinate_set = #set, memory_space = #ktdp.memory_space<global>} : memref<128xf32>
    %tile_out = ktdp.construct_access_tile %view_out[%zero] {access_tile_order = #map, access_tile_set = #set} : memref<128xf32> -> !ktdp.access_tile<128xindex>

    %in = ktdp.load %tile_in : <128xindex> -> tensor<128xf32>
    %init0 = tensor.empty() : tensor<128xf32>
    %a = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%in : tensor<128xf32>) outs(%init0 : tensor<128xf32>) {
    ^bb0(%x: f32, %out: f32):
      %e = spyreop.exp %x : f32
      linalg.yield %e : f32
    } -> tensor<128xf32>
    ktdp.store %a, %tile_mid : tensor<128xf32>, <128xindex>
    %init1 = tensor.empty() : tensor<128xf32>
    %b = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%a : tensor<128xf32>) outs(%init1 : tensor<128xf32>) {
    ^bb0(%x: f32, %out: f32):
      %s = spyreop.sqrt %x : f32
      linalg.yield %s : f32
    } -> tensor<128xf32>
    ktdp.store %b, %tile_out : tensor<128xf32>, <128xindex>
    return
  }
}

// -----

// A single compute has no compute-to-compute edge, so the pass leaves the
// function exactly as it found it -- same arguments, no module attribute, and no
// extra memory operations.
#map = affine_map<(d0) -> (d0)>
#set = affine_set<(d0) : (d0 >= 0, -d0 + 127 >= 0)>

// CHECK-NOT: ktdp.hbm_roundtrip_buffers
// CHECK-LABEL: func.func @one_compute(
// CHECK-SAME:      %[[IN:.*]]: index, %[[OUT:.*]]: index) attributes
// CHECK-COUNT-1: ktdp.load
// CHECK-COUNT-1: ktdp.store
// CHECK-NOT:     ktdp.load
module {
  func.func @one_compute(%base_in: index, %base_out: index) attributes {grid = [1]} {
    %zero = arith.constant 0 : index
    %view_in = ktdp.construct_memory_view %base_in, sizes: [128], strides: [1] {coordinate_set = #set, memory_space = #ktdp.memory_space<global>} : memref<128xf32>
    %tile_in = ktdp.construct_access_tile %view_in[%zero] {access_tile_order = #map, access_tile_set = #set} : memref<128xf32> -> !ktdp.access_tile<128xindex>
    %view_out = ktdp.construct_memory_view %base_out, sizes: [128], strides: [1] {coordinate_set = #set, memory_space = #ktdp.memory_space<global>} : memref<128xf32>
    %tile_out = ktdp.construct_access_tile %view_out[%zero] {access_tile_order = #map, access_tile_set = #set} : memref<128xf32> -> !ktdp.access_tile<128xindex>

    %in = ktdp.load %tile_in : <128xindex> -> tensor<128xf32>
    %init = tensor.empty() : tensor<128xf32>
    %a = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%in : tensor<128xf32>) outs(%init : tensor<128xf32>) {
    ^bb0(%x: f32, %out: f32):
      %e = spyreop.exp %x : f32
      linalg.yield %e : f32
    } -> tensor<128xf32>
    ktdp.store %a, %tile_out : tensor<128xf32>, <128xindex>
    return
  }
}

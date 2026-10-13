// RUN: spyre-triton-opt %s -split-input-file -verify-diagnostics | FileCheck %s

// The tts.spyreop_hint ATTRIBUTE, checked by the dialect's attribute verifier on
// whichever op carries it. Only what one op can show: the two fields and their
// types, a target that is a registered op, and an op that is not a constant.

// The spelling LowerTTSMarkers writes verifies and round-trips.
// CHECK-LABEL: tt.func @verifies
// CHECK: math.exp {{.*}} {tts.spyreop_hint = {id = 0 : i64, target = "spyreop.sigmoid"}}
tt.func @verifies(%x: tensor<128xf32>) -> tensor<128xf32> {
  %e = math.exp %x {tts.spyreop_hint = {id = 0 : i64, target = "spyreop.sigmoid"}} : tensor<128xf32>
  tt.return %e : tensor<128xf32>
}

// -----

// Not a dictionary.
tt.func @not_a_dictionary(%x: tensor<128xf32>) -> tensor<128xf32> {
  // expected-error @+1 {{'tts.spyreop_hint' must be a dictionary of exactly a string 'target' and an i64 'id', got "sigmoid"}}
  %e = math.exp %x {tts.spyreop_hint = "sigmoid"} : tensor<128xf32>
  tt.return %e : tensor<128xf32>
}

// -----

// A missing field.
tt.func @missing_id(%x: tensor<128xf32>) -> tensor<128xf32> {
  // expected-error @+1 {{'tts.spyreop_hint' must be a dictionary of exactly a string 'target' and an i64 'id', got {target = "spyreop.sigmoid"}}}
  %e = math.exp %x {tts.spyreop_hint = {target = "spyreop.sigmoid"}} : tensor<128xf32>
  tt.return %e : tensor<128xf32>
}

// -----

// A field of the wrong type.
tt.func @i32_id(%x: tensor<128xf32>) -> tensor<128xf32> {
  // expected-error @+1 {{'tts.spyreop_hint' must be a dictionary of exactly a string 'target' and an i64 'id'}}
  %e = math.exp %x {tts.spyreop_hint = {id = 0 : i32, target = "spyreop.sigmoid"}} : tensor<128xf32>
  tt.return %e : tensor<128xf32>
}

// -----

// A field LowerTTSMarkers does not write.
tt.func @extra_field(%x: tensor<128xf32>) -> tensor<128xf32> {
  // expected-error @+1 {{'tts.spyreop_hint' must be a dictionary of exactly a string 'target' and an i64 'id'}}
  %e = math.exp %x {tts.spyreop_hint = {id = 0 : i64, target = "spyreop.sigmoid", size = 4 : i64}} : tensor<128xf32>
  tt.return %e : tensor<128xf32>
}

// -----

// A target that is not a registered op.
tt.func @unknown_target(%x: tensor<128xf32>) -> tensor<128xf32> {
  // expected-error @+1 {{'tts.spyreop_hint' targets "spyreop.no_such_op", which is not a registered op}}
  %e = math.exp %x {tts.spyreop_hint = {id = 0 : i64, target = "spyreop.no_such_op"}} : tensor<128xf32>
  tt.return %e : tensor<128xf32>
}

// -----

// A constant, which LowerTTSMarkers never hints.
tt.func @constant() -> tensor<128xf32> {
  // expected-error @+1 {{'tts.spyreop_hint' is not set on a constant}}
  %c = arith.constant {tts.spyreop_hint = {id = 0 : i64, target = "spyreop.sigmoid"}} dense<1.0> : tensor<128xf32>
  tt.return %c : tensor<128xf32>
}

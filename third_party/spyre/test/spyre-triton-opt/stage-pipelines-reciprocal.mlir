// RUN: spyre-triton-opt %s --spyre-ttir-to-ktir | FileCheck %s --check-prefix=KTIR
// RUN: spyre-triton-opt %s --spyre-ttir-to-ktir --spyre-prepare-spyrecode | FileCheck %s --check-prefix=PHYS
// A prefix of its own, with nothing but NOT directives, so it scans the whole
// output rather than the span between two positive checks.
// RUN: spyre-triton-opt %s --spyre-ttir-to-ktir --spyre-prepare-spyrecode | FileCheck %s --check-prefix=NOIMM
// The negative control, and the point of this file: the same stage with
// FoldDataMovementGenerics taken out of it. Spelled as the stage's pass list
// minus that one pass, because a registered pipeline cannot have a pass removed
// from the CLI.
// RUN: spyre-triton-opt %s --spyre-ttir-to-ktir | spyre-triton-opt --normalize-for-device --drop-reduction-init-fill --convert-elementwise-to-linalg --linalg-generalize-named-ops --unalias-linalg-outs --lower-spyre-ops --combine-spyre-ops | FileCheck %s --check-prefix=NOFOLD

// That a division by one leaves the stage as the UNARY intrinsic, and the
// ordering edge that makes it so.
//
// No single pass can make this claim, which is why it is here rather than in
// either pass's own test. LowerSpyreOps emits `spyreop.realdiv` unconditionally
// and CombineSpyreOps rewrites it, but only when the `1.0` is a constant visible
// from the divide -- and what puts it in that form is a third pass,
// FoldDataMovementGenerics, whose elementwise fusion folds the splat into the
// generic's body. Three passes, one outcome, and the two edges between them are
// the whole content.
//
// WHY IT MATTERS THAT THE CONSTANT GOES. A float immediate reaching a Spyre
// compute unit is not read back as it was written, so a divide by a rounded one
// is not the divide that was written. Nothing downstream refuses it -- dbo-opt
// has no diagnostic for a float immediate operand -- so a regression here is a
// wrong answer, not a failure. `reduce/softmax_on_stick` is the kernel that
// depends on it; this file is the same claim at a size a lit test can read.
//
// The kernel is the smallest thing that carries the shape: a 1-D reciprocal, the
// numerator a splat `arith.constant` beside the divide rather than a scalar, which
// is how `tl.full([N], 1.0)` arrives. No descriptor layout, no reduce -- what is
// under test is the pass sequence, not what any of them does to a compute op.

module {
  tt.func public @recip_kernel(%x_ptr: !tt.ptr<f32>, %out_ptr: !tt.ptr<f32>) attributes {noinline = false} {
    %n = arith.constant 1024 : i32
    %s = arith.constant 1 : i64
    %one = arith.constant dense<1.000000e+00> : tensor<1024xf32>
    %pid = tt.get_program_id x : i32
    %x_desc = tt.make_tensor_descriptor %x_ptr, [%n], [%s] : <f32>, <1024xf32>
    %o_desc = tt.make_tensor_descriptor %out_ptr, [%n], [%s] : <f32>, <1024xf32>
    %off = arith.muli %pid, %n : i32
    %x = tt.descriptor_load %x_desc[%off] : !tt.tensordesc<1024xf32> -> tensor<1024xf32>
    %r = arith.divf %one, %x : tensor<1024xf32>
    tt.descriptor_store %o_desc[%off], %r : !tt.tensordesc<1024xf32>, tensor<1024xf32>
    tt.return
  }
}

// The `ktir` artifact keeps the divide as written, on tensors and in arith. No
// spyreop op belongs to this stage at all -- both passes that make one are in
// `spyrecode` -- so a kernel that stops here is the arithmetic its author wrote.
//
// KTIR-LABEL: func.func @recip_kernel
// KTIR: arith.constant dense<1.000000e+00> : tensor<1024xf32>
// KTIR: arith.divf
// KTIR-NOT: spyreop

// Through the whole of `spyrecode`, the divide is gone and the unary intrinsic is
// what a scalar body holds.
//
// PHYS-LABEL: func.func @recip_kernel
// PHYS: linalg.generic
// PHYS: spyreop.reciprocal

// And the immediate is gone with it -- the claim the kernel depends on, scanned
// over the whole output rather than one span. Both spellings are named: the splat
// the author wrote and the scalar the fold would have left inside the body.
//
// NOIMM-NOT: spyreop.realdiv
// NOIMM-NOT: arith.constant dense<1.000000e+00>
// NOIMM-NOT: arith.constant 1.000000e+00

// THE NEGATIVE CONTROL. Without FoldDataMovementGenerics the numerator is still a
// splat `ins` operand and the body reads it as a BLOCK ARGUMENT, which no constant
// matcher reads -- so CombineSpyreOps declines, the binary op survives, and the
// float immediate ships. This is what the ordering edge is protecting against,
// and it is the case that would otherwise only be visible as a numerical answer.
//
// NOFOLD-LABEL: func.func @recip_kernel
// NOFOLD: arith.constant dense<1.000000e+00> : tensor<1024xf32>
// NOFOLD: spyreop.realdiv
// NOFOLD-NOT: spyreop.reciprocal

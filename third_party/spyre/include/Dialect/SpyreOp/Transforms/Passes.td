#ifndef SPYREOP_TRANSFORMS_PASSES
#define SPYREOP_TRANSFORMS_PASSES

include "mlir/Pass/PassBase.td"

// ============================================================================
// Transforms whose subject is the spyreop dialect's own intrinsics -- rewriting
// one spelling of a computation into another spelling of the SAME computation,
// both of them spyreop.  A pass here neither introduces spyreop from another
// dialect nor takes it anywhere else: that is a conversion, and it belongs in
// Conversion/TritonToKTIR/Passes.td, where LowerSpyreOps is.
//
// The split against the tree's other two pass libraries, since spyreop is the
// fourth subject and the criterion is worth stating once:
//   Conversion/TritonToKTIR/  crosses a dialect boundary (tt -> ktdp, linalg,
//                             spyreop, func)
//   Dialect/KTDP/Transforms/  ktdp's own abstractions -- memory views, access
//                             tiles
//   Dialect/SpyreOp/Transforms/  this one: spyreop's own intrinsics
//   Transforms/               upstream structure, or the whole program
//
// WHY THESE PATTERNS ARE NOT OP CANONICALIZERS.  A peephole on one dialect is
// ordinarily that dialect's own `hasCanonicalizer`, and upstream MLIR would put
// it there.  spyreop is defined in the ktir-mlir-frontend submodule, which is a
// pure op-definition library -- it declares no folder and no canonicalizer for
// any of its ops, and has no Transforms directory to put one in.  So a pass in
// this tree is where a spyreop peephole can live at all without coordinating a
// submodule change, and it is additionally where the POLICY belongs: which
// spelling the Spyre toolchain prefers is a fact about dbo-opt and the device,
// not about the op definition.
//
// Each pass declares its contract below:
//   Input    - dialects/ops it expects to see
//   Lowering - ops it converts and what they become
//   Output   - what the IR looks like after this pass
//
// For the order the passes are actually run in, see lib/Pipeline.cpp.
// ============================================================================

def CombineSpyreOps : Pass<"combine-spyre-ops", "mlir::ModuleOp"> {
    let summary = "Rewrite a spyreop intrinsic into the more specialized "
                  "spyreop intrinsic that computes the same thing.";
    let description = [{
        Input: KTIR after LowerSpyreOps — every scalar float computation the
        device has an intrinsic for is already a spyreop op, typically inside a
        linalg.generic body. Nothing else is read, and no other dialect is
        introduced: a pattern here has spyreop on both sides.

        Lowering:
        - spyreop.realdiv(c, x) where c is a constant 1.0 → spyreop.reciprocal(x)

        Output: no spyreop.realdiv with a constant-1.0 numerator remains.
        Everything else — a realdiv over two data values, a realdiv by any other
        constant, every other spyreop op — passes through unchanged. The
        numerator's arith.constant is left to die by ordinary dead-op
        elimination, which the greedy driver this pass runs does; nothing here
        erases it, so a numerator shared with another reader keeps working with
        no special case.

        WHY THE RECIPROCAL IS WORTH A PASS. A float immediate reaching a Spyre
        compute unit is not read back as it was written, so a division by a
        rounded 1.0 is not the division that was written. Nothing downstream
        refuses it — dbo-opt has no diagnostic for a float immediate operand, and
        the failure is numerical rather than structural — so the unary intrinsic,
        which takes the immediate out of the program entirely, is what protects
        the kernel. The one kernel that depends on it is
        `reduce/softmax_on_stick`, whose meta.py records the observation.

        WHAT MAY BE ADDED HERE, so the pass does not become a junk drawer. A
        pattern belongs here only if all three hold.

        1. Source and target are both spyreop ops. Crossing a dialect boundary
           is a conversion and belongs in Conversion/; rewriting within upstream
           dialects belongs in NormalizeForDevice. Rule 1 draws the line by what
           a rewrite touches, not by what motivates it.
        2. The two spellings compute the same value, and the rewrite is a
           PREFERENCE rather than a repair — which spelling the device prefers,
           and why. A rewrite that changes the computed value is not a peephole.
        3. The preference names the downstream behaviour it is for, so a reader
           can check whether it still holds.

        The identified next tenant, not implemented: spyreop.reciprocal of
        spyreop.sqrt → spyreop.rsqrt. It is the only other pair of spyreop
        spellings of one computation the dialect has — spyreop defines no float
        add, multiply, subtract, negate or FMA, so the usual algebraic peepholes
        have no operands to match. Deliberately left out, because it would change
        the emitted program for every kernel computing 1/sqrt(x) and because it
        needs a dbo-opt new enough to register spyreop.rsqrt at all.

        PRECONDITION, and it is not this pass's to establish: the numerator must
        already be a constant VISIBLE FROM THE DIVIDE — an arith.constant in the
        same region, or hoisted above the enclosing linalg.generic. A splat
        constant passed as an `ins` OPERAND of the generic arrives in the body as
        a block argument, and a block argument is not a constant, so this pass
        declines it and the realdiv survives with its immediate. Converting that
        form would mean dropping an operand, its block argument and its indexing
        map from the generic, which is a rewrite of the generic rather than of a
        spyreop op and so fails rule 1 above.

        What establishes the form is FoldDataMovementGenerics, whose elementwise
        fusion folds the splat's producer into the body. That is measured, not
        assumed: after ConvertElementwiseToLinalg alone, `reduce/softmax_on_stick`
        has the block-argument form and this pass would decline it; after
        FoldDataMovementGenerics it has the hoisted scalar constant. So the
        ordering constraint below is load-bearing, and it is the reason this pass
        follows the whole physicalization half of the `spyrecode` stage rather
        than sitting next to the lowering that creates its input.

        Ordering: after LowerSpyreOps, which is what produces the spyreop ops
        this pass reads — anchored before it, there is nothing to match. And
        after FoldDataMovementGenerics, per the precondition above; LowerSpyreOps
        already follows it, so the two constraints do not compete.
    }];

    let dependentDialects = [
        "mlir::spyreop::SpyreOpDialect"
    ];
}

#endif // SPYREOP_TRANSFORMS_PASSES

#ifndef KTDP_TRANSFORMS_REWRITEDESCRIPTORLAYOUT_OPSUTILS_H
#define KTDP_TRANSFORMS_REWRITEDESCRIPTORLAYOUT_OPSUTILS_H


#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Operation.h"

namespace mlir::triton::ktdp {

// Predicates about a single op, shared by this pass's phases. A predicate lands
// here when more than one phase asks the same question of an op -- keeping one
// answer in one place is what stops the analysis and the rewrite drifting into
// disagreeing about the same operation.
//
// Two kinds live here, and the distinction is worth keeping in mind when adding
// a third. `isElementwiseForLayout` is about an op's own nature -- true or false
// independently of this pass. `mayConsumePhysicalAsLogical` is about what this
// pass is entitled to leave alone, so it is a statement about the rewrite's
// contract rather than about the op, and it is expected to change as the rewrite
// learns to emit more.

/// True when `op` is elementwise in the sense this pass's elementwise patterns
/// rely on: the result's element at a coordinate is a function of the operands'
/// elements at THAT SAME coordinate, so a physical relabelling of the operands
/// is also a physical relabelling of the result.
///
/// Four sites call this: three patterns -- ElementwisePropagation
/// (PhysicalTypeAnalysis), ElementwiseRequirement (RequirementAnalysis) and
/// RewriteElementwisePattern (ContractionSynthesis) -- so the analysis predicts a
/// type for exactly the ops the rewrite retypes; plus pendingElementwiseRetype
/// (ContractionSynthesis), which is not a pattern but a PREDICTOR of the rewrite,
/// telling a consumer to defer until a retype lands. It has to agree with the
/// rewrite for the same reason the analysis does: predicting a retype that never
/// comes makes its caller defer on a change that is not coming.
///
/// Before it, each site spelled the test as "one ranked-tensor result, and every
/// ranked-tensor operand agrees on a shape" -- a statement about the operand
/// list, not the op, satisfied vacuously by any op with a single tensor operand. What kept the reshape family, linalg.broadcast and
/// tt.expand_dims out was registration order in the two analysis pattern sets
/// and reachability for the rewrite, neither of which is a precondition: order
/// protects only the ops someone remembered to register first, and
/// RewriteElementwisePattern has no ordered list to sit in at all.
///
/// Two spellings of elementwise reach this pass and each gets upstream's own
/// predicate. A structured op: `linalg::isElementwise` (every loop parallel,
/// every indexing map the identity), which excludes contractions, reductions
/// and a broadcast-carrying generic by construction. A scalar op lifted to
/// tensors (arith.*, math.*): `hasElementwiseMappableTraits`.
inline bool isElementwiseForLayout(Operation *op) {
  // A LinalgOp is asked as a LinalgOp even when it also carries scalar traits:
  // its iteration space, not its operand list, is what decides.
  if (auto linalgOp = dyn_cast<linalg::LinalgOp>(op))
    return linalg::isElementwise(linalgOp);
  return OpTrait::hasElementwiseMappableTraits(op);
}

/// True when `op` is entitled to read a value this pass physicalized and still
/// describe it at the logical shape -- i.e. physical-in / logical-out is the
/// correct outcome for it, not a gap in the rewrite. The legality gate
/// (Phase 4) asks this as its last question, after
/// establishing that the op reads a physical value and was not itself retyped.
///
/// An op that is NOT on this list, and reaches the gate, needs teaching -- three
/// files, and all three are required:
///
///   PhysicalTypeAnalysis.cpp   a PhysicalPropagationPattern saying what
///                              physical type the op's result carries given its
///                              operand's. Register it BEFORE
///                              ElementwisePropagation, which would otherwise
///                              claim any op whose operand shapes agree.
///   RequirementAnalysis.cpp    the backward twin, if a store's layout
///                              requirement has to cross the op. Omit it and
///                              the two directions disagree about the same op.
///   ContractionSynthesis.cpp   the rewrite that actually retypes the result,
///                              and registration in populateContractionPatterns.
///
/// Teaching it here instead -- adding the op to this list -- is the wrong answer
/// unless physical-in / logical-out is genuinely correct for it. This list is an
/// exemption, not a suppression.
///
/// The two entries are not the same kind of thing, and it is worth not pretending
/// otherwise:
///
///   linalg.reduce         A GENUINE exemption. Phase 2 rewrites a reduce in
///                         place, absorbing the stick dims into `dimensions`
///                         (e.g. ins tensor<2x64x64xf32> dimensions = [0, 2]),
///                         so it really does read the physical value and really
///                         does collapse the reduced axis. Physical-in /
///                         logical-out is the correct answer for it.
///
///   tensor.extract_slice  SCAFFOLDING, not an exemption. Phase 2 emits these
///                         itself, inside the stick loops it builds -- so the
///                         gate is flagging the pass's own output. The op is
///                         listed because the gate cannot yet tell "Phase 2
///                         minted this" from "the input contained this". The
///                         better fix is for the gate to know which values Phase
///                         2 created, at which point this entry deletes.
///
/// Note ktdp.store is deliberately NOT here, though it also consumes a physical
/// value and produces no stick-shaped result: a sink has no results at all, so
/// the gate's own check 2 excuses it as "nothing left to convert". That is a
/// statement about the op's shape rather than its name, so it does not need one.
///
/// The set is measured, not assumed. The gate's walk was instrumented over every
/// rewrite-descriptor-layout lit fixture: these three, plus the elementwise ops
/// the gate's own earlier check admits by their result being physical too, are
/// the complete set of consumers of a physicalized value.
///
/// Named linalg.matmul, linalg.batch_matmul and linalg.transpose are absent ON
/// PURPOSE, and the contrast with linalg.reduce above is the thing to understand
/// here. Phase 2 wraps a contraction in a stick loop and feeds it the extracted
/// slices, so its operands are `tensor.extract_slice` results -- which are NOT in
/// physicalValues. The gate's check 1 ("does it read anything physical?") excuses
/// it before this question is ever asked. A reduce, rewritten in place, does read
/// the physical value and so does reach it. Same logical-shaped output, different
/// answer, because the operands differ.
///
/// So adding a contraction here would answer a question nobody asks, and would
/// mask a real gap once a later step puts one on a physical operand directly.
///
/// FLAG: this goes stale at the step that emits the reduce as a linalg.generic
/// with reduction iterators. `isa<linalg::ReduceOp>` stops matching, so a
/// stick-wide generic reduce would be wrongly treated as a gap. The replacement
/// is a test on the op's iterator types, not on its class.
inline bool mayConsumePhysicalAsLogical(Operation *op) {
  return isa<tensor::ExtractSliceOp, linalg::ReduceOp>(op);
}

} // namespace mlir::triton::ktdp

#endif // KTDP_TRANSFORMS_REWRITEDESCRIPTORLAYOUT_OPSUTILS_H

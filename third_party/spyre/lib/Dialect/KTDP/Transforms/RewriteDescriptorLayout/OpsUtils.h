#ifndef KTDP_TRANSFORMS_REWRITEDESCRIPTORLAYOUT_OPSUTILS_H
#define KTDP_TRANSFORMS_REWRITEDESCRIPTORLAYOUT_OPSUTILS_H

#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Operation.h"

namespace mlir::triton::ktdp {

// Predicates about a single op, shared by this pass's phases. A predicate lands
// here when more than one phase asks the same question of an op -- keeping one
// answer in one place is what stops the analysis and the rewrite drifting into
// disagreeing about the same operation.

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

} // namespace mlir::triton::ktdp

#endif // KTDP_TRANSFORMS_REWRITEDESCRIPTORLAYOUT_OPSUTILS_H

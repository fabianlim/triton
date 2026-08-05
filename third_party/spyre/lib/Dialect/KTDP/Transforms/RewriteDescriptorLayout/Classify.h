#ifndef KTDP_TRANSFORMS_REWRITEDESCRIPTORLAYOUT_CLASSIFY_H
#define KTDP_TRANSFORMS_REWRITEDESCRIPTORLAYOUT_CLASSIFY_H

#include "RewriteDescriptorLayout/PermutationUtils.h"
#include "RewriteDescriptorLayout/Types.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir::triton::ktdp {

/// Assign a role to each physical dim of an operand.
///   >= 0  : parallel dim, maps to output axis [value]
///   -1    : reduction dim
void buildDimRoles(const OperandCoords &coords,
                   llvm::ArrayRef<int64_t> canonicalAxes,
                   llvm::SmallVectorImpl<int64_t> &roles);

/// Classify one operand's physical dims into OperandPlan fields.
OperandPlan classify(Value val, const OperandCoords &coords,
                     llvm::ArrayRef<int64_t> dimRoles);

/// Build a plan for a scratchpad operand (no marker, logical shape).
OperandPlan classifyScratchpad(Value val, const SourceOperandSpec &opSpec);

/// Resolve per-operand transpose + extents and perform cross-operand fix-up.
void resolveAndReconcile(llvm::SmallVectorImpl<OperandPlan> &plans,
                         const SourceOpSpec &spec);

/// Per-dim op-tile slice extent.
int64_t opSliceExtent(const OperandPlan &plan, int p);

} // namespace mlir::triton::ktdp

#endif // KTDP_TRANSFORMS_REWRITEDESCRIPTORLAYOUT_CLASSIFY_H

#ifndef KTDP_TRANSFORMS_REWRITEDESCRIPTORLAYOUT_CONTRACTIONSYNTHESIS_H
#define KTDP_TRANSFORMS_REWRITEDESCRIPTORLAYOUT_CONTRACTIONSYNTHESIS_H

#include "RewriteDescriptorLayout/Types.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir::triton::ktdp {

/// Phase 2: walk module and dispatch contractions until stable.
/// Returns failure if any dispatch emits an error diagnostic.
LogicalResult synthesizeContractions(mlir::ModuleOp module, PassContext &ctx);

} // namespace mlir::triton::ktdp

#endif // KTDP_TRANSFORMS_REWRITEDESCRIPTORLAYOUT_CONTRACTIONSYNTHESIS_H

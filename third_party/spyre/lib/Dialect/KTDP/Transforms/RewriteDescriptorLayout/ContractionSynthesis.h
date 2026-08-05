#ifndef KTDP_TRANSFORMS_REWRITEDESCRIPTORLAYOUT_CONTRACTIONSYNTHESIS_H
#define KTDP_TRANSFORMS_REWRITEDESCRIPTORLAYOUT_CONTRACTIONSYNTHESIS_H

#include "RewriteDescriptorLayout/Types.h"
#include "mlir/IR/PatternMatch.h"

namespace mlir::triton::ktdp {

/// Phase 2: populate greedy rewrite patterns for contraction synthesis.
/// Source patterns (matmul, reduce) run at benefit 2; sink patterns (store)
/// run at benefit 1.
void populateContractionPatterns(mlir::RewritePatternSet &patterns,
                                 const PassContext &ctx);

} // namespace mlir::triton::ktdp

#endif // KTDP_TRANSFORMS_REWRITEDESCRIPTORLAYOUT_CONTRACTIONSYNTHESIS_H

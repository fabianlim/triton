#ifndef KTDP_TRANSFORMS_REWRITEDESCRIPTORLAYOUT_TYPES_H
#define KTDP_TRANSFORMS_REWRITEDESCRIPTORLAYOUT_TYPES_H

#include "mlir/IR/Value.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

namespace mlir::triton::ktdp {

struct PassContext {
  llvm::DenseMap<mlir::Value, triton::SpyreTensorLayoutOp> &physMemViewToMarker;
  llvm::DenseMap<mlir::Value, llvm::SmallVector<int64_t>> &physicalLoadToTransposePerm;
};

} // namespace mlir::triton::ktdp

#endif // KTDP_TRANSFORMS_REWRITEDESCRIPTORLAYOUT_TYPES_H

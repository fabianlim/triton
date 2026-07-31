#ifndef KTDP_TRANSFORMS_REWRITEDESCRIPTORLAYOUT_TYPES_H
#define KTDP_TRANSFORMS_REWRITEDESCRIPTORLAYOUT_TYPES_H

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

namespace mlir::triton::ktdp {

struct PassContext {
  const llvm::DenseMap<mlir::Value, triton::SpyreTensorLayoutOp> &physMemViewToMarker;
  const llvm::DenseMap<mlir::Value, llvm::SmallVector<int64_t>> &physicalLoadToTransposePerm;
  /// Set by patterns to indicate a fatal error that should abort the pass.
  mutable bool hadError = false;
};

/// Per-operand coord-map info read from a still-live marker.
struct OperandCoords {
  llvm::ArrayRef<int64_t> src; // phys_src
  llvm::ArrayRef<int64_t> op;  // phys_op  (0=Identity,1=FloorDiv,2=Mod)
  llvm::ArrayRef<int64_t> arg; // phys_arg
  unsigned logicalRank;
  llvm::ArrayRef<int64_t> physBlock;

  static OperandCoords fromMarker(triton::SpyreTensorLayoutOp marker,
                                  unsigned logRank,
                                  llvm::ArrayRef<int64_t> physBlock) {
    return {marker.getPhysSrc(), marker.getPhysOp(), marker.getPhysArg(),
            logRank, physBlock};
  }
};

/// How a single physical dim is sliced when extracting the per-iteration tile.
enum class SliceKind {
  StickIndex,       // floor/loopDims dim: offset = this operand's own loop IV,
                    // size = 1 (selects one stick along a stick-index dim).
  StickifiedBlock,  // opInnerDim spanning >1 stick (B's K-flat): offset =
                    // reduction IV * stickSize, size = stickSize (one stick).
  WholeBlock,       // lane / opSlice / single-stick opInnerDim: offset = 0,
                    // size = physBlock[p] (taken whole as part of the 2D tile).
};

/// Pure output of classify(): per-physical-dim role assignments.
struct ClassifiedDims {
  int                lane;        // innermost phys dim = rank-1
  int64_t            stickSize;   // stick/lane width = physBlock[lane]
  llvm::SmallVector<int>   floorDims;   // parallel stick-index dims
  llvm::SmallVector<int>   reduceDims;  // all -1 dims in right-to-left order
  int                opInnerDim;  // rightmost reduceDim; -1 if none
  llvm::SmallVector<int>   loopDims;    // reduceDims minus opInnerDim
  llvm::SmallVector<int>   opTileDims;  // residual >= 0 non-floor dims
  llvm::SmallVector<SliceKind> sliceKind; // per-phys-dim slice behavior
};

/// One operand's full plan: classification + resolution results.
struct OperandPlan {
  Value               value;      // SSA tensor (physical on memory side)
  OperandCoords       coords;     // coord map + shape
  llvm::SmallVector<int64_t> dimRoles;  // per-phys-dim role (>= 0 | -1)
  ClassifiedDims      dims;       // output of classify()

  // Resolved fields — filled by resolveAndReconcile() after classify().
  llvm::SmallVector<int64_t> transposePerm;
  llvm::SmallVector<int64_t> opExtents;
};

/// Per-operand descriptor for a source contraction op.
struct SourceOperandSpec {
  llvm::SmallVector<int64_t> canonicalAxes;
};

/// Descriptor for one source contraction op (e.g. linalg.matmul).
struct SourceOpSpec {
  llvm::SmallVector<SourceOperandSpec> operands;
  unsigned logicalRank;
  llvm::function_ref<Value(OpBuilder &, Location, llvm::ArrayRef<Value>,
                           Value, RankedTensorType)>
      emitOp;
};

} // namespace mlir::triton::ktdp

#endif // KTDP_TRANSFORMS_REWRITEDESCRIPTORLAYOUT_TYPES_H

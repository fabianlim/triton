#include "RewriteDescriptorLayout/Classify.h"

#include "mlir/IR/BuiltinTypes.h"

#include <algorithm>

namespace mlir::triton::ktdp {

void buildDimRoles(const OperandCoords &coords,
                   llvm::ArrayRef<int64_t> canonicalAxes,
                   llvm::SmallVectorImpl<int64_t> &roles) {
  int n = (int)coords.src.size();
  roles.resize(n);
  for (int p = 0; p < n; ++p) {
    int64_t logDim = coords.src[p];
    roles[p] = (logDim < (int64_t)canonicalAxes.size())
                   ? canonicalAxes[logDim]
                   : -1;
  }
}

OperandPlan classify(Value val, const OperandCoords &coords,
                     llvm::ArrayRef<int64_t> dimRoles) {
  int rank = (int)dimRoles.size();
  OperandPlan plan;
  plan.value     = val;
  plan.coords    = coords;
  plan.dimRoles  = llvm::SmallVector<int64_t>(dimRoles.begin(), dimRoles.end());
  plan.lane      = rank - 1;
  plan.stickSize = coords.physBlock[rank - 1];
  plan.opInnerDim = -1;

  for (int p = rank - 1; p >= 0; --p) {
    int64_t role = dimRoles[p];
    bool isFloor = (role >= 0 &&
                    static_cast<CoordOp>(coords.op[p]) == CoordOp::FloorDiv);
    if (role == -1) {
      plan.reduceDims.push_back(p);
      if (plan.opInnerDim == -1) {
        plan.opInnerDim = p;
        plan.opTileDims.push_back(p);
      } else {
        plan.loopDims.push_back(p);
      }
    } else if (isFloor) {
      plan.floorDims.push_back(p);
    } else {
      plan.opTileDims.push_back(p);
    }
  }
  std::reverse(plan.floorDims.begin(), plan.floorDims.end());
  std::reverse(plan.loopDims.begin(), plan.loopDims.end());
  std::reverse(plan.opTileDims.begin(), plan.opTileDims.end());

  plan.sliceKind.assign(rank, SliceKind::WholeBlock);
  auto markList = [&](llvm::ArrayRef<int> dims) {
    for (int p : dims)
      plan.sliceKind[p] = SliceKind::StickIndex;
  };
  markList(plan.floorDims);
  markList(plan.loopDims);
  if (plan.opInnerDim != -1 &&
      coords.physBlock[plan.opInnerDim] > plan.stickSize)
    plan.sliceKind[plan.opInnerDim] = SliceKind::StickifiedBlock;
  return plan;
}

OperandPlan classifyScratchpad(Value val, const SourceOperandSpec &opSpec) {
  auto tensorTy = mlir::cast<RankedTensorType>(val.getType());
  int rank = (int)tensorTy.getRank();
  OperandPlan plan;
  plan.value = val;
  plan.physBlockStorage.assign(tensorTy.getShape().begin(),
                               tensorTy.getShape().end());
  plan.coords.src        = {};
  plan.coords.op         = {};
  plan.coords.arg        = {};
  plan.coords.logicalRank = (unsigned)rank;
  plan.coords.physBlock   = plan.physBlockStorage;

  plan.lane       = rank - 1;
  plan.stickSize  = tensorTy.getDimSize(rank - 1);
  plan.opInnerDim = -1;
  for (int p = 0; p < rank; ++p) {
    plan.opTileDims.push_back(p);
    plan.dimRoles.push_back(opSpec.canonicalAxes[p]);
  }
  plan.sliceKind.assign(rank, SliceKind::WholeBlock);

  plan.transposePerm      = {};
  for (int p : plan.opTileDims)
    plan.opExtents.push_back(tensorTy.getDimSize(p));
  return plan;
}

int64_t opSliceExtent(const OperandPlan &plan, int p) {
  return plan.sliceKind[p] == SliceKind::StickifiedBlock
             ? plan.stickSize
             : plan.coords.physBlock[p];
}

void resolveAndReconcile(llvm::SmallVectorImpl<OperandPlan> &plans,
                         const SourceOpSpec &spec) {
  // Step 1 — resolve per-operand fields.
  for (unsigned i = 0; i < plans.size(); ++i) {
    OperandPlan &plan = plans[i];
    const SourceOperandSpec &opSpec = spec.operands[i];

    plan.transposePerm = computeTransposePerm(
        plan.opTileDims, plan.dimRoles, opSpec.canonicalAxes);

    plan.opExtents.clear();
    for (int p : plan.opTileDims)
      plan.opExtents.push_back(opSliceExtent(plan, p));
  }

  // Step 2 — StickifiedBlock demotion.
  bool anyLoop = false;
  for (auto &p : plans)
    if (!p.loopDims.empty()) { anyLoop = true; break; }
  if (!anyLoop) {
    for (auto &plan : plans)
      for (auto &sk : plan.sliceKind)
        if (sk == SliceKind::StickifiedBlock)
          sk = SliceKind::WholeBlock;
    // Re-derive extents now that sliceKind has changed.
    for (unsigned i = 0; i < plans.size(); ++i) {
      OperandPlan &plan = plans[i];
      plan.opExtents.clear();
      for (int p : plan.opTileDims)
        plan.opExtents.push_back(opSliceExtent(plan, p));
    }
  }
}

} // namespace mlir::triton::ktdp

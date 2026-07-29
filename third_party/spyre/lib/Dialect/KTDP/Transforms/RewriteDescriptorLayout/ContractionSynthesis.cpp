#include "RewriteDescriptorLayout/ContractionSynthesis.h"
#include "RewriteDescriptorLayout/Classify.h"
#include "RewriteDescriptorLayout/PermutationUtils.h"
#include "RewriteDescriptorLayout/Types.h"

#include "Ktdp/KtdpOps.hpp"
#include "Ktdp/KtdpTypes.hpp"
#include "triton/Dialect/Triton/IR/Dialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/ErrorHandling.h"

#include <algorithm>

namespace mlir::triton::ktdp {

// True iff op is a single-result elementwise op with exactly one
// RankedTensor operand.
static bool isSingleTensorElementwiseOp(Operation *op) {
  if (op->getNumResults() != 1 || op->getNumOperands() == 0)
    return false;
  int tensorOps = 0;
  for (auto operand : op->getOperands())
    if (isa<RankedTensorType>(operand.getType()))
      ++tensorOps;
  return tensorOps == 1;
}

// Walk backward from `val` through single-tensor elementwise ops to the
// ktdp.load that produced it. Returns null if not found.
static mlir::ktdp::LoadOp walkToLoad(Value val) {
  Value v = val;
  while (true) {
    auto *defOp = v.getDefiningOp();
    if (!defOp)
      return mlir::ktdp::LoadOp{};
    if (auto ld = dyn_cast<mlir::ktdp::LoadOp>(defOp))
      return ld;
    if (!isSingleTensorElementwiseOp(defOp))
      return mlir::ktdp::LoadOp{};
    for (auto operand : defOp->getOperands())
      if (isa<RankedTensorType>(operand.getType())) { v = operand; break; }
  }
}

// Walk back from an operand through the elementwise chain to the
// ktdp.load, then look up the physical memView -> marker map.
static triton::SpyreTensorLayoutOp
findMarkerForOperand(Value operand, PassContext &ctx) {
  auto ld = walkToLoad(operand);
  if (!ld)
    return {};
  auto tile = dyn_cast<mlir::ktdp::ConstructAccessTilesOp>(
      ld.getAccessTile().getDefiningOp());
  if (!tile)
    return {};
  auto it = ctx.physMemViewToMarker.find(tile.getBase());
  return it != ctx.physMemViewToMarker.end() ? it->second
                                             : triton::SpyreTensorLayoutOp{};
}

// Emit a stick loop (scf.for) or inline for trip <= 1.
static Value emitStickLoop(OpBuilder &b, Location loc, int64_t tripCount,
                           Value acc,
                           llvm::function_ref<Value(OpBuilder &, Value, Value)> body) {
  if (tripCount <= 1) {
    Value s0 = arith::ConstantIndexOp::create(b, loc, 0);
    return body(b, s0, acc);
  }
  Value c0 = arith::ConstantIndexOp::create(b, loc, 0);
  Value c1 = arith::ConstantIndexOp::create(b, loc, 1);
  Value ub = arith::ConstantIndexOp::create(b, loc, tripCount);
  auto forOp = scf::ForOp::create(b, loc, c0, ub, c1, ValueRange{acc});
  OpBuilder ib = OpBuilder::atBlockBegin(forOp.getBody());
  Value stepped =
      body(ib, forOp.getInductionVar(), forOp.getRegionIterArgs()[0]);
  scf::YieldOp::create(ib, loc, ValueRange{stepped});
  b.setInsertionPointAfter(forOp);
  return forOp.getResult(0);
}

// Extract an op-tile stick slice from `plan`.
static Value extractOpSlice(OpBuilder &b, Location loc,
                            const OperandPlan &plan,
                            RankedTensorType resultTy, Value stickIV,
                            Value parallelIV = nullptr) {
  auto idx = [&](int64_t v) -> OpFoldResult { return b.getIndexAttr(v); };
  llvm::ArrayRef<int64_t> physBlock = plan.coords.physBlock;
  int rank = (int)physBlock.size();
  llvm::SmallVector<OpFoldResult> offsets(rank), sizes(rank), strides(rank, idx(1));
  for (int p = 0; p < rank; ++p) {
    switch (plan.sliceKind[p]) {
    case SliceKind::StickIndex: {
      Value selectedIV = (plan.dimRoles[p] >= 0 && parallelIV) ? parallelIV : stickIV;
      Value iv = (physBlock[p] > 1) ? selectedIV : Value{};
      if (!iv) {
        offsets[p] = idx(0);
      } else if (iv.getType().isIndex()) {
        offsets[p] = iv;
      } else {
        offsets[p] = arith::IndexCastOp::create(b, loc,
                         b.getIndexType(), iv).getResult();
      }
      sizes[p] = idx(1);
      break;
    }
    case SliceKind::StickifiedBlock: {
      Value sIdx = stickIV.getType().isIndex()
                       ? stickIV
                       : arith::IndexCastOp::create(b, loc,
                             b.getIndexType(), stickIV).getResult();
      Value stickSz = arith::ConstantIndexOp::create(b, loc, plan.stickSize);
      offsets[p] = arith::MulIOp::create(b, loc, sIdx, stickSz).getResult();
      sizes[p]   = idx(plan.stickSize);
      break;
    }
    case SliceKind::WholeBlock:
      offsets[p] = idx(0);
      sizes[p]   = idx(physBlock[p]);
      break;
    }
  }
  return tensor::ExtractSliceOp::create(
      b, loc, resultTy, plan.value, offsets, sizes, strides);
}

// Source stage emission: extract slices, optional transpose, call emitOp.
template <typename OpT>
static LogicalResult emitSourceStage(
    OpT op,
    llvm::function_ref<Value(OpBuilder &, Location, llvm::ArrayRef<Value>, Value,
                             RankedTensorType)>
        emitOp,
    llvm::ArrayRef<OperandPlan> plans) {
  OpBuilder b(op);
  Location loc = op.getLoc();

  Value cVal = op.getDpsInits()[0];
  auto accElemTy = cast<RankedTensorType>(cVal.getType()).getElementType();

  // Per-operand op-tile slice types.
  llvm::SmallVector<RankedTensorType> sliceTys;
  for (unsigned i = 0; i < plans.size(); ++i) {
    const OperandPlan &plan = plans[i];
    auto elemTy = cast<RankedTensorType>(plan.value.getType()).getElementType();
    sliceTys.push_back(RankedTensorType::get(plan.opExtents, elemTy));
  }

  // Derive acc shape from the union of all (outputAxis, extent) pairs.
  int64_t maxAxis = -1;
  for (auto &plan : plans)
    for (unsigned j = 0; j < plan.opTileDims.size(); ++j) {
      int p = plan.opTileDims[j];
      int64_t role = plan.dimRoles[p];
      if (role >= 0 && role > maxAxis)
        maxAxis = role;
    }
  llvm::SmallVector<int64_t> accDims(maxAxis + 1, 0);
  for (auto &plan : plans)
    for (unsigned j = 0; j < plan.opTileDims.size(); ++j) {
      int p = plan.opTileDims[j];
      int64_t role = plan.dimRoles[p];
      if (role >= 0)
        accDims[role] = plan.opExtents[j];
    }
  auto accTy = RankedTensorType::get(accDims, accElemTy);

  // Transpose helper.
  auto emitTranspose = [&](Value src, llvm::ArrayRef<int64_t> perm) -> Value {
    auto srcTy = cast<RankedTensorType>(src.getType());
    auto mlirPerm = invertPerm(perm);
    llvm::SmallVector<int64_t> outShape(mlirPerm.size());
    for (unsigned i = 0; i < mlirPerm.size(); ++i)
      outShape[i] = srcTy.getDimSize(mlirPerm[i]);
    auto outTy = RankedTensorType::get(outShape, srcTy.getElementType());
    Value empty = tensor::EmptyOp::create(b, loc, outTy.getShape(),
                                          srcTy.getElementType());
    return linalg::TransposeOp::create(b, loc, src, empty,
        b.getDenseI64ArrayAttr(mlirPerm)).getResult()[0];
  };

  // Determine the stick loop trip count (stickFactor).
  int64_t stickFactor = 1;
  for (auto &plan : plans) {
    for (int p : plan.loopDims) {
      if (static_cast<CoordOp>(plan.coords.op[p]) != CoordOp::FloorDiv)
        continue;
      int64_t logDim = plan.dimRoles[p];
      if (logDim >= 0)
        continue;
      int64_t f;
      if (plan.sliceKind[p] == SliceKind::StickifiedBlock)
        f = plan.coords.physBlock[p] / plan.stickSize;
      else
        f = plan.coords.physBlock[p];
      if (f <= 1)
        continue;
      if (stickFactor != 1 && stickFactor != f)
        llvm_unreachable("emitSourceStage: plans disagree on stickFactor");
      stickFactor = f;
    }
  }

  // Reduction-only path (no parallel scatter for Stage 4).
  Value stickIV;
  Value result = emitStickLoop(b, loc, stickFactor, cVal,
      [&](OpBuilder &bb, Value s, Value acc) {
    stickIV = s;
    OpBuilder saved = b;
    b = bb;
    llvm::SmallVector<Value> slices;
    for (unsigned i = 0; i < plans.size(); ++i) {
      Value slicePhys = extractOpSlice(b, loc, plans[i], sliceTys[i], stickIV);
      slices.push_back(!plans[i].transposePerm.empty()
                           ? emitTranspose(slicePhys, plans[i].transposePerm)
                           : slicePhys);
    }
    Value r = emitOp(b, loc, slices, acc, accTy);
    b = saved;
    return r;
  });

  op.getResult(0).replaceAllUsesWith(result);
  op.erase();
  return success();
}

// Classify one operand and populate plans[i].
template <typename OpT>
static LogicalResult dispatchSource(OpT op, const SourceOpSpec &spec,
                                    PassContext &ctx) {
  unsigned nOps = spec.operands.size();
  llvm::SmallVector<OperandPlan, 2> plans(nOps);

  for (unsigned i = 0; i < nOps; ++i) {
    Value operand = op.getInputs()[i];
    auto ld = walkToLoad(operand);

    if (ld) {
      auto marker = findMarkerForOperand(operand, ctx);
      if (!marker) {
        auto tensorTy = dyn_cast<RankedTensorType>(operand.getType());
        if (!tensorTy ||
            tensorTy.getRank() != (int64_t)spec.operands[i].canonicalAxes.size())
          return op.emitError(
              "spyre_tensor_layout: physical operand load has no layout marker");
        plans[i] = classifyScratchpad(operand, spec.operands[i]);
        continue;
      }
      auto physShape = cast<RankedTensorType>(operand.getType()).getShape();
      OperandCoords coords = OperandCoords::fromMarker(marker, spec.logicalRank,
                                                       physShape);
      // Compose erased transpose perm into canonicalAxes.
      llvm::SmallVector<int64_t> effectiveCanonicalAxes = spec.operands[i].canonicalAxes;
      {
        auto it = ctx.physicalLoadToTransposePerm.find(ld.getResult());
        if (it != ctx.physicalLoadToTransposePerm.end()) {
          const auto &tau = it->second;
          assert(tau.size() == effectiveCanonicalAxes.size() &&
                 "transpose perm size must match canonicalAxes size");
          llvm::SmallVector<int64_t> reordered(effectiveCanonicalAxes.size());
          for (unsigned j = 0; j < tau.size(); ++j)
            reordered[j] = effectiveCanonicalAxes[tau[j]];
          effectiveCanonicalAxes = std::move(reordered);
        }
      }
      llvm::SmallVector<int64_t> dimRoles;
      buildDimRoles(coords, effectiveCanonicalAxes, dimRoles);
      plans[i] = classify(operand, coords, dimRoles);
    } else {
      auto tensorTy = dyn_cast<RankedTensorType>(operand.getType());
      if (!tensorTy ||
          tensorTy.getRank() != (int64_t)spec.operands[i].canonicalAxes.size())
        return op.emitError(
            "spyre_tensor_layout: source op operand is neither a physical "
            "load nor a logical (scratchpad) tensor of the expected rank");
      plans[i] = classifyScratchpad(operand, spec.operands[i]);
    }
  }

  resolveAndReconcile(plans, spec);

  // R6 check: scratchpad on multi-stick reduction axis.
  for (unsigned i = 0; i < nOps; ++i) {
    if (plans[i].coords.src.empty())
      continue;
    bool multiStickReduction = false;
    for (int p : plans[i].loopDims)
      if (plans[i].coords.physBlock[p] > 1) { multiStickReduction = true; break; }
    if (!multiStickReduction)
      continue;
    for (unsigned j = 0; j < nOps; ++j) {
      if (i == j) continue;
      if (plans[j].coords.src.empty())
        return op.emitError(
            "spyre_tensor_layout: operands share a stickified contraction "
            "axis but not all are annotated — any two operands sharing a "
            "stickified contraction axis must both carry a "
            "tt.spyre_tensor_layout marker with the same stick size on that "
            "axis (R6)");
    }
  }

  return emitSourceStage(op, spec.emitOp, plans);
}

// linalg.matmul instantiation.
static LogicalResult dispatchMatmul(linalg::MatmulOp mm, PassContext &ctx) {
  SourceOpSpec spec;
  spec.operands = {SourceOperandSpec{{0, -1}},   // A=(m,k)
                   SourceOperandSpec{{-1, 1}}};  // B=(k,n)
  spec.logicalRank = 2;
  spec.emitOp = [](OpBuilder &b, Location loc,
                   llvm::ArrayRef<Value> slices, Value acc,
                   RankedTensorType accTy) -> Value {
    return linalg::MatmulOp::create(b, loc, accTy,
        ValueRange{slices[0], slices[1]}, ValueRange{acc}).getResult(0);
  };
  return dispatchSource(mm, spec, ctx);
}

// Return true and dispatch if `op` needs Phase 2 processing, false if not.
static LogicalResult dispatchOne(Operation *op, bool &changed, PassContext &ctx) {
  auto sourceNeedsDispatch = [&](linalg::LinalgOp linalgOp, unsigned logicalRank) {
    return llvm::any_of(linalgOp.getDpsInputOperands(), [&](OpOperand *operand) {
      auto t = dyn_cast<RankedTensorType>(operand->get().getType());
      if (!t || t.getRank() <= (int)logicalRank)
        return false;
      return static_cast<bool>(findMarkerForOperand(operand->get(), ctx));
    });
  };

  if (auto mm = dyn_cast<linalg::MatmulOp>(op))
    return sourceNeedsDispatch(mm, 2) ? (changed = true, dispatchMatmul(mm, ctx)) : success();

  return success();
}

bool synthesizeContractions(mlir::ModuleOp module, PassContext &ctx) {
  bool anyChanged = false;
  bool changed = true;
  while (changed) {
    changed = false;
    llvm::SmallVector<Operation *> candidates;
    module.walk([&](Operation *op) {
      if (isa<linalg::MatmulOp>(op))
        candidates.push_back(op);
    });
    for (auto *op : candidates) {
      if (failed(dispatchOne(op, changed, ctx)))
        return anyChanged; // error already emitted
    }
    if (changed)
      anyChanged = true;
  }
  return anyChanged;
}

} // namespace mlir::triton::ktdp

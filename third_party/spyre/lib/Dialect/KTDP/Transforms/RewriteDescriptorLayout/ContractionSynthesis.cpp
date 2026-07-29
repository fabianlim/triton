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
#include <numeric>

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

// Emit linalg.transpose with the given permutation (input->output form).
// linalg.transpose uses "output<-input" form, so we invert here.
static Value emitTranspose(OpBuilder &b, Location loc, Value src,
                           llvm::ArrayRef<int64_t> perm) {
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

  // Transpose helper (delegates to free function).
  auto doTranspose = [&](Value src, llvm::ArrayRef<int64_t> perm) -> Value {
    return emitTranspose(b, loc, src, perm);
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
                           ? doTranspose(slicePhys, plans[i].transposePerm)
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

// Walk the forward use chain from value through elementwise ops to a
// ktdp.store, then look up the store's access tile base in physMemViewToMarker.
static triton::SpyreTensorLayoutOp findMarkerForStore(Value value,
                                                       PassContext &ctx) {
  llvm::SmallVector<Value> worklist = {value};
  while (!worklist.empty()) {
    Value v = worklist.pop_back_val();
    for (auto *user : v.getUsers()) {
      if (auto st = dyn_cast<mlir::ktdp::StoreOp>(user)) {
        auto tile = dyn_cast<mlir::ktdp::ConstructAccessTilesOp>(
            st.getAccessTile().getDefiningOp());
        if (!tile)
          continue;
        auto it = ctx.physMemViewToMarker.find(tile.getBase());
        if (it != ctx.physMemViewToMarker.end())
          return it->second;
      }
      if (!isSingleTensorElementwiseOp(user))
        continue;
      worklist.push_back(user->getResult(0));
    }
  }
  return {};
}

// Sink stage: scatter a logical data tile into the physical D tensor shape.
static LogicalResult emitSinkStage(mlir::ktdp::StoreOp st,
                                   const OperandPlan &dPlan) {
  Value inputTile = st.getDataTile();
  OpBuilder b(st);
  Location loc = st.getLoc();

  Type elemTy = cast<RankedTensorType>(inputTile.getType()).getElementType();

  llvm::ArrayRef<int64_t> physBlock = dPlan.coords.physBlock;
  int physRank = (int)physBlock.size();
  int64_t stickSize = physBlock[dPlan.lane];

  if (dPlan.floorDims.empty())
    return st.emitError(
        "spyre_tensor_layout: store sink stage requires at least one "
        "parallel floor dim in the output layout");
  if (!dPlan.loopDims.empty())
    return st.emitError(
        "spyre_tensor_layout: store sink stage: unexpected reduction dim");

  unsigned logRank = dPlan.coords.logicalRank;
  llvm::SmallVector<int64_t> sinkPerm;
  {
    llvm::SmallVector<int64_t> canonicalAxesD(logRank);
    std::iota(canonicalAxesD.begin(), canonicalAxesD.end(), 0);
    auto fwdPerm = computeTransposePerm(dPlan.opTileDims, dPlan.dimRoles,
                                        canonicalAxesD);
    if (!fwdPerm.empty()) {
      auto inv = invertPerm(fwdPerm);
      bool isIdentity = true;
      for (unsigned d = 0; d < logRank; ++d)
        if (inv[d] != (int64_t)d) { isIdentity = false; break; }
      if (!isIdentity)
        sinkPerm = std::move(inv);
    }
  }

  auto logDimToPos = [&](int64_t d) -> unsigned {
    return sinkPerm.empty() ? (unsigned)d : (unsigned)sinkPerm[d];
  };

  if (!sinkPerm.empty())
    inputTile = emitTranspose(b, loc, inputTile, sinkPerm);

  auto idx = [&](int64_t v) -> OpFoldResult { return b.getIndexAttr(v); };

  llvm::SmallVector<int64_t> sinkShape(physBlock.begin(), physBlock.end());
  Value physicalSink = tensor::EmptyOp::create(b, loc, sinkShape, elemTy);

  llvm::SmallVector<OpFoldResult> inputOffsetsBase(logRank, idx(0));
  llvm::SmallVector<OpFoldResult> inputSizesBase(logRank);
  llvm::SmallVector<OpFoldResult> inputStrides(logRank, idx(1));
  for (int p : dPlan.opTileDims) {
    int64_t logDim = dPlan.dimRoles[p];
    if (logDim >= 0 && (unsigned)logDim < logRank)
      inputSizesBase[logDimToPos(logDim)] = idx(physBlock[p]);
  }
  for (int p : dPlan.floorDims) {
    int64_t logDim = dPlan.dimRoles[p];
    if (logDim >= 0 && (unsigned)logDim < logRank)
      inputSizesBase[logDimToPos(logDim)] = idx(stickSize);
  }

  llvm::SmallVector<OpFoldResult> sinkOffsetsBase(physRank, idx(0));
  llvm::SmallVector<OpFoldResult> sinkSizes(physRank);
  llvm::SmallVector<OpFoldResult> sinkStrides(physRank, idx(1));
  for (int p = 0; p < physRank; ++p)
    sinkSizes[p] = llvm::is_contained(dPlan.floorDims, p)
                       ? idx(1) : idx(physBlock[p]);

  Value acc = physicalSink;
  for (int p : dPlan.floorDims) {
    int64_t logDim = dPlan.dimRoles[p];
    if (logDim < 0 || (unsigned)logDim >= logRank) continue;

    unsigned tileDim = logDimToPos(logDim);
    int64_t tripCount = physBlock[p];
    Value stickSizeVal = arith::ConstantIndexOp::create(b, loc, stickSize);

    llvm::SmallVector<int64_t> slShape(logRank);
    for (int p2 : dPlan.opTileDims) {
      int64_t ld = dPlan.dimRoles[p2];
      if (ld >= 0 && (unsigned)ld < logRank)
        slShape[logDimToPos(ld)] = physBlock[p2];
    }
    slShape[tileDim] = stickSize;
    auto slTy = RankedTensorType::get(slShape, elemTy);

    acc = emitStickLoop(b, loc, tripCount, acc,
        [&](OpBuilder &bb, Value s, Value sinkAccumulator) -> Value {
          llvm::SmallVector<OpFoldResult> inOff = inputOffsetsBase;
          inOff[tileDim] =
              arith::MulIOp::create(bb, loc, s, stickSizeVal).getResult();
          Value inputSlice = tensor::ExtractSliceOp::create(
              bb, loc, slTy, inputTile, inOff, inputSizesBase, inputStrides);

          llvm::SmallVector<OpFoldResult> sinkOff = sinkOffsetsBase;
          sinkOff[p] = s;
          return tensor::InsertSliceOp::create(
              bb, loc, inputSlice, sinkAccumulator, sinkOff, sinkSizes, sinkStrides);
        });
  }

  st.getDataTileMutable().set(acc);
  return success();
}

// Dispatch a store with an annotated output descriptor.
static LogicalResult dispatchSink(mlir::ktdp::StoreOp st,
                                  triton::SpyreTensorLayoutOp marker,
                                  PassContext &ctx) {
  auto tileTy = cast<mlir::ktdp::AccessTileType>(st.getAccessTile().getType());
  llvm::ArrayRef<int64_t> physBlock = tileTy.getShape();

  unsigned logRank =
      cast<RankedTensorType>(st.getDataTile().getType()).getRank();
  OperandCoords dC = OperandCoords::fromMarker(marker, logRank, physBlock);

  int physRank = (int)physBlock.size();
  llvm::SmallVector<int64_t> dimRoleD(physRank);
  for (int p = 0; p < physRank; ++p)
    dimRoleD[p] = marker.getPhysSrc()[p];

  OperandPlan dPlan = classify(st.getDataTile(), dC, dimRoleD);
  return emitSinkStage(st, dPlan);
}

// Dispatch a linalg.reduce whose input has a layout marker.
static LogicalResult dispatchReduce(linalg::ReduceOp rd, PassContext &ctx) {
  auto marker = findMarkerForOperand(rd.getInputs()[0], ctx);
  if (!marker)
    return rd.emitError(
        "spyre_tensor_layout: dispatchReduce called but no marker on input");
  unsigned logicalRank = 0;
  for (int64_t src : marker.getPhysSrc())
    if ((unsigned)(src + 1) > logicalRank)
      logicalRank = (unsigned)(src + 1);
  auto reductionDims = rd.getDimensions();
  (void)reductionDims;

  llvm::SmallVector<int64_t> canonicalAxes(logicalRank, -1);
  unsigned outAxis = 0;
  for (unsigned d = 0; d < logicalRank; ++d)
    if (!llvm::is_contained(rd.getDimensions(), (int64_t)d))
      canonicalAxes[d] = outAxis++;

  Block &combinerBlock = rd.getOperation()->getRegion(0).front();
  llvm::SmallVector<Operation *> combinerOps;
  for (Operation &op : combinerBlock.without_terminator())
    combinerOps.push_back(&op);
  auto combinerYield = cast<linalg::YieldOp>(combinerBlock.getTerminator());
  llvm::SmallVector<Value> yieldVals(combinerYield.getValues().begin(),
                                     combinerYield.getValues().end());
  llvm::SmallVector<Value> origBlockArgs(combinerBlock.getArguments().begin(),
                                         combinerBlock.getArguments().end());

  unsigned outputRank = logicalRank - (unsigned)rd.getDimensions().size();
  auto emitReduceOp = [outputRank,
                       combinerOps = std::move(combinerOps),
                       yieldVals = std::move(yieldVals),
                       origBlockArgs = std::move(origBlockArgs)](
                          OpBuilder &b, Location loc,
                          llvm::ArrayRef<Value> slices, Value acc,
                          RankedTensorType accTy) -> Value {
    auto sliceTy = cast<RankedTensorType>(slices[0].getType());
    llvm::SmallVector<int64_t> dims;
    for (unsigned d = outputRank; d < (unsigned)sliceTy.getRank(); ++d)
      dims.push_back((int64_t)d);
    return linalg::ReduceOp::create(
        b, loc, ValueRange{slices[0]}, ValueRange{acc}, dims,
        [&](OpBuilder &inner, Location iloc, ValueRange args) {
          IRMapping mapping;
          for (unsigned i = 0; i < origBlockArgs.size(); ++i)
            mapping.map(origBlockArgs[i], args[i]);
          for (Operation *op : combinerOps)
            inner.clone(*op, mapping);
          llvm::SmallVector<Value> mapped;
          for (Value v : yieldVals)
            mapped.push_back(mapping.lookupOrDefault(v));
          linalg::YieldOp::create(inner, iloc, mapped);
        }).getResult(0);
  };
  SourceOpSpec spec;
  spec.operands = {SourceOperandSpec{canonicalAxes}};
  spec.logicalRank = logicalRank;
  spec.emitOp = emitReduceOp;
  return dispatchSource(rd, spec, ctx);
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

  if (auto rd = dyn_cast<linalg::ReduceOp>(op)) {
    auto rdMarker = findMarkerForOperand(rd.getInputs()[0], ctx);
    unsigned logicalInputRank = 2;
    if (rdMarker) {
      for (int64_t src : rdMarker.getPhysSrc())
        if ((unsigned)(src + 1) > logicalInputRank)
          logicalInputRank = (unsigned)(src + 1);
    }
    return sourceNeedsDispatch(rd, logicalInputRank)
               ? (changed = true, dispatchReduce(rd, ctx))
               : success();
  }

  if (auto st = dyn_cast<mlir::ktdp::StoreOp>(op)) {
    auto dataTy = dyn_cast<RankedTensorType>(st.getDataTile().getType());
    auto tileTy = dyn_cast<mlir::ktdp::AccessTileType>(
        st.getAccessTile().getType());
    if (!dataTy || !tileTy ||
        dataTy.getRank() == (int)tileTy.getShape().size())
      return success();
    auto marker = findMarkerForStore(st.getDataTile(), ctx);
    if (!marker)
      return success();
    changed = true;
    return dispatchSink(st, marker, ctx);
  }

  return success();
}

bool synthesizeContractions(mlir::ModuleOp module, PassContext &ctx) {
  bool anyChanged = false;
  bool changed = true;
  while (changed) {
    changed = false;
    llvm::SmallVector<Operation *> candidates;
    module.walk([&](Operation *op) {
      if (isa<linalg::MatmulOp, linalg::ReduceOp, mlir::ktdp::StoreOp>(op))
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

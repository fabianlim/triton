//===- RewriteDescriptorLayout.cpp ----------------------------------------===//
//
// Rewrites logical tensor descriptors to their physical (stick-tiled) layout,
// driven by tt.spyre_tensor_layout markers. Runs after LowerComputeOps and
// before LowerInterTile in the TTIR->KTDP pipeline, so that tt.dot is already
// lowered to linalg.matmul before operands are physicalized.
//
// The physical layout is the OpSpec `device_coordinates` form, carried on the
// marker as three i64 arrays, one entry per physical dim:
//   phys_src[k] : logical dim k derives from
//   phys_op[k]  : 0 = identity, 1 = floordiv, 2 = mod
//   phys_arg[k] : divisor (floordiv) / modulus (mod); ignored for identity
// e.g. [M,N] stick-on-N -> phys_src=[1,0,1] phys_op=[1,0,2] phys_arg=[64,0,64]
//   => device_size [N//64, M, N%64].
//
// Staged model:
//   Phase 1 — physicalize each annotated descriptor (memView + access tiles +
//             loads + stores)
//   Phase 3 — erase all markers (and their now-dead bridge casts)
//
//===----------------------------------------------------------------------===//

#include "Dialect/KTDP/Transforms/Passes.h"
#include "Dialect/KTDP/Transforms/Utility.h"
#include "RewriteDescriptorLayout/PermutationUtils.h"
#include "RewriteDescriptorLayout/Types.h"
#include "RewriteDescriptorLayout/ContractionSynthesis.h"
#include "Ktdp/KtdpAttrs.hpp"
#include "Ktdp/KtdpDialect.hpp"
#include "Ktdp/KtdpOps.hpp"
#include "Ktdp/KtdpTypes.hpp"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Types.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <numeric>
#include <optional>

#define DEBUG_TYPE "rewrite-descriptor-layout"

namespace mlir::triton::ktdp {

#define GEN_PASS_DEF_REWRITEDESCRIPTORLAYOUT
#include "Dialect/KTDP/Transforms/Passes.h.inc"

} // namespace mlir::triton::ktdp

namespace {

using namespace mlir;
using namespace mlir::triton::ktdp;

/// Walk a def-chain of index arithmetic back to the single BlockArgument it
/// derives from.
BlockArgument traceToMLIRBlockArg(Value v) {
  while (true) {
    if (auto ba = dyn_cast<BlockArgument>(v))
      return ba;
    auto *op = v.getDefiningOp();
    if (!op)
      return nullptr;
    if (isa<arith::IndexCastOp, arith::IndexCastUIOp,
            arith::TruncIOp, arith::ExtSIOp, arith::ExtUIOp>(op)) {
      v = op->getOperand(0);
      continue;
    }
    if (isa<arith::MulIOp, arith::DivSIOp, arith::RemSIOp, arith::AddIOp>(op)) {
      if (op->getNumOperands() == 2 && getConstantInt(op->getOperand(1)))
        { v = op->getOperand(0); continue; }
    }
    return nullptr;
  }
}


struct RewriteDescriptorLayoutPass
    : public mlir::triton::ktdp::impl::RewriteDescriptorLayoutBase<
          RewriteDescriptorLayoutPass> {

  using RewriteDescriptorLayoutBase::RewriteDescriptorLayoutBase;

  // Maps each physical ConstructMemoryViewOp result -> its source marker.
  DenseMap<Value, triton::SpyreTensorLayoutOp> physMemViewToMarker;

  // Loops already rescaled to stick granularity.
  DenseSet<scf::ForOp> rescaledLoops;

  // Maps a physical ktdp.load result -> the logical permutation of a
  // linalg.transpose that was erased in Phase 1.
  llvm::DenseMap<mlir::Value, SmallVector<int64_t>> physicalLoadToTransposePerm;

  // Resolved from the pass option: true = "device" (physical row-major strides),
  // false = "host" (derive strides from logical strides via coord map).
  bool hwDataLayout = false;

  // --- Stride computation ---

  // Compute physical strides as row-major of the physical shape.
  static std::pair<SmallVector<int64_t>, SmallVector<Value>>
  buildPhysicalStrides(unsigned physRank, ArrayRef<int64_t> physStaticSizes,
                       ArrayRef<Value> physDynSizes, OpBuilder &b,
                       Location loc) {
    SmallVector<int64_t> physStaticStrides(physRank);
    SmallVector<Value> physDynStrides;

    bool hasAnyDynStride = false;
    for (unsigned k = 0; k < physRank; ++k) {
      if (physStaticSizes[k] == ShapedType::kDynamic) {
        physStaticStrides[k] = ShapedType::kDynamic;
        hasAnyDynStride = true;
      }
    }
    if (!hasAnyDynStride) {
      physStaticStrides[physRank - 1] = 1;
      for (int k = (int)physRank - 2; k >= 0; --k)
        physStaticStrides[k] =
            physStaticStrides[k + 1] * physStaticSizes[k + 1];
    } else {
      SmallVector<Value> strideSsaVals(physRank);
      Value one =
          arith::ConstantOp::create(b, loc, b.getIndexAttr(1)).getResult();
      strideSsaVals[physRank - 1] = one;
      auto getSizeVal = [&](unsigned dim) -> Value {
        if (physStaticSizes[dim] != ShapedType::kDynamic)
          return arith::ConstantOp::create(
                     b, loc, b.getIndexAttr(physStaticSizes[dim]))
              .getResult();
        int pos = 0;
        for (unsigned d = 0; d < dim; ++d)
          if (physStaticSizes[d] == ShapedType::kDynamic)
            ++pos;
        return physDynSizes[pos];
      };
      for (int k = (int)physRank - 2; k >= 0; --k) {
        Value innerSize = getSizeVal(k + 1);
        strideSsaVals[k] =
            arith::MulIOp::create(b, loc, strideSsaVals[k + 1], innerSize)
                .getResult();
      }
      for (unsigned k = 0; k < physRank; ++k) {
        physStaticStrides[k] = ShapedType::kDynamic;
        physDynStrides.push_back(strideSsaVals[k]);
      }
    }
    return {std::move(physStaticStrides), std::move(physDynStrides)};
  }

  // Compute physical strides from logical strides via the coordinate map.
  static FailureOr<std::pair<SmallVector<int64_t>, SmallVector<Value>>>
  buildLogicalStrides(unsigned physRank, ArrayRef<int64_t> physSrc,
                      ArrayRef<int64_t> physOp, ArrayRef<int64_t> physArg,
                      ArrayRef<int64_t> logStaticStrides,
                      ArrayRef<Value> logDynStrides,
                      ArrayRef<int> logDynStrideIdx, OpBuilder &b, Location loc,
                      llvm::function_ref<InFlightDiagnostic()> emitError) {
    SmallVector<int64_t> physStaticStrides(physRank);
    SmallVector<Value> physDynStrides;

    bool hasAnyDynStride = false;
    for (unsigned k = 0; k < physRank; ++k) {
      int64_t s = physSrc[k];
      auto op = static_cast<CoordOp>(physOp[k]);
      int64_t arg = physArg[k];
      int64_t logSt = logStaticStrides[s];

      if (logSt == ShapedType::kDynamic) {
        physStaticStrides[k] = ShapedType::kDynamic;
        hasAnyDynStride = true;
      } else if (op == CoordOp::FloorDiv) {
        physStaticStrides[k] = logSt * arg;
      } else {
        physStaticStrides[k] = logSt;
      }
    }
    if (hasAnyDynStride) {
      for (unsigned k = 0; k < physRank; ++k) {
        if (physStaticStrides[k] != ShapedType::kDynamic)
          continue;
        int64_t s = physSrc[k];
        auto op = static_cast<CoordOp>(physOp[k]);
        int64_t arg = physArg[k];
        if (logDynStrideIdx[s] < 0)
          return emitError()
                 << "spyre_tensor_layout: expected dynamic stride for dim";
        Value logDynSt = logDynStrides[logDynStrideIdx[s]];
        if (op == CoordOp::FloorDiv) {
          Value argVal =
              arith::ConstantOp::create(b, loc, b.getIndexAttr(arg));
          physDynStrides.push_back(
              arith::MulIOp::create(b, loc, logDynSt, argVal).getResult());
        } else {
          physDynStrides.push_back(logDynSt);
        }
      }
    }
    return std::make_pair(std::move(physStaticStrides),
                          std::move(physDynStrides));
  }

  // --- Loop rescaling ---

  // After rescaleEnclosingLoop(iv, factor), fix muli(iv, C) constants.
  void scaleDownIVMuls(BlockArgument iv, int64_t factor) {
    if (factor <= 1)
      return;
    for (Operation *user : llvm::make_early_inc_range(iv.getUsers())) {
      auto muli = dyn_cast<arith::MulIOp>(user);
      if (!muli || muli.getLhs() != iv)
        continue;
      auto cst = getConstantInt(muli.getRhs());
      if (!cst || (*cst % factor) != 0)
        continue;
      OpBuilder b(muli);
      Value newCst = arith::ConstantOp::create(
          b, muli.getLoc(),
          b.getIntegerAttr(muli.getRhs().getType(), *cst / factor));
      muli.getRhs().replaceAllUsesWith(newCst);
    }
  }

  // Rescale an scf.for loop to stick granularity.
  void rescaleEnclosingLoop(scf::ForOp forOp, int64_t factor) {
    LLVM_DEBUG(llvm::dbgs() << "  rescaling loop by factor " << factor << "\n");
    Type ivTy = forOp.getInductionVar().getType();
    OpBuilder b(forOp);
    Location loc = forOp.getLoc();
    if (factor > 1) {
      Value factorV = arith::ConstantOp::create(b, loc,
                          b.getIntegerAttr(ivTy, factor));
      Value newUb = arith::MulIOp::create(b, loc,
                        forOp.getUpperBound(), factorV).getResult();
      forOp.setUpperBound(newUb);
      forOp.setStep(factorV);
    } else {
      Value c1v = arith::ConstantOp::create(b, loc,
                      b.getIntegerAttr(ivTy, 1));
      forOp.setStep(c1v);
    }
  }

  // --- Access tile rewriting ---

  // Rebuild ConstructAccessTilesOp with the physical memView + block shape.
  LogicalResult rewriteAccessTile(mlir::ktdp::ConstructAccessTilesOp tileOp,
                                  Value newMemView,
                                  ArrayRef<int64_t> physSrc,
                                  ArrayRef<int64_t> physOp,
                                  ArrayRef<int64_t> physArg) {
    OpBuilder b(tileOp);
    Location loc = tileOp.getLoc();
    MLIRContext *ctx = b.getContext();

    auto logTileType = tileOp.getResult().getType();
    ArrayRef<int64_t> logBlock = logTileType.getShape();
    unsigned logRank = logBlock.size();
    unsigned physRank = physSrc.size();

    // Compute physical block shape via applyCoordMap.
    SmallVector<int64_t> physBlock;
    if (!applyCoordMap(logBlock, physSrc, physOp, physArg, physBlock))
      return tileOp.emitError(
          "spyre_tensor_layout: cannot derive static block_shape");
    for (unsigned k = 0; k < physRank; ++k)
      if (physSrc[k] < 0 || physSrc[k] >= (int64_t)logRank)
        return tileOp.emitError("spyre_tensor_layout: phys_src out of range");

    LLVM_DEBUG({
      llvm::dbgs() << "  access tile physBlock: ";
      llvm::interleaveComma(physBlock, llvm::dbgs());
      llvm::dbgs() << "\n";
    });

    // Validate stick width.
    for (unsigned k = 0; k < physRank; ++k) {
      if (static_cast<CoordOp>(physOp[k]) != CoordOp::Mod)
        continue;
      int64_t logExtent = logBlock[physSrc[k]];
      if (logExtent != ShapedType::kDynamic && logExtent < physArg[k])
        return tileOp.emitError(
                   "spyre_tensor_layout: block extent of stick dim (")
               << logExtent << ") is smaller than the stick size ("
               << physArg[k] << "); a stick dim cannot be sub-stick";
    }

    // Map the logical index operands to physical index operands.
    //
    // The op's base_map may have fewer inputs than logRank when the custom
    // parser deduplicates identical SSA operands (e.g. [%x, %x] becomes a
    // single operand with base_map (d0) -> (d0, d0)).  Expand through the
    // base_map to recover per-logical-dim values.
    SmallVector<Value> rawIndices(tileOp.getIndices().begin(),
                                  tileOp.getIndices().end());
    AffineMap baseMap = tileOp.getBaseMap();
    SmallVector<Value> logIdx(logRank);
    if (baseMap.getNumResults() == logRank &&
        baseMap.getNumInputs() == rawIndices.size()) {
      for (unsigned d = 0; d < logRank; ++d) {
        AffineExpr expr = baseMap.getResult(d);
        if (auto dimExpr = dyn_cast<AffineDimExpr>(expr))
          logIdx[d] = rawIndices[dimExpr.getPosition()];
        else
          logIdx[d] = rawIndices[0];
      }
    } else {
      logIdx.assign(rawIndices.begin(), rawIndices.end());
    }

    // Two passes: first rescale loops (mutating side effect), then compute
    // indices.  This avoids a latent issue where two physical dims sharing the
    // same logical source SSA value could see inconsistent IR if rescaling for
    // the first dim mutated state read by the second.

    // Pass 1: rescale enclosing loops for all FloorDiv dims (idempotent via
    // rescaledLoops set).
    for (unsigned k = 0; k < physRank; ++k) {
      if (static_cast<CoordOp>(physOp[k]) != CoordOp::FloorDiv)
        continue;
      Value logI = logIdx[physSrc[k]];
      BlockArgument iv = traceToMLIRBlockArg(logI);
      scf::ForOp forOp = iv ? dyn_cast_or_null<scf::ForOp>(
                                  iv.getOwner()->getParentOp())
                            : nullptr;
      if (forOp && forOp.getInductionVar() == iv) {
        if (rescaledLoops.insert(forOp).second) {
          rescaleEnclosingLoop(forOp, physBlock[k]);
          scaleDownIVMuls(iv, physBlock[k]);
        }
      }
    }

    // Pass 2: compute all physical index values from (now-stable) IR.
    SmallVector<Value> physIdx;
    for (unsigned k = 0; k < physRank; ++k) {
      int64_t src = physSrc[k];
      auto op = static_cast<CoordOp>(physOp[k]);
      int64_t arg = physArg[k];
      Value logI = logIdx[src];
      switch (op) {
      case CoordOp::Identity:
        physIdx.push_back(logI);
        break;
      case CoordOp::FloorDiv: {
        BlockArgument iv = traceToMLIRBlockArg(logI);
        scf::ForOp forOp = iv ? dyn_cast_or_null<scf::ForOp>(
                                    iv.getOwner()->getParentOp())
                              : nullptr;
        if (forOp && forOp.getInductionVar() == iv) {
          Value ivIdx = iv.getType().isIndex()
                            ? iv
                            : arith::IndexCastOp::create(b, loc,
                                  b.getIndexType(), iv).getResult();
          physIdx.push_back(ivIdx);
        } else {
          Value c = arith::ConstantOp::create(b, loc, b.getIndexAttr(arg));
          physIdx.push_back(
              arith::DivSIOp::create(b, loc, logI, c).getResult());
        }
        break;
      }
      case CoordOp::Mod: {
        Value c = arith::ConstantOp::create(b, loc, b.getIndexAttr(arg));
        physIdx.push_back(
            arith::RemSIOp::create(b, loc, logI, c).getResult());
        break;
      }
      }
    }

    auto physTileType = mlir::ktdp::AccessTileType::get(physBlock,
                                                         b.getIndexType());
    auto identityMap = AffineMap::getMultiDimIdentityMap(physRank, ctx);
    auto coordSet = buildRangeSetND(ctx, physBlock);

    auto newTile = mlir::ktdp::ConstructAccessTilesOp::create(
        b, loc, physTileType, newMemView,
        identityMap, physIdx, /*symbol_operands=*/ValueRange{},
        coordSet, identityMap);

    // Update consumers (ktdp.load / ktdp.store).
    for (auto *user : llvm::make_early_inc_range(tileOp.getResult().getUsers())) {
      if (auto ld = dyn_cast<mlir::ktdp::LoadOp>(user)) {
        retypeLoad(ld, newTile.getResult(), physBlock);
      } else if (auto st = dyn_cast<mlir::ktdp::StoreOp>(user)) {
        redirectStoreAccessTile(st, newTile.getResult());
      } else {
        return user->emitError(
            "spyre_tensor_layout: unexpected user of access tile");
      }
    }

    tileOp.erase();
    return success();
  }

  // Rebuild ConstructIndirectAccessTilesOp over the physical memView.
  LogicalResult rewriteIndirectAccessTile(
      mlir::ktdp::ConstructIndirectAccessTilesOp tileOp, Value newMemView,
      ArrayRef<int64_t> physSrc, ArrayRef<int64_t> physOp,
      ArrayRef<int64_t> physArg) {
    OpBuilder b(tileOp);
    Location loc = tileOp.getLoc();
    MLIRContext *ctx = b.getContext();

    unsigned physRank = physSrc.size();

    auto logTileType = tileOp.getResult().getType();
    ArrayRef<int64_t> logBlock = logTileType.getShape();
    unsigned logRank = logBlock.size();

    auto oldKinds = tileOp.getPerDimSubscriptKinds();
    auto oldMaps  = tileOp.getPerDimSubscriptMaps();
    unsigned numCaptured = tileOp.getCapturedVariables().size();

    // Capability gate.
    if (logRank != 2)
      return tileOp.emitError(
          "spyre_tensor_layout: physicalizing an indirect access tile is only "
          "supported for a rank-2 gather (got logical rank ")
          << logRank << ")";
    if (!cast<BoolAttr>(oldKinds[0]).getValue() ||
        cast<BoolAttr>(oldKinds[1]).getValue())
      return tileOp.emitError(
          "spyre_tensor_layout: physicalizing an indirect access tile assumes "
          "logical dim 0 is indirect (gather) and logical dim 1 is direct; "
          "got a different subscript-kind layout");
    for (unsigned p = 0; p < physRank; ++p)
      if (physSrc[p] == 0 &&
          static_cast<CoordOp>(physOp[p]) != CoordOp::Identity)
        return tileOp.emitError(
            "spyre_tensor_layout: stick-splitting the indirect (gather) row "
            "dim is not supported");

    SmallVector<int64_t> physBlock;
    if (!applyCoordMap(logBlock, physSrc, physOp, physArg, physBlock))
      return tileOp.emitError(
          "spyre_tensor_layout: cannot derive static block_shape for "
          "indirect access tile");

    unsigned newDimCount = numCaptured + physRank;
    auto newVar = [&](unsigned slot) { return getAffineDimExpr(slot, ctx); };

    // Reconstruct logical iteration variables from physical ones.
    SmallVector<AffineExpr> logicalFromPhysical(logRank);
    SmallVector<bool> contributed(logRank, false);
    for (unsigned p = 0; p < physRank; ++p) {
      int64_t L = physSrc[p];
      if (L < 0 || L >= (int64_t)logRank)
        return tileOp.emitError(
            "spyre_tensor_layout: phys_src out of range for indirect tile");
      auto op = static_cast<CoordOp>(physOp[p]);
      int64_t arg = physArg[p];
      AffineExpr v = newVar(numCaptured + p);

      AffineExpr piece;
      switch (op) {
      case CoordOp::Identity: piece = v;       break;
      case CoordOp::FloorDiv: piece = v * arg; break;
      case CoordOp::Mod:      piece = v;       break;
      }

      logicalFromPhysical[L] =
          contributed[L] ? logicalFromPhysical[L] + piece : piece;
      contributed[L] = true;
    }

    // Build substitution from old domain to new domain.
    SmallVector<AffineExpr> oldToNew(numCaptured + logRank);
    for (unsigned c = 0; c < numCaptured; ++c)
      oldToNew[c] = newVar(c);
    for (unsigned L = 0; L < logRank; ++L)
      oldToNew[numCaptured + L] = logicalFromPhysical[L];

    // Build per-physical-dim kinds + maps.
    SmallVector<Attribute> newKinds, newMaps;
    for (unsigned p = 0; p < physRank; ++p) {
      int64_t L = physSrc[p];
      auto op  = static_cast<CoordOp>(physOp[p]);
      int64_t arg = physArg[p];

      auto oldKindAttr = cast<BoolAttr>(oldKinds[L]);
      auto oldMapAttr  = cast<AffineMapAttr>(oldMaps[L]);

      AffineExpr oldExpr = oldMapAttr.getValue().getResult(0);
      AffineExpr reExpr = oldExpr.replaceDims(oldToNew);

      AffineExpr physExpr;
      switch (op) {
      case CoordOp::Identity: physExpr = reExpr; break;
      case CoordOp::FloorDiv: physExpr = reExpr.floorDiv(arg); break;
      case CoordOp::Mod:      physExpr = reExpr % arg; break;
      }

      newKinds.push_back(oldKindAttr);
      newMaps.push_back(AffineMapAttr::get(
          AffineMap::get(newDimCount, /*symbolCount=*/0, physExpr, ctx)));
    }

    // Build new intermediate-variable space.
    SmallVector<AffineExpr> setConstraints;
    SmallVector<bool> setEqFlags;
    for (unsigned p = 0; p < physRank; ++p) {
      AffineExpr v = getAffineDimExpr(p, ctx);
      setConstraints.push_back(v);
      setEqFlags.push_back(false);
      setConstraints.push_back(
          getAffineConstantExpr(physBlock[p] - 1, ctx) - v);
      setEqFlags.push_back(false);
    }
    auto newSpaceSet = IntegerSet::get(physRank, 0, setConstraints, setEqFlags);
    auto newSpaceOrder = AffineMap::getMultiDimIdentityMap(physRank, ctx);

    auto physTileType = mlir::ktdp::AccessTileType::get(physBlock,
                                                         b.getIndexType());

    auto newTile = mlir::ktdp::ConstructIndirectAccessTilesOp::create(
        b, loc, physTileType, newMemView,
        ArrayAttr::get(ctx, newKinds),
        ArrayAttr::get(ctx, newMaps),
        tileOp.getIndirectMemrefs(),
        tileOp.getCapturedVariables(),
        tileOp.getSymbolOperands(),
        newSpaceSet, newSpaceOrder);

    // Update consumers.
    for (auto *user : llvm::make_early_inc_range(tileOp.getResult().getUsers())) {
      if (auto ld = dyn_cast<mlir::ktdp::LoadOp>(user)) {
        retypeLoad(ld, newTile.getResult(), physBlock);
      } else {
        return user->emitError(
            "spyre_tensor_layout: unexpected user of indirect access tile");
      }
    }

    tileOp.erase();
    return success();
  }

  // --- Load/store consumer updates ---

  // True if the op is a shape-constraining op whose result shape is NOT
  // inherited from a single physical input.
  static bool isContractionOp(Operation *op) {
    if (isa<linalg::ReduceOp>(op))
      return true;
    auto linalgOp = dyn_cast<linalg::LinalgOp>(op);
    return linalgOp && linalg::isaContractionOpInterface(linalgOp);
  }

  // Retype ktdp.load: replace with a new load of the physical tensor type.
  void retypeLoad(mlir::ktdp::LoadOp ld, Value newTile,
                  ArrayRef<int64_t> physBlock) {
    OpBuilder b(ld);
    auto elemTy = cast<RankedTensorType>(ld.getResult().getType())
                      .getElementType();
    auto physResTy = RankedTensorType::get(physBlock, elemTy);
    auto newLd = mlir::ktdp::LoadOp::create(b, ld.getLoc(), physResTy, newTile);
    retypeChain(ld.getResult(), newLd.getResult());
    ld.erase();
  }

  // Redirect ktdp.store's access tile operand to the new physical tile.
  void redirectStoreAccessTile(mlir::ktdp::StoreOp st, Value newTile) {
    st.getAccessTileMutable().set(newTile);
  }

  // Forward-retype the elementwise op chain.
  void retypeChain(Value oldVal, Value newVal) {
    Value physLoadResult = newVal;
    oldVal.replaceAllUsesWith(newVal);
    SmallVector<Operation *> worklist(newVal.getUsers().begin(),
                                      newVal.getUsers().end());
    while (!worklist.empty()) {
      Operation *op = worklist.pop_back_val();
      if (op->getNumResults() != 1)
        continue;
      if (isContractionOp(op))
        continue;
      if (auto tr = dyn_cast<linalg::TransposeOp>(op)) {
        auto perm = SmallVector<int64_t>(tr.getPermutation());
        LLVM_DEBUG({
          llvm::dbgs() << "  erasing transpose, perm: ";
          llvm::interleaveComma(perm, llvm::dbgs());
          llvm::dbgs() << "\n";
        });
        physicalLoadToTransposePerm[physLoadResult] = perm;
        Value physInput = tr.getInput();
        SmallVector<Operation *> fmrConsumers(tr.getResult()[0].getUsers().begin(),
                                              tr.getResult()[0].getUsers().end());
        tr.getResult()[0].replaceAllUsesWith(physInput);
        worklist.append(fmrConsumers.begin(), fmrConsumers.end());
        tr.erase();
        continue;
      }
      auto resTy  = dyn_cast<RankedTensorType>(op->getResult(0).getType());
      auto opndTy = op->getNumOperands() > 0
                        ? dyn_cast<RankedTensorType>(op->getOperand(0).getType())
                        : nullptr;
      if (!resTy || !opndTy || resTy.getShape() == opndTy.getShape())
        continue;
      op->getResult(0).setType(
          RankedTensorType::get(opndTy.getShape(), resTy.getElementType()));
      worklist.append(op->getResult(0).getUsers().begin(),
                      op->getResult(0).getUsers().end());
    }
  }

  // --- Phase 1: physicalize one descriptor ---

  LogicalResult rewriteOnePhysicalize(triton::SpyreTensorLayoutOp marker) {
    Value desc = marker.getDesc();

    if (!isLoweredDescriptor(desc))
      return marker.emitError(
          "spyre_tensor_layout: desc operand is not a lowered descriptor — "
          "pass must run after LowerDescriptorMemory");

    Value memView = getDescriptorMemView(desc);
    auto memViewOp = memView.getDefiningOp<mlir::ktdp::ConstructMemoryViewOp>();
    if (!memViewOp)
      return marker.emitError(
          "spyre_tensor_layout: cannot locate construct_memory_view behind cast");

    ArrayRef<int64_t> physSrc = marker.getPhysSrc();
    ArrayRef<int64_t> physOp  = marker.getPhysOp();
    ArrayRef<int64_t> physArg = marker.getPhysArg();
    unsigned physRank = physSrc.size();

    LLVM_DEBUG({
      llvm::dbgs() << "[rewrite-descriptor-layout] physicalizing: physRank="
                   << physRank << "\n";
      llvm::dbgs() << "  physSrc="; llvm::interleaveComma(physSrc, llvm::dbgs()); llvm::dbgs() << "\n";
      llvm::dbgs() << "  physOp="; llvm::interleaveComma(physOp, llvm::dbgs()); llvm::dbgs() << "\n";
      llvm::dbgs() << "  physArg="; llvm::interleaveComma(physArg, llvm::dbgs()); llvm::dbgs() << "\n";
    });

    // --- 1. Build physical construct_memory_view ---
    Value newMemView;
    {
      OpBuilder b(memViewOp);
      Location loc = memViewOp.getLoc();
      MLIRContext *ctx = b.getContext();

      ArrayRef<int64_t> logStaticSizes   = memViewOp.getStaticSizes();
      ArrayRef<int64_t> logStaticStrides = memViewOp.getStaticStrides();
      SmallVector<Value> logDynSizes(memViewOp.getSizes().begin(),
                                     memViewOp.getSizes().end());
      SmallVector<Value> logDynStrides(memViewOp.getStrides().begin(),
                                       memViewOp.getStrides().end());

      SmallVector<int> logDynIdx(logStaticSizes.size(), -1);
      {
        int dynPos = 0;
        for (unsigned i = 0; i < logStaticSizes.size(); ++i)
          if (logStaticSizes[i] == ShapedType::kDynamic)
            logDynIdx[i] = dynPos++;
      }
      SmallVector<int> logDynStrideIdx(logStaticStrides.size(), -1);
      {
        int dynPos = 0;
        for (unsigned i = 0; i < logStaticStrides.size(); ++i)
          if (logStaticStrides[i] == ShapedType::kDynamic)
            logDynStrideIdx[i] = dynPos++;
      }

      SmallVector<int64_t> physStaticSizes;
      SmallVector<Value>   physDynSizes;

      for (unsigned k = 0; k < physRank; ++k) {
        int64_t src = physSrc[k];
        auto op = static_cast<CoordOp>(physOp[k]);
        int64_t arg = physArg[k];
        if (src < 0 || src >= (int64_t)logStaticSizes.size())
          return marker.emitError("spyre_tensor_layout: phys_src out of range");

        int64_t logSz = logStaticSizes[src];
        auto physSz = applyStatic(logSz, op, arg);
        if (physSz) {
          physStaticSizes.push_back(*physSz);
        } else {
          physStaticSizes.push_back(ShapedType::kDynamic);
          if (op == CoordOp::FloorDiv) {
            if (logDynIdx[src] < 0)
              return marker.emitError(
                  "spyre_tensor_layout: expected dynamic size for floordiv dim");
            Value logDynSize = logDynSizes[logDynIdx[src]];
            Value argIdx = arith::ConstantOp::create(
                b, loc, b.getIndexAttr(arg));
            physDynSizes.push_back(
                arith::CeilDivSIOp::create(b, loc, logDynSize, argIdx).getResult());
          } else {
            if (logDynIdx[src] < 0)
              return marker.emitError(
                  "spyre_tensor_layout: expected dynamic size for identity dim");
            physDynSizes.push_back(logDynSizes[logDynIdx[src]]);
          }
        }
      }

      LLVM_DEBUG({
        llvm::dbgs() << "  physical sizes: ";
        llvm::interleaveComma(physStaticSizes, llvm::dbgs());
        llvm::dbgs() << "\n";
      });

      // Compute physical strides.
      SmallVector<int64_t> physStaticStrides;
      SmallVector<Value> physDynStrides;
      if (hwDataLayout) {
        std::tie(physStaticStrides, physDynStrides) =
            buildPhysicalStrides(physRank, physStaticSizes, physDynSizes, b, loc);
      } else {
        auto result =
            buildLogicalStrides(physRank, physSrc, physOp, physArg,
                                logStaticStrides, logDynStrides, logDynStrideIdx,
                                b, loc,
                                [&]() { return marker.emitError(); });
        if (failed(result))
          return failure();
        std::tie(physStaticStrides, physDynStrides) = std::move(*result);
      }

      // Physical memref type.
      auto logMemrefType = cast<MemRefType>(memViewOp.getResult().getType());
      auto physMemrefType = MemRefType::get(physStaticSizes,
                                            logMemrefType.getElementType());
      auto memSpaceAttr = memViewOp.getMemorySpace();
      auto coordSet = IntegerSetAttr::get(buildRangeSetND(ctx, physStaticSizes));

      newMemView = mlir::ktdp::ConstructMemoryViewOp::create(
                      b, loc, physMemrefType,
                      memViewOp.getOffset(),
                      physDynSizes, physDynStrides,
                      physStaticSizes, physStaticStrides,
                      memSpaceAttr, coordSet)
                      .getResult();
    }

    // Record the physical memView -> marker mapping for Phase 2.
    physMemViewToMarker[newMemView] = marker;

    // --- 2. Rebuild each access tile that uses the old memView ---
    SmallVector<mlir::ktdp::ConstructAccessTilesOp> tiles;
    SmallVector<mlir::ktdp::ConstructIndirectAccessTilesOp> indirectTiles;
    for (auto *user : memView.getUsers()) {
      if (auto tile = dyn_cast<mlir::ktdp::ConstructAccessTilesOp>(user))
        tiles.push_back(tile);
      else if (auto tile =
                   dyn_cast<mlir::ktdp::ConstructIndirectAccessTilesOp>(user))
        indirectTiles.push_back(tile);
    }

    for (auto tileOp : tiles) {
      if (failed(rewriteAccessTile(tileOp, newMemView, physSrc, physOp, physArg)))
        return failure();
    }
    for (auto tileOp : indirectTiles) {
      if (failed(rewriteIndirectAccessTile(tileOp, newMemView, physSrc, physOp, physArg)))
        return failure();
    }

    return success();
  }

  // --- Phase 3: marker cleanup ---

  // Erase a marker and its now-dead bridge cast.
  void eraseMarker(triton::SpyreTensorLayoutOp marker) {
    if (!marker->getBlock())
      return;
    Value desc = marker.getDesc();
    auto castOp = desc.getDefiningOp<UnrealizedConversionCastOp>();
    marker.erase();
    if (castOp && castOp.use_empty())
      castOp.erase();
  }

  // --- Pass entry point ---

  void runOnOperation() override {
    ModuleOp module = getOperation();

    // Resolve the data-layout option.
    hwDataLayout = (dataLayout == "device");

    // Collect markers up front; mutating while walking invalidates the cursor.
    SmallVector<triton::SpyreTensorLayoutOp> markers;
    module.walk([&](triton::SpyreTensorLayoutOp op) { markers.push_back(op); });

    LLVM_DEBUG(llvm::dbgs() << "[rewrite-descriptor-layout] found "
                            << markers.size() << " layout markers\n");

    // Phase 1: physicalize each annotated descriptor.
    for (auto marker : markers)
      if (failed(rewriteOnePhysicalize(marker)))
        return signalPassFailure();

    LLVM_DEBUG(llvm::dbgs() << "[rewrite-descriptor-layout] Phase 1 complete, "
                            << "entering Phase 2 (contraction synthesis)\n");

    // Phase 2: synthesize contractions via greedy pattern rewrite.
    {
      PassContext ctx{physMemViewToMarker, physicalLoadToTransposePerm};
      RewritePatternSet patterns(module.getContext());
      populateContractionPatterns(patterns, ctx);
      // Collect candidate ops (only op types our patterns target).
      SmallVector<Operation *> candidates;
      module.walk([&](Operation *op) {
        if (isa<linalg::MatmulOp, linalg::BatchMatmulOp, linalg::ReduceOp,
                mlir::ktdp::StoreOp>(op))
          candidates.push_back(op);
      });
      GreedyRewriteConfig config;
      config.enableFolding(false);
      config.enableConstantCSE(false);
      config.setStrictness(GreedyRewriteStrictness::ExistingAndNewOps);
      (void)applyOpPatternsGreedily(candidates,
                                    FrozenRewritePatternSet(std::move(patterns)),
                                    config);
      if (ctx.hadError)
        return signalPassFailure();
    }

    LLVM_DEBUG(llvm::dbgs() << "[rewrite-descriptor-layout] Phase 2 complete, "
                            << "erasing " << markers.size() << " markers\n");

    // Phase 3: erase all markers (and their now-dead bridge casts).
    for (auto marker : markers)
      eraseMarker(marker);
  }
};

} // namespace

//===- LowerComputeOps.cpp - Lower tt compute ops to linalg/tensor --------===//
//
// Lowers Triton compute ops to linalg and tensor dialect.
//
// Every op this pass puts on a compute chain is a `linalg.generic`, and a
// coordinate change is carried as an operand indexing map rather than as an op
// of its own. Two reasons the named linalg ops are gone:
//
//   1. Some cannot reach a binary. The backend's compute allowlist is
//      add/mul/sub/reduce, so a named linalg.matmul, linalg.broadcast or
//      linalg.transpose is rejected; the same computation as a generic is not.
//   2. A named op is a fusion wall. Upstream's elementwise fusion composes
//      indexing maps and matches generic-into-generic only, so every named op
//      left on a chain is a point where fusion silently stops.
//
// Everything emitted here is LOGICAL: no shape, map or iterator list encodes a
// hardware constraint. The stick width and tile extents belong to
// RewriteDescriptorLayout, which reads an annotation this pass cannot see.
//
// Each emitted generic carries a `doc` naming the Triton op it came from, since
// the op name no longer does. It is for readers only -- no pass keys on it.
//
// Patterns are organized into groups:
//
//   Group A — Shape manipulation (no compute; a map, not an op)
//   Group B — Reduction (combiner region, identity element)
//   Group C — Matrix multiply (contraction stated as maps)
//
// Trivially dead ops left by prior passes are swept at the end of this pass.
//
//===----------------------------------------------------------------------===//

#include "Dialect/KTDP/Transforms/Passes.h"
#include "Dialect/KTDP/Transforms/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;

namespace mlir::triton::ktdp {
#define GEN_PASS_DEF_LOWERCOMPUTEOPS
#include "Dialect/KTDP/Transforms/Passes.h.inc"
} // namespace mlir::triton::ktdp

namespace {

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

/// Returns the identity attribute for a reduction combiner op.
/// Returns std::nullopt if the combiner op is not recognised.
static std::optional<TypedAttr>
getReductionNeutralAttr(Operation *combinerOp, Type elemType,
                        MLIRContext *ctx) {
  Builder b(ctx);
  if (isa<arith::MaxNumFOp>(combinerOp)) {
    auto ftype = cast<FloatType>(elemType);
    return TypedAttr(b.getFloatAttr(
        ftype, APFloat::getInf(ftype.getFloatSemantics(), /*neg=*/true)));
  }
  if (isa<arith::MinNumFOp>(combinerOp)) {
    auto ftype = cast<FloatType>(elemType);
    return TypedAttr(b.getFloatAttr(
        ftype, APFloat::getInf(ftype.getFloatSemantics(), /*neg=*/false)));
  }
  // arith.select has no algebraic neutral element, but is used as the index
  // lane in multi-operand reductions (e.g. argmax). The value lane's -inf
  // neutral guarantees every real element replaces the init, so the index
  // init is never the final answer. Use -1 as an obvious invalid-index
  // sentinel so incorrect results are detectable rather than silently 0.
  if (isa<arith::SelectOp>(combinerOp)) {
    if (auto itype = dyn_cast<IntegerType>(elemType))
      return TypedAttr(b.getIntegerAttr(itype, -1));
  }
  return arith::getNeutralElement(combinerOp);
}

/// Build a `linalg.generic` whose body only forwards its input: one input, one
/// output, and a body that is a bare `linalg.yield` of the input block
/// argument. `inMap` says where each output coordinate reads its input from, so
/// the coordinate change lives entirely in that map and no data is computed.
///
/// This is the emission for every pure shape op on a compute chain --
/// tt.broadcast, tt.expand_dims and tt.trans differ only in `inMap`:
///   broadcast   `(d0, d1) -> (d0, 0)`    a constant 0 in each expanded dim
///   expand_dims `(d0, d1) -> (d1)`       the size-1 dim is dropped
///   trans       `(d0, d1) -> (d1, d0)`   a permutation
///
/// The output map is always the identity: the result is written in its own
/// coordinate order, and the iterators are all `parallel` because every output
/// element is independent.
///
/// `doc` names the originating Triton op. It prints in the assembly and is for
/// readers only -- no pass keys on it. See the pass header for why every
/// emitted generic carries one.
static Value buildCopyGeneric(ConversionPatternRewriter &rewriter, Location loc,
                              Value input, RankedTensorType resultType,
                              AffineMap inMap, StringRef doc) {
  MLIRContext *ctx = rewriter.getContext();
  int64_t rank = resultType.getRank();

  Value empty =
      mlir::triton::ktdp::createEmptyTensor(rewriter, loc, resultType);

  SmallVector<AffineMap> maps = {inMap,
                                 AffineMap::getMultiDimIdentityMap(rank, ctx)};
  SmallVector<utils::IteratorType> iterators(rank,
                                             utils::IteratorType::parallel);

  auto generic = linalg::GenericOp::create(
      rewriter, loc, TypeRange{resultType}, ValueRange{input},
      ValueRange{empty}, maps, iterators,
      [](OpBuilder &b, Location loc, ValueRange args) {
        linalg::YieldOp::create(b, loc, args[0]);
      });
  generic.setDocAttr(rewriter.getStringAttr(doc));
  return generic.getResult(0);
}

//===----------------------------------------------------------------------===//
// Group A — Shape manipulation (pure tensor restructuring, no compute)
//
// No combiner region, no init value -- each of these restates a coordinate
// change and computes nothing.
//
// A3, A4 and A5 are the three that became generics, and they share one shape: a
// copy-only body (a bare linalg.yield of the input block argument) and
// all-parallel iterators, differing only in the operand map. buildCopyGeneric
// above is that shared template; each pattern's job is just to build the map.
//
//   A1. tt.splat      → linalg.fill        scalar → tensor (fill all elems)
//   A2. tt.reshape    → tensor.reshape      same elems, new shape
//   A3. tt.expand_dims→ linalg.generic      map drops the inserted size-1 dim
//   A4. tt.broadcast  → linalg.generic      map reads a constant 0 in each
//                                            expanded dim
//   A5. tt.trans      → linalg.generic      map permutes
//   A6. tt.join       → tensor.expand_shape + tensor.concat
//                                            two tensors → new minor dim
//   A7. tt.split      → tensor.extract_slice × 2
//                                            last dim=2 → two tensors
//
// A1, A2, A6 and A7 stay as they were: none is a compute-chain op. linalg.fill
// in particular must stay named -- DropReductionInitFill finds a reduction's
// init by looking for a linalg.fill on its outs.
//===----------------------------------------------------------------------===//

/// A1. tt.splat → linalg.fill
///   scalar 42.0  →  tensor<4x8xf32> filled with 42.0
///
/// Alternative: tt.splat → tensor.splat (single op, simpler).
/// linalg.fill is preferred when downstream passes fuse linalg ops.
struct ConvertTTSplat : public OpConversionPattern<triton::SplatOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::SplatOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto resultType = cast<RankedTensorType>(op.getResult().getType());
    Value empty = mlir::triton::ktdp::createEmptyTensor(rewriter, op.getLoc(),
                                                        resultType);
    auto fill = linalg::FillOp::create(rewriter, op.getLoc(),
                                       adaptor.getSrc(), empty);
    rewriter.replaceOp(op, fill.getResult(0));
    return success();
  }
};

/// A2. tt.reshape → tensor.reshape
///   tensor<512xf32>  →  tensor<16x32xf32>  (same 512 elements)
struct ConvertTTReshape : public OpConversionPattern<triton::ReshapeOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::ReshapeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto resultType = cast<RankedTensorType>(op.getResult().getType());

    SmallVector<Value> dims;
    for (int64_t d : resultType.getShape())
      dims.push_back(arith::ConstantOp::create(
          rewriter, loc, rewriter.getIndexType(), rewriter.getIndexAttr(d)));

    auto shapeTensor = tensor::FromElementsOp::create(rewriter, loc, dims);
    auto reshape = tensor::ReshapeOp::create(
        rewriter, loc, resultType, adaptor.getSrc(), shapeTensor);
    rewriter.replaceOp(op, reshape.getResult());
    return success();
  }
};

/// A3. tt.expand_dims → linalg.generic with a rank-reducing operand map
///   tensor<8xf32>  →  tensor<8x1xf32>  (insert size-1 dim at axis)
///
/// The inserted dim has extent 1, so it contributes no input coordinate: the
/// operand map simply omits it. Reading `(d0, d1) -> (d0)` at rank 2 says "the
/// input is indexed by the surviving dims, in order" -- which is exactly what
/// inserting a size-1 dim means, expressed as an access pattern.
///
/// Emitted as a generic rather than `tensor.expand_shape` so that no non-generic
/// op sits on a compute chain. The shape-carrying reassociation the expand_shape
/// form needs is subsumed by the map.
struct ConvertTTExpandDims
    : public OpConversionPattern<triton::ExpandDimsOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::ExpandDimsOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    MLIRContext *ctx = rewriter.getContext();
    Value input = adaptor.getSrc();
    int32_t axis = op.getAxis();

    auto resultType = cast<RankedTensorType>(op.getResult().getType());
    int64_t rank = resultType.getRank();

    // Every result dim but `axis` indexes the input, in order.
    SmallVector<AffineExpr> exprs;
    for (int64_t i = 0; i < rank; ++i)
      if (i != axis)
        exprs.push_back(getAffineDimExpr(i, ctx));

    auto inMap = AffineMap::get(rank, /*symbolCount=*/0, exprs, ctx);
    rewriter.replaceOp(op, buildCopyGeneric(rewriter, loc, input, resultType,
                                            inMap, "tt.expand_dims"));
    return success();
  }
};

/// A4. tt.broadcast → linalg.generic with a constant-0 operand map
///   tensor<1x8xf32>  →  tensor<4x8xf32>  (expand size-1 dims)
///
/// The `0` in the operand map IS the broadcast: as the loop sweeps an expanded
/// dim, the operand is read at index 0 every time, so the single input row is
/// reused across the whole extent. That is the same access pattern
/// `linalg.broadcast` describes, stated as a map instead of an op identity --
/// and unlike the named op it is accepted by the backend's compute allowlist.
struct ConvertTTBroadcast
    : public OpConversionPattern<triton::BroadcastOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::BroadcastOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    MLIRContext *ctx = rewriter.getContext();
    Value input = adaptor.getSrc();
    auto inputType = cast<RankedTensorType>(input.getType());
    auto resultType = cast<RankedTensorType>(op.getResult().getType());
    int64_t rank = resultType.getRank();

    // A dim the input already holds at full extent is read at the loop index;
    // a size-1 dim being expanded is read at a constant 0.
    SmallVector<AffineExpr> exprs;
    bool anyBroadcast = false;
    for (int64_t i = 0; i < rank; ++i) {
      if (inputType.getShape()[i] == 1 && resultType.getShape()[i] != 1) {
        exprs.push_back(getAffineConstantExpr(0, ctx));
        anyBroadcast = true;
      } else {
        exprs.push_back(getAffineDimExpr(i, ctx));
      }
    }

    // Nothing to expand: the broadcast was already a no-op on these shapes.
    if (!anyBroadcast) {
      rewriter.replaceOp(op, input);
      return success();
    }

    auto inMap = AffineMap::get(rank, /*symbolCount=*/0, exprs, ctx);
    rewriter.replaceOp(op, buildCopyGeneric(rewriter, loc, input, resultType,
                                            inMap, "tt.broadcast"));
    return success();
  }
};

/// A5. tt.trans → linalg.generic with a permuting operand map
///   tensor<4x8xf32>  →  tensor<8x4xf32>  (permute dims via order attr)
///
/// `order` is Triton's source-to-result permutation: `order[i]` is the input dim
/// that becomes result dim `i`. The operand map has to answer the opposite
/// question -- given a result coordinate, where in the input does it read -- so
/// its expression list is indexed by input dim and holds the result dim that
/// feeds it. That is the inverse of `order`, which is why the loop below writes
/// `exprs[order[i]] = d_i` rather than reading `order` straight into position.
///
/// Note this MATERIALIZES the transpose: the generic's result is an
/// actually-permuted tensor, and no permutation travels further down the chain.
struct ConvertTTTrans : public OpConversionPattern<triton::TransOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::TransOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    MLIRContext *ctx = rewriter.getContext();
    auto resultType = cast<RankedTensorType>(op.getResult().getType());
    int64_t rank = resultType.getRank();

    SmallVector<int64_t> perm(op.getOrder().begin(), op.getOrder().end());
    SmallVector<AffineExpr> exprs(rank);
    for (int64_t i = 0; i < rank; ++i)
      exprs[perm[i]] = getAffineDimExpr(i, ctx);

    auto inMap = AffineMap::get(rank, /*symbolCount=*/0, exprs, ctx);
    rewriter.replaceOp(op, buildCopyGeneric(rewriter, loc, adaptor.getSrc(),
                                            resultType, inMap, "tt.trans"));
    return success();
  }
};

/// A6. tt.join → tensor.expand_shape + tensor.concat
///   tensor<4x8xf32>, tensor<4x8xf32>  →  tensor<4x8x2xf32>
///   Result[..., 0] = lhs, Result[..., 1] = rhs.
struct ConvertTTJoin : public OpConversionPattern<triton::JoinOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::JoinOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto resultType = cast<RankedTensorType>(op.getResult().getType());
    int64_t rank = resultType.getRank();

    auto inputType = cast<RankedTensorType>(adaptor.getLhs().getType());
    int64_t inputRank = inputType.getRank();

    // Expand lhs and rhs: <4x8xf32> → <4x8x1xf32>
    SmallVector<int64_t> expandedShape(inputType.getShape());
    expandedShape.push_back(1);
    auto expandedType = RankedTensorType::get(
        expandedShape, inputType.getElementType());

    SmallVector<ReassociationIndices> reassoc;
    for (int64_t i = 0; i < inputRank - 1; ++i)
      reassoc.push_back({i});
    reassoc.push_back({inputRank - 1, inputRank});

    Value lhsExpanded = tensor::ExpandShapeOp::create(
        rewriter, loc, expandedType, adaptor.getLhs(), reassoc);
    Value rhsExpanded = tensor::ExpandShapeOp::create(
        rewriter, loc, expandedType, adaptor.getRhs(), reassoc);

    // Concatenate along the new last dim: <4x8x1> ++ <4x8x1> → <4x8x2>
    auto concat = tensor::ConcatOp::create(
        rewriter, loc, rank - 1,
        ValueRange{lhsExpanded, rhsExpanded});
    rewriter.replaceOp(op, concat.getResult());
    return success();
  }
};

/// A7. tt.split → tensor.extract_slice × 2
///   tensor<4x8x2xf32>  →  tensor<4x8xf32>, tensor<4x8xf32>
///   Extracts src[..., 0] and src[..., 1], then collapses the last dim.
struct ConvertTTSplit : public OpConversionPattern<triton::SplitOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::SplitOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value src = adaptor.getSrc();
    auto srcType = cast<RankedTensorType>(src.getType());
    int64_t rank = srcType.getRank();

    // Extract slice for index 0 and 1 along the last dim.
    SmallVector<OpFoldResult> offsets(rank, rewriter.getIndexAttr(0));
    SmallVector<OpFoldResult> sizes;
    for (int64_t i = 0; i < rank - 1; ++i)
      sizes.push_back(rewriter.getIndexAttr(srcType.getShape()[i]));
    sizes.push_back(rewriter.getIndexAttr(1));
    SmallVector<OpFoldResult> strides(rank, rewriter.getIndexAttr(1));

    // <4x8x1xelemtype>
    SmallVector<int64_t> sliceShape(srcType.getShape().drop_back());
    sliceShape.push_back(1);
    auto sliceType = RankedTensorType::get(
        sliceShape, srcType.getElementType());

    Value lhsSlice = tensor::ExtractSliceOp::create(
        rewriter, loc, sliceType, src, offsets, sizes, strides);

    offsets.back() = rewriter.getIndexAttr(1);
    Value rhsSlice = tensor::ExtractSliceOp::create(
        rewriter, loc, sliceType, src, offsets, sizes, strides);

    // Collapse <4x8x1> → <4x8>
    auto outType = cast<RankedTensorType>(op.getOutLHS().getType());
    SmallVector<ReassociationIndices> reassoc;
    for (int64_t i = 0; i < rank - 2; ++i)
      reassoc.push_back({i});
    reassoc.push_back({rank - 2, rank - 1});

    Value lhs = tensor::CollapseShapeOp::create(
        rewriter, loc, outType, lhsSlice, reassoc);
    Value rhs = tensor::CollapseShapeOp::create(
        rewriter, loc, outType, rhsSlice, reassoc);

    rewriter.replaceOp(op, {lhs, rhs});
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Group B — Reduction (compute along one axis, produces smaller tensor)
//
// These require extracting the combiner region, computing the identity
// element, and emitting the linalg op with the cloned combiner body.
//
//   B1. tt.reduce → linalg.reduce   reduce along axis with combiner
//
// Planned:
//   B2. tt.scan   → scf.for / linalg  prefix scan along axis
//===----------------------------------------------------------------------===//

/// B1. tt.reduce → linalg.generic that reduces the axis away
///   tensor<4x8xf32> →reduce(axis=1)→ tensor<4xf32>  (e.g. row-wise sum)
///
/// Same result shape the named `linalg.reduce` produced -- the reduced axis is
/// gone from the result, not kept as a size-1 dim -- so every consumer of a
/// tt.reduce result sees the type it saw before. What changes is only the op
/// class: a generic, whose maps state the reduction explicitly.
///
/// The loop space is the INPUT's dims, one per input coordinate, and the reduced
/// axis carries the `reduction` iterator. The output map simply omits that axis:
///
///   ins  (d0, d1) -> (d0, d1)      axis 1 is being reduced
///   outs (d0, d1) -> (d0)          d1 absent -- every d1 folds into one element
///   iterators ["parallel", "reduction"]
///
/// Emitted as a generic rather than `linalg.reduce` because a named op is a wall
/// that upstream elementwise fusion cannot cross: fusion composes indexing maps
/// and matches generic-into-generic only, so a named reduce stops the chain
/// feeding it. In a softmax that is exactly where the chain is -- the `exp`
/// feeding the sum -- so the named form blocks fusion at the point it matters.
///
/// Multi-operand (e.g. argmax): all N inputs are reduced together in a single
/// generic. tt.reduce's combiner block args are [lhs0..lhsN-1, rhs0..rhsN-1];
/// a generic's are [ins0..insN-1, outs0..outsN-1], which is the same order, so
/// the combiner region clones over with a direct 1:1 mapping.
///
/// The per-operand `linalg.fill` init is unchanged and must stay a named
/// `linalg.fill`: DropReductionInitFill finds a reduction's init by looking for
/// one on the `outs`, so converting it would silently break that match.
struct ConvertTTReduce : public OpConversionPattern<triton::ReduceOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::ReduceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    MLIRContext *ctx = op.getContext();

    int32_t axis = op.getAxis();
    unsigned numOperands = adaptor.getOperands().size();

    auto firstInputType =
        cast<RankedTensorType>(adaptor.getOperands()[0].getType());
    int64_t inputRank = firstInputType.getRank();

    // Result shape: the input's, with the reduced axis dropped.
    SmallVector<int64_t> resultShape;
    for (int64_t i = 0; i < inputRank; ++i)
      if (i != axis)
        resultShape.push_back(firstInputType.getShape()[i]);

    // One loop per input dim. The input reads every one; the output reads all
    // but the reduced axis, which is what makes that axis a reduction.
    SmallVector<AffineExpr> inExprs, outExprs;
    for (int64_t i = 0; i < inputRank; ++i) {
      inExprs.push_back(getAffineDimExpr(i, ctx));
      if (i != axis)
        outExprs.push_back(getAffineDimExpr(i, ctx));
    }
    auto inMap = AffineMap::get(inputRank, /*symbolCount=*/0, inExprs, ctx);
    auto outMap = AffineMap::get(inputRank, /*symbolCount=*/0, outExprs, ctx);

    SmallVector<utils::IteratorType> iterators(inputRank,
                                              utils::IteratorType::parallel);
    iterators[axis] = utils::IteratorType::reduction;

    // A generic takes its maps in ins-then-outs order, and every input here
    // shares the same access pattern, as does every output.
    SmallVector<AffineMap> maps(numOperands, inMap);
    maps.append(numOperands, outMap);

    // Build one (empty, fill) init tensor per operand.
    // Neutral element: use the combiner op that produces the i-th yield value.
    Block &combinerBlock = op.getCombineOp().front();
    Operation *terminator = combinerBlock.getTerminator();

    SmallVector<Value> inputs(adaptor.getOperands().begin(),
                              adaptor.getOperands().end());
    SmallVector<Value> inits;
    SmallVector<Type> resultTypes;
    for (unsigned i = 0; i < numOperands; ++i) {
      auto inputType = cast<RankedTensorType>(inputs[i].getType());
      Type elemType = inputType.getElementType();

      Value termYield = terminator->getOperand(i);
      Operation *combinerOp = termYield.getDefiningOp();
      auto neutralAttr = getReductionNeutralAttr(combinerOp, elemType, ctx);
      if (!neutralAttr)
        return failure();
      TypedAttr attr = *neutralAttr;
      Value fillVal = arith::ConstantOp::create(rewriter, loc, attr);
      auto emptyOp =
          tensor::EmptyOp::create(rewriter, loc, resultShape, elemType);
      inits.push_back(
          linalg::FillOp::create(rewriter, loc, fillVal, emptyOp.getResult())
              .getResult(0));
      resultTypes.push_back(RankedTensorType::get(resultShape, elemType));
    }

    // Snapshot the combiner block contents before creating the generic.
    SmallVector<Operation *> combinerOps;
    for (auto &innerOp : combinerBlock.without_terminator())
      combinerOps.push_back(&innerOp);

    SmallVector<Value> yieldOperands(terminator->operand_begin(),
                                     terminator->operand_end());

    auto generic = linalg::GenericOp::create(
        rewriter, loc, resultTypes, inputs, inits, maps, iterators,
        [&](OpBuilder &b, Location loc, ValueRange args) {
          IRMapping mapping;
          mapping.map(combinerBlock.getArguments(), args);
          for (auto *innerOp : combinerOps)
            b.clone(*innerOp, mapping);
          SmallVector<Value> mapped;
          for (Value v : yieldOperands)
            mapped.push_back(mapping.lookup(v));
          linalg::YieldOp::create(b, loc, mapped);
        });
    generic.setDocAttr(rewriter.getStringAttr("tt.reduce"));

    // A rank-0 tt.reduce result is a bare scalar, so extract it from the
    // rank-0 tensor the generic produced.
    SmallVector<Value> results;
    for (unsigned i = 0; i < numOperands; ++i) {
      Value r = generic.getResult(i);
      if (resultShape.empty())
        r = tensor::ExtractOp::create(rewriter, loc, r, ValueRange{});
      results.push_back(r);
    }
    rewriter.replaceOp(op, results);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Group C — Matrix multiply
//
//   C1. tt.dot → linalg.matmul   d = matmul(a, b) + c
//===----------------------------------------------------------------------===//

/// C1. tt.dot → linalg.generic stating the contraction
///   2-D: tensor<M×K> @ tensor<K×N> + tensor<M×N> → tensor<M×N>
///   3-D: tensor<B×M×K> @ tensor<B×K×N> + tensor<B×M×N> → tensor<B×M×N>
///
/// The loop order is (batch...,  m, n, k), with `k` the one reduction. Each
/// operand's map picks the dims it is indexed by:
///   A   (m, n, k) -> (m, k)
///   B   (m, n, k) -> (k, n)
///   C   (m, n, k) -> (m, n)      -- and the result, same map
/// `n` is absent from A's map and `m` from B's, which is what makes this a
/// contraction rather than an elementwise op; `k` is absent from C's, which is
/// the dim being summed over. Upstream's `inferContractionDims` recovers exactly
/// these roles from these maps, so the emission states the same fact the named
/// `linalg.matmul` implied by its identity -- but as data the next pass can read.
///
/// Emitted as a generic because the named form cannot reach a binary: the
/// backend's compute allowlist is add/mul/sub/reduce, and a named
/// `linalg.matmul` is rejected outright.
struct ConvertTTDot : public OpConversionPattern<triton::DotOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::DotOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    MLIRContext *ctx = rewriter.getContext();

    auto aType = cast<RankedTensorType>(adaptor.getA().getType());
    auto resultType = cast<RankedTensorType>(op.getResult().getType());
    Type elemType = resultType.getElementType();

    // One leading batch dim per rank above 2, then m, n, k.
    int64_t numBatch = aType.getRank() - 2;
    int64_t numLoops = numBatch + 3;
    int64_t mPos = numBatch, nPos = numBatch + 1, kPos = numBatch + 2;

    SmallVector<AffineExpr> batch;
    for (int64_t i = 0; i < numBatch; ++i)
      batch.push_back(getAffineDimExpr(i, ctx));
    AffineExpr m = getAffineDimExpr(mPos, ctx);
    AffineExpr n = getAffineDimExpr(nPos, ctx);
    AffineExpr k = getAffineDimExpr(kPos, ctx);

    auto mapOf = [&](AffineExpr x, AffineExpr y) {
      SmallVector<AffineExpr> exprs(batch);
      exprs.push_back(x);
      exprs.push_back(y);
      return AffineMap::get(numLoops, /*symbolCount=*/0, exprs, ctx);
    };
    SmallVector<AffineMap> maps = {mapOf(m, k), mapOf(k, n), mapOf(m, n)};

    SmallVector<utils::IteratorType> iterators(numLoops,
                                              utils::IteratorType::parallel);
    iterators[kPos] = utils::IteratorType::reduction;

    // tt.dot accumulates into C, so the body reads the running value from the
    // `outs` block argument rather than starting from a neutral element.
    //
    // A mixed-precision dot -- f16 inputs into an f32 accumulator, which is the
    // common case -- needs each operand widened to the accumulator's type before
    // the multiply, or the product is narrower than the accumulator and the add
    // fails its same-type constraint. The named `linalg.matmul` inserted that
    // extension for us; stating the body explicitly means stating it too.
    bool isFloat = isa<FloatType>(elemType);
    auto generic = linalg::GenericOp::create(
        rewriter, loc, TypeRange{resultType},
        ValueRange{adaptor.getA(), adaptor.getB()},
        ValueRange{adaptor.getC()}, maps, iterators,
        [&](OpBuilder &b, Location loc, ValueRange args) {
          // Widen an operand to the accumulator type. Signed extension for
          // integers: Triton's dot operands are signed.
          auto widen = [&](Value v) -> Value {
            if (v.getType() == elemType)
              return v;
            if (isFloat)
              return arith::ExtFOp::create(b, loc, elemType, v).getResult();
            return arith::ExtSIOp::create(b, loc, elemType, v).getResult();
          };
          Value lhs = widen(args[0]);
          Value rhs = widen(args[1]);
          Value prod =
              isFloat ? arith::MulFOp::create(b, loc, lhs, rhs).getResult()
                      : arith::MulIOp::create(b, loc, lhs, rhs).getResult();
          Value sum =
              isFloat ? arith::AddFOp::create(b, loc, args[2], prod).getResult()
                      : arith::AddIOp::create(b, loc, args[2], prod).getResult();
          linalg::YieldOp::create(b, loc, sum);
        });
    generic.setDocAttr(rewriter.getStringAttr("tt.dot"));
    rewriter.replaceOp(op, generic.getResult(0));
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Pass
//===----------------------------------------------------------------------===//

struct LowerComputeOpsPass
    : public mlir::triton::ktdp::impl::LowerComputeOpsBase<
          LowerComputeOpsPass> {

  void runOnOperation() override {
    ModuleOp module = getOperation();
    MLIRContext *ctx = &getContext();

    ConversionTarget target(*ctx);
    // Illegal: Triton compute ops that have conversion patterns below.
    //   Group A (shape): splat, reshape, expand_dims, broadcast, trans, join, split
    //   Group B (reduce): reduce + reduce.return (region replaced wholesale)
    //   Group C (matmul): dot
    target.addIllegalOp<triton::SplatOp, triton::ReshapeOp,
                        triton::ExpandDimsOp, triton::BroadcastOp,
                        triton::TransOp, triton::JoinOp, triton::SplitOp,
                        triton::ReduceOp, triton::ReduceReturnOp,
                        triton::DotOp>();
    // Legal: output dialects that conversion patterns lower into.
    //   linalg (generic for every compute op; fill for splat and reduce inits),
    //   tensor (reshape, expand_shape, collapse_shape, extract_slice, concat, empty),
    //   arith/math (constants, index casts, cloned combiner body ops)
    target.addLegalDialect<linalg::LinalgDialect, tensor::TensorDialect,
                           arith::ArithDialect, math::MathDialect>();
    target.addLegalOp<ModuleOp, UnrealizedConversionCastOp,
                      triton::SpyreTensorLayoutOp>();

    RewritePatternSet patterns(ctx);
    patterns.add<ConvertTTSplat, ConvertTTReshape,          // Group A
                 ConvertTTExpandDims, ConvertTTBroadcast,
                 ConvertTTTrans, ConvertTTJoin, ConvertTTSplit,
                 ConvertTTReduce,                          // Group B
                 ConvertTTDot>(ctx);                       // Group C

    if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
      module.emitError("LowerComputeOps: failed to convert compute ops");
      signalPassFailure();
      return;
    }

    mlir::triton::ktdp::cleanupDeadOps(module);
  }
};

} // namespace

namespace mlir::triton::ktdp {
std::unique_ptr<OperationPass<ModuleOp>> createLowerComputeOpsPass() {
  return std::make_unique<LowerComputeOpsPass>();
}
} // namespace mlir::triton::ktdp

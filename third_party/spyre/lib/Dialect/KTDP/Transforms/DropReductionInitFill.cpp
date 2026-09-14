//===- DropReductionInitFill.cpp - Drop a reduction's init fill -----------===//
//
// Removes the `linalg.fill` that supplies the `outs` of a reduction, repointing
// that `outs` at the `tensor.empty` the fill wrote into.
//
// Why this is needed:
//   tt.reduce lowers (LowerComputeOps) to a linalg.reduce whose `outs` is
//   tensor.empty + linalg.fill of the combiner's neutral element, which is what
//   upstream linalg semantics call for. The Spyre dataflow-scheduler will not
//   take it: KTIRLegalityCheck's named-op allowlist is add/mul/sub/reduce, so
//   the fill is rejected outright, and with that check widened the fill is
//   generalized into a second linalg.generic and trips
//   ConstructThreeStagePipeline's one-compute-op-per-group assertion (the fill
//   feeds an init operand, so the existing elementwise fusion — which only walks
//   `ins` — never absorbs it). Hand-written reference KTIR states a bare
//   tensor.empty for exactly this reason.
//
// Why the rewrite is sound:
//   A reduction's payload READS its init operand — `linalg.reduce` names the
//   operand `$inits` and its own ODS example writes `arith.addf %out, %in` — and
//   `tensor.empty` has explicitly "unspecified" contents. So the rewritten IR is
//   only well-defined because something downstream restores the accumulator
//   before it is read: MapReductionPartials, which derives the neutral element
//   from the combiner and emits its own `linalg.fill` of it. The rewrite is
//   therefore sound when that pass will run on this op.
//
//   Two structural gates, and deliberately NO judgement about the combiner:
//
//     isMapReductionPartialsShape  one `ins`, one `init` — mirroring that pass's
//                                  own asserts. Excludes MATMUL above all: a
//                                  contraction is an addf-accumulate reduction,
//                                  but MapReductionPartials never rewrites a
//                                  matmul, so nothing would restore its init and
//                                  the fill is load-bearing.
//     simpleReductionPayload       body is exactly payload + yield, init read by
//                                  the payload — the shape that pass clones and
//                                  LinalgLowering maps to one vectorchain op.
//
//   Failing either means the fill is load-bearing, so it is LEFT ALONE and no
//   diagnostic is emitted — this pass is not responsible for ops it cannot reason
//   about, and failing on them would make any pipeline that merely *contains* a
//   matmul unable to run this fix.
//
// Why nothing here judges the combiner or the fill's value:
//   An earlier version gated on an allowlist of `addf`/`subf` and required the fill
//   to state zero, on the premise that MapReductionPartials' reset was "a hardcoded
//   0.0". That premise was FALSE: the scheduler's `getNeutralAttr` derives the
//   neutral per combiner (mulf 1.0, maximumf -inf, minimumf +inf, integer addi/subi
//   0, muli 1) and fills that value back. The allowlist was refusing reductions the
//   scheduler handles, and *erroring* on them, so a max reduce could not get
//   through this pass at all.
//
//   The replacement is not a copy of that table. Downstream's gates keep moving --
//   which combiners it names, which it lowers to what -- and a copy here would go
//   out of step silently, since a drift shows up as a wrong number rather than a
//   build break. Every such condition is downstream's to state, and it states them
//   with diagnostics naming the op. This pass answers one question instead: is
//   there a `linalg.fill` on a reduction's `outs` that the scheduler will not
//   accept? If so it goes, and what the scheduler then makes of the reduction is
//   the scheduler's business.
//
//   The consequence to accept: a fill stating something other than the neutral is
//   dropped too, and a reduction that meant "sum, plus 2.5" loses the 2.5. Nothing
//   in the pipeline emits that today -- LowerComputeOps fills the combiner's
//   neutral by construction -- and the alternative is this pass carrying a model of
//   downstream's reset semantics, which is the thing that was wrong before.

// Algorithm:
//   1. Collect linalg ops that have at least one reduction iterator
//      (collect-then-rewrite, so erasing fills cannot invalidate the walk).
//      Ops with no reduction loop are left alone, which is what keeps the other
//      producer of linalg.fill in this pipeline — tt.splat — out of scope.
//   2. Skip any op that is not MapReductionPartials-shaped.
//   3. For each remaining `outs` operand defined by a linalg.fill whose own
//      output is a tensor.empty:
//      a. Skip    — the body is not a simple reduction.
//      b. Rewrite — point the operand at the tensor.empty, and erase the fill
//                   if nothing else uses it. No other condition is consulted.
//
//===----------------------------------------------------------------------===//

#include "Dialect/KTDP/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;

namespace mlir::triton::ktdp {
#define GEN_PASS_DEF_DROPREDUCTIONINITFILL
#include "Dialect/KTDP/Transforms/Passes.h.inc"
} // namespace mlir::triton::ktdp

namespace {

/// True iff `op` has the shape MapReductionPartials actually handles, and will
/// therefore get its accumulator restored by that pass's own fill of the
/// combiner's neutral element.
///
/// This is the whole soundness argument, so the check mirrors that pass's own
/// preconditions rather than approximating them: it asserts a single `ins` and a
/// single `init`.
///
/// The single-input condition is what excludes a **matmul**, and it is the only
/// thing that does: a contraction looks exactly like an `addf`-accumulate
/// reduction filled with its own neutral, so nothing about the fill itself
/// distinguishes it — but MapReductionPartials never rewrites a matmul, so
/// nothing would restore that accumulator and the fill is load-bearing. Same for
/// any other multi-operand reduction (argmax and friends, which carry an index
/// lane).
bool isMapReductionPartialsShape(linalg::LinalgOp op) {
  return op.getNumDpsInputs() == 1 && op.getNumDpsInits() == 1;
}

/// The payload op of a *simple* reduction body: the one op computing the yielded
/// value, when the body is exactly that op plus the yield, and the init block
/// argument is one of its operands. Null otherwise.
///
/// This is `linalg.reduce`'s "shortened print form" shape, and it is what
/// MapReductionPartials + LinalgLowering handle — they clone the region wholesale
/// into a buffer-semantics generic and map the payload onto a single
/// `vectorchain` binary op. A body doing anything else is not our business.
Operation *simpleReductionPayload(linalg::LinalgOp op, OpOperand &init) {
  Block *body = op.getBlock();
  if (!body || body->getOperations().size() != 2)
    return nullptr;
  auto yield = dyn_cast<linalg::YieldOp>(body->getTerminator());
  if (!yield || yield->getNumOperands() != 1)
    return nullptr;
  Operation *payload = yield->getOperand(0).getDefiningOp();
  if (!payload || payload->getBlock() != body)
    return nullptr;
  // The init must actually be read. If it is not, there is no init to discard
  // and this pass has nothing to do.
  Value initArg = op.getMatchingBlockArgument(&init);
  if (!llvm::is_contained(payload->getOperands(), initArg))
    return nullptr;
  return payload;
}

struct DropReductionInitFillPass
    : public mlir::triton::ktdp::impl::DropReductionInitFillBase<
          DropReductionInitFillPass> {
  void runOnOperation() override {
    ModuleOp mod = getOperation();
    IRRewriter rewriter(&getContext());

    // Collect first: the rewrite erases fills, which would invalidate a walk in
    // progress. Reductions only — see the header on tt.splat.
    SmallVector<linalg::LinalgOp> reductions;
    mod.walk([&](linalg::LinalgOp op) {
      if (op.getNumReductionLoops() > 0)
        reductions.push_back(op);
    });

    for (auto op : reductions)
      dropOne(op, rewriter);
  }

  /// Drops every `linalg.fill` feeding an `outs` operand of `op`.
  ///
  /// There is exactly one kind of non-rewrite, and it is silent: `op` is not a
  /// reduction MapReductionPartials will ever touch (matmul or another
  /// multi-operand contraction, a multi-result reduction, a non-trivial body).
  /// The fill is load-bearing there, because nothing downstream restores the
  /// accumulator, so leaving it is the correct and conservative answer.
  /// Diagnosing it is not this pass's job either: whatever cannot lower it will
  /// say so, and failing here would make any pipeline that merely *contains* a
  /// matmul unable to run this fix.
  ///
  /// This pass emits no diagnostics and cannot fail, hence the void return —
  /// every condition it once rejected on belonged downstream. See the header.
  void dropOne(linalg::LinalgOp op, IRRewriter &rewriter) {
    // Not a shape MapReductionPartials rewrites -> nothing restores the
    // accumulator downstream -> the fill must stay. This is the matmul case.
    if (!isMapReductionPartialsShape(op))
      return;

    for (OpOperand &out : op.getDpsInitsMutable()) {
      auto fill = out.get().getDefiningOp<linalg::FillOp>();
      if (!fill)
        continue;

      // The fill must be writing into a fresh tensor, not over live data:
      // repointing `outs` at its output substitutes that output's contents for
      // the stated init, and only tensor.empty makes that a no-op.
      if (!fill.getOutputs()[0].getDefiningOp<tensor::EmptyOp>())
        continue;

      // A body this pass does not recognise as a simple reduction is left alone
      // for the same reason as the matmul case above.
      if (!simpleReductionPayload(op, out))
        continue;

      out.set(fill.getOutputs()[0]);
      // Only this reduction used it in the pipeline's own output, but a fill is
      // a normal value and something else may hold it.
      if (fill->use_empty())
        rewriter.eraseOp(fill);
    }
  }
};

} // namespace

namespace mlir::triton::ktdp {

std::unique_ptr<OperationPass<ModuleOp>> createDropReductionInitFillPass() {
  return std::make_unique<DropReductionInitFillPass>();
}

} // namespace mlir::triton::ktdp

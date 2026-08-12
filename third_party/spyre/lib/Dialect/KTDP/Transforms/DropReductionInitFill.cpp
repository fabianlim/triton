//===- DropReductionInitFill.cpp - Drop a zero reduction init fill --------===//
//
// Removes a `linalg.fill` of zero that supplies the `outs` of a reduction,
// repointing that `outs` at the `tensor.empty` the fill wrote into.
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
// Why the rewrite is sound, and why it is gated on ZERO rather than on the
// combiner's neutral element:
//   A reduction's payload reads its init operand, and tensor.empty has
//   unspecified contents, so dropping the fill would change the result were it
//   not for the scheduler emitting its own accumulator reset ahead of the
//   reduction loop. That reset is hardcoded to zero
//   (MapReductionPartials: "Step 2: zero-fill the accumulator"), and the init
//   the KTIR states is discarded either way. So this pass is sound exactly when
//   the fill value is the value the scheduler will write anyway: zero.
//
//   A non-zero fill — 1.0 for a mulf combiner, -inf for a max — is therefore
//   NOT dropped. It is reported, because both alternatives are wrong: dropping
//   it discards a stated init, and passing it through hits the assertion above.
//   Such a reduction cannot currently be compiled correctly at all, since the
//   scheduler would reset it to zero regardless of what this pass does. The
//   diagnostic is the only honest outcome, and it belongs here rather than as
//   a wrong answer several passes later.
//
// Algorithm:
//   1. Collect linalg ops that have at least one reduction iterator
//      (collect-then-rewrite, so erasing fills cannot invalidate the walk).
//      Ops with no reduction loop are left alone, which is what keeps the other
//      producer of linalg.fill in this pipeline — tt.splat — out of scope.
//   2. For each such op, for each `outs` operand defined by a linalg.fill whose
//      own output is a tensor.empty:
//      a. Reject  — the fill value is not a constant zero.
//      b. Rewrite — point the operand at the tensor.empty, and erase the fill
//                   if nothing else uses it.
//
//===----------------------------------------------------------------------===//

#include "Dialect/KTDP/Transforms/Passes.h"

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

/// True iff `v` is defined by a constant whose value is zero, of either a float
/// or an integer type. m_AnyZeroFloat accepts -0.0 as well as +0.0; both are
/// additive identities, and both are what the scheduler's reset writes.
bool isConstantZero(Value v) {
  return matchPattern(v, m_AnyZeroFloat()) || matchPattern(v, m_Zero());
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

    // Keep going after a rejection so one run reports every fill it cannot
    // handle, rather than costing the user one recompile per reduction.
    bool anyFailed = false;
    for (auto op : reductions) {
      if (failed(dropOne(op, rewriter)))
        anyFailed = true;
    }
    if (anyFailed)
      signalPassFailure();
  }

  /// Drops every zero `linalg.fill` feeding an `outs` operand of `op`. Returns
  /// failure if any such fill is not a constant zero, after emitting one
  /// diagnostic per offending operand. Operands that can be rewritten are
  /// rewritten even when a sibling is rejected, so the IR is left partially
  /// modified on failure.
  LogicalResult dropOne(linalg::LinalgOp op, IRRewriter &rewriter) {
    LogicalResult result = success();
    for (OpOperand &out : op.getDpsInitsMutable()) {
      auto fill = out.get().getDefiningOp<linalg::FillOp>();
      if (!fill)
        continue;

      // The fill must be writing into a fresh tensor, not over live data:
      // repointing `outs` at its output substitutes that output's contents for
      // the stated init, and only tensor.empty makes that a no-op.
      if (!fill.getOutputs()[0].getDefiningOp<tensor::EmptyOp>())
        continue;

      if (!isConstantZero(fill.getInputs()[0])) {
        op->emitError("reduction 'outs' operand #")
            << out.getOperandNumber()
            << " is initialised by a linalg.fill of a non-zero value; the "
               "dataflow-scheduler rejects linalg.fill and resets a reduction "
               "accumulator to zero regardless of the combiner, so this "
               "reduction cannot be lowered without discarding its stated "
               "initial value";
        result = failure();
        continue;
      }

      out.set(fill.getOutputs()[0]);
      // Only this reduction used it in the pipeline's own output, but a fill is
      // a normal value and something else may hold it.
      if (fill->use_empty())
        rewriter.eraseOp(fill);
    }
    return result;
  }
};

} // namespace

namespace mlir::triton::ktdp {

std::unique_ptr<OperationPass<ModuleOp>> createDropReductionInitFillPass() {
  return std::make_unique<DropReductionInitFillPass>();
}

} // namespace mlir::triton::ktdp
